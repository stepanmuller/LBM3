constexpr float reynoldsNumber = 100.f; // based on mean velocity and pipe diameter

constexpr int GRID_LEVEL_COUNT = 1;
constexpr float RES_GLOBAL = 0.2f; // mm

constexpr float uzInlet = 0.01f; // mean lattice velocity; centreline velocity is twice this
constexpr int ITERATION_COUNT = 40000;
constexpr int PLOTTER_PERIOD = 40000;

constexpr int WALL_REFINEMENT_COUNT = 0;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float pipeDiameterPhys = 20.f; // mm
constexpr float pipeLengthPhys = 20.f; 	 // mm
constexpr float RInletPhys = 0.5f * pipeDiameterPhys; // mm
constexpr float NU_PHYS = 1.5e-5f; // m2/s, air
constexpr float RHO_PHYS = 1.225f; // kg/m3, air
constexpr float uzInletPhys = reynoldsNumber * NU_PHYS / (pipeDiameterPhys / 1000.f); // m/s, mean velocity
constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL / 1000.f); // s

#include "../../include/types.h"

std::string STLPathPipe = "pipeD20L200STL.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( const int& iCell, const int& jCell, const int& kCell,
											bool &refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r2 = x*x + y*y;
	const float refinementR2 = ( 0.5f * RInletPhys ) *  ( 0.5f * RInletPhys );
	const float refinementZ = pipeLengthPhys * 0.25f;
	refinementMarker = false;
	if ( Info.gridID == 0 )
	{
		if ( r2 < refinementR2 && z > - refinementZ && z < refinementZ ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
										const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r2 = x * x + y * y;
	BC.uz = 2.f * uzInlet * std::max( 0.f, 1.f - r2 / (RInletPhys * RInletPhys) );
}

__cuda_callable__ void getOpenBC( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
								const InfoStruct& Info )
{
	if ( kCell == 0 ) // Inlet
	{
		float x, y, z;
		getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
		const float r2 = x * x + y * y;

		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 2.f * uzInlet * std::max( 0.f, 1.f - r2 / (RInletPhys * RInletPhys) );
		BC.nonReflective = false;
		BC.openBCID = 0;
	}
	else if ( kCell == Info.cellCountZ-1 ) // Outlet
	{
		BC.dirichletRho = true;
		BC.dRho = 0.f;
		BC.nonReflective = true;
		BC.openBCID = 1;
	}
}

__cuda_callable__ void getLocalBC( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
								const InfoStruct& Info )
{
	BC.collisionLimiter = 0.f; // use K15
	if ( BC.wallID >= 0 )
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	BC.useIBB = false;
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	// ZY section cut
	const float xCut = 0.f;
	exportSectionCutPlotZY( grids, xCut, iterationsFinished );
	if (system("python3 plotGridsFullCustom.py") != 0) {}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathPipe );
	std::vector<STLStruct> rotorSTLs( 0 );

	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMin = - pipeLengthPhys*0.5f;
	DomainBounds.zMax = pipeLengthPhys*0.5f;

	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );

	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = false;
	initializeTracker( Tracker, grids );

	plotGrids( 0, grids );

	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();

	for ( int iterationsFinished = 1; iterationsFinished <= ITERATION_COUNT; iterationsFinished++ )
	{
		updateAllGrids( grids, 0 );

		if ( iterationsFinished % TRACKER_PERIOD == 0 )
		{
			updateTracker( Tracker, grids );
		}

		if ( iterationsFinished % PLOTTER_PERIOD == 0 )
		{
			lapTimer.stop();
			std::cout << "Iterations finished: " << iterationsFinished << std::endl;
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)fluidUpdatesPerIteration * (float)PLOTTER_PERIOD;
			const float glups = updateCount / lapTime / 1000000000.f;
			std::cout << "GLUPS: " << glups << std::endl;

			plotTracker( Tracker );
			plotGrids( iterationsFinished, grids );

			lapTimer.reset();
			lapTimer.start();
		}
	}
	
	// Final ZY section cut for centerline pressure.
	const float xCut = 0.f;
	exportSectionCutPlotZY(grids, xCut, ITERATION_COUNT);
	const auto& imageInfo = grids.back().Info;
	std::ostringstream command;
	command << "python3 exportCenterlinePressure.py "
			<< imageInfo.oz << ' ' << imageInfo.res << ' '
			<< imageInfo.cellCountY << ' ' << imageInfo.cellCountZ;

	if (system(command.str().c_str()) != 0) {}
	
	return EXIT_SUCCESS;
}
