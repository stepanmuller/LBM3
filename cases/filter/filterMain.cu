constexpr float litersPerMinute = 1.f;

// coarse settings
constexpr float RES_GLOBAL = 0.1f;
constexpr float uzInlet = 0.04; 
constexpr int ITERATION_COUNT = 10000;

constexpr int PLOTTER_PERIOD = 1000;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float inletAreaM2 = 3.14159265358979324f * 0.0102f * 0.0102f;
constexpr float uzInletPhys = 0.001f * ( litersPerMinute / 60.f ) / inletAreaM2; 					// m/s, physical velocity

constexpr int GRID_LEVEL_COUNT = 2;
constexpr int WALL_REFINEMENT_COUNT = 6;

constexpr float NU_PHYS = 1.5e-5f;														// m2/s
constexpr float RHO_PHYS = 1.225f;														// kg/m3
constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000); 			// s

#include "../../include/types.h"

std::string STLPathCylinder = "cylinder_102_mm.stl";
std::string STLPathFilter = "filter_mm.stl";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	if ( Info.gridID == 0 )
	{
		refinementMarker = false;
		if ( z > -16.f && z < 16.f ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	BC.uz = uzInlet; 
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( kCell == 0 ) // Inlet
	{
		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzInlet;
		BC.nonReflective = true;
		BC.openBCID = 0;
	}
	else if ( kCell == Info.cellCountZ-1  ) // Outlet
	{
		BC.dirichletRho = true;
		BC.dRho = 0.f;
		BC.nonReflective = true;
		BC.openBCID = 1;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( Info.gridID == 0 ) BC.collisionLimiter = 0.f;
	else BC.collisionLimiter = 0.01f;
	if ( BC.wallID >= 0 ) 
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
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
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 2 );
	readSTL( gridStaticSTLs[0], STLPathCylinder );
	readSTL( gridStaticSTLs[1], STLPathFilter );
	
	std::vector<STLStruct> rotorSTLs( 0 ); // there are no rotors
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	TrackerStruct Tracker;
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
	
	return EXIT_SUCCESS;
}
