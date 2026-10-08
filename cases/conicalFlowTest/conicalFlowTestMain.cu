// coarse
constexpr float RES_GLOBAL = 0.5f; 
constexpr int GRID_LEVEL_COUNT = 2;
constexpr int ITERATION_COUNT = 100; 

constexpr int PLOTTER_PERIOD = 20;

constexpr int WALL_REFINEMENT_COUNT = 3;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;	// kg/m3 water
constexpr float NU_PHYS = 1e-6;		// m2/s water

constexpr float uzInlet = 0.01f; 															// also works as nominal LBM Mach number	
constexpr float uzInletPhys = 10.f;															// m/s

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathCone = "coneD50ToD25.stl";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float rz = std::sqrt( x*x + y*y );
	if ( Info.gridID == 0 )
	{
		refinementMarker = false;
		if ( z > 20.f ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	return; // this leaves default zero velocity, zero pressure
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( kCell == 0 ) // inlet
	{
		BC.openBCID = 0;
		BC.dirichletU = true;
		BC.nonReflective = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzInlet;
	}
	else if ( kCell == Info.cellCountZ-1 ) // outlet
	{
		BC.openBCID = 1;
		BC.dirichletRho = true;
		BC.nonReflective = false;
		BC.dRho = 0.f;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	if ( BC.wallID == 0 ) // cone wall
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	BC.collisionLimiter = 0.01f;
	const float distanceFromBoundary = TNL::min( z - Info.Bounds.zMin, Info.Bounds.zMax - z );
	if ( distanceFromBoundary < 5.f ) BC.collisionLimiter = 0.f;
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	const float xCut = 0.f;
	exportSectionCutPlotZY( grids, xCut, iterationsFinished );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathCone );
	
	std::vector<STLStruct> rotorSTLs( 0 );
	
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
