// coarse
constexpr float RES_GLOBAL = 0.4f; 
constexpr int GRID_LEVEL_COUNT = 2;
constexpr int ITERATION_COUNT = 20000; 

constexpr int PLOTTER_PERIOD = 1000;

constexpr int WALL_REFINEMENT_COUNT = 5;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;	// kg/m3 water
constexpr float NU_PHYS = 1e-5;		// m2/s water

constexpr float YMIN = -20.f;
constexpr float YMAX = 20.f;

constexpr float uzInlet = 0.01f; 															// also works as nominal LBM Mach number	
constexpr float uzInletMax = 0.05f;
constexpr float uzInletPhys = 2.f;															// m/s

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPath = "shearFlowTestSTL.STL";

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
		if ( rz < 10.f && z > 25.f && z < 75.f ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	BC.uz = uzInlet + ( y - YMIN ) / ( YMAX - YMIN ) * ( uzInletMax - uzInlet );
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	if ( kCell == 0 ) // inlet
	{
		BC.openBCID = 0;
		BC.dirichletU = true;
		BC.nonReflective = false;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzInlet + ( y - YMIN ) / ( YMAX - YMIN ) * ( uzInletMax - uzInlet );
	}
	else if ( kCell == Info.cellCountZ-1 ) // outlet
	{
		BC.openBCID = 1;
		BC.dirichletRho = true;
		BC.nonReflective = true;
		BC.dRho = 0.f;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	if ( BC.wallID == 0 ) // pipe wall
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = BC.uz = uzInlet + ( y - YMIN ) / ( YMAX - YMIN ) * ( uzInletMax - uzInlet );
	}
	BC.collisionLimiter = 0.f;
	BC.overwriteIBBLinks = 0.5f;
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
	readSTL( gridStaticSTLs[0], STLPath );
	
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
