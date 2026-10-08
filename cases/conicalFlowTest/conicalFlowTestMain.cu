// coarse
constexpr float RES_GLOBAL = 1.0f; 
constexpr int GRID_LEVEL_COUNT = 2;
constexpr int ITERATION_COUNT = 1000; 

constexpr int PLOTTER_PERIOD = 1000;

constexpr int WALL_REFINEMENT_COUNT = 3;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = false;

constexpr float RHO_PHYS = 997.0f;	// kg/m3 water
constexpr float NU_PHYS = 1e-6;		// m2/s water

constexpr float uzInlet = 0.01f; 															// also works as nominal LBM Mach number	
constexpr float massFlowPhys = 335.f;														// kg/s
constexpr float RInlet = 150.f;																// mm
constexpr float inletAreamm2 = 3.14159f * RInlet * RInlet;									// mm2
constexpr float uzInletPhys = massFlowPhys / ( RHO_PHYS * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float radiansPerSecond = -198.967f;												// rad/s

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathStator = "TORQUE_TEST_STATOR.STL";
std::string STLPathRotor = "TORQUE_TEST_ROTOR.STL";

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
		float zMin = -17.f;
		float zMax = 3.f;
		float rzMax = 164.f;
		float rzMin = 0.f;
		refinementMarker = false;
		if ( rz > rzMin && rz < rzMax && z < zMax && z > zMin ) refinementMarker = true;
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
	return; // there are no open boundaries
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = radiansPerSecond * (r / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( BC.wallID == 0 ) // stator
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( BC.wallID == 1 ) // rotor
	{
		BC.ux = - vt * (y / r);
		BC.uy = vt * (x / r);
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
	int iCut, jCut, kCut;
	const float yTemp = 0.f;
	// ZY section cut shows the inlet pipe
	float xCut = 0.f; float zCut = -3.f;
	getIJKCellIndexFromXYZ( iCut, jCut, kCut, xCut, yTemp, zCut, grids[GRID_LEVEL_COUNT-1].Info);
	exportSectionCutPlotZY( grids, iCut, iterationsFinished );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	exportSectionCutPlotXY( grids, kCut, iterationsFinished+1 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 2 );
	readSTL( gridStaticSTLs[0], STLPathStator );
	readSTL( gridStaticSTLs[1], STLPathRotor );
	
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
