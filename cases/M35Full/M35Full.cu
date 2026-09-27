constexpr float RES_GLOBAL = 1.f; 
constexpr float uzInlet = 0.01f; 
constexpr int GRID_LEVEL_COUNT = 4;
constexpr int ITERATION_COUNT = 60000; 												

constexpr int PLOTTER_PERIOD = 1000;

constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = true;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;							// kg/m3 water
constexpr float NU_PHYS = 1e-6;								// m2/s water

constexpr float uzInletPhys = 16.25f;						// m/s
constexpr float radiansPerSecond = 2700.f;					// rad/s

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathIntake = "../../../BruteforceOptimizer/M35Geometry/M35IntakeSTL.STL";
std::string STLPathPump = "../../../BruteforceOptimizer/M35Geometry/M35PumpSTL.STL";
std::string STLPathImpeller = "../../../BruteforceOptimizer/M35Geometry/M35ImpellerSTL.STL";
std::string STLPathShaft = "../../../BruteforceOptimizer/M35Geometry/M35ShaftSTL.STL";

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
		if ( x > -50.f && x < 50.f && y > -60.f ) refinementMarker = true;
	}
	if ( Info.gridID == 1 )
	{
		if ( x < -40.f || x > 40.f ) refinementMarker = false;
		if ( x > -25.f && x < 25.f && y > -30.f ) refinementMarker = true;
	}
	if ( Info.gridID == 2 )
	{
		refinementMarker = false;
		if ( rz < 25.f && z > -11.f ) refinementMarker = true;
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
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float rz = std::sqrt( x * x + y * y );
	if ( jCell == 0 || iCell == 0 || iCell == Info.cellCountX-1 )
	{ 	// lake sides
		BC.openBCID = 3;
		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzInlet; 
		BC.nonReflective = false;
	}
	else if ( kCell == 0 ) 
	{	// lake inlet
		BC.openBCID = 0;
		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzInlet;
		BC.nonReflective = false;
	}
	else if ( kCell == Info.cellCountZ-1 && rz > 25.f ) 
	{	// lake outlet
		BC.openBCID = 1;
		BC.dirichletRho = true;
		BC.dRho = 0.f;
	}
	else if ( kCell == Info.cellCountZ-1 && rz <= 25.f ) 
	{	// pump outlet
		BC.openBCID = 2;
		BC.dirichletRho = true;
		BC.dRho = 0.f;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float rz = std::sqrt( x * x + y * y );
	const float vtPhys = radiansPerSecond * (rz / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( BC.wallID == 0 || BC.wallID == 1 ) // intake, stator
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( BC.wallID == 2 ) // Impeller shaft
	{
		BC.ux = - vt * (y / rz);
		BC.uy = vt * (x / rz);
		BC.uz = 0.f;
	}
	if ( Info.gridID == 0 || Info.gridID == 1 ) 
	{
		BC.collisionLimiter = 0.f; // switch to K15 for coarsest levels
		BC.overwriteIBBLinks = 0.5f;
	}
	if ( Info.gridID == 2 ) BC.collisionLimiter = 0.01f;
	if ( z <= -100.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.overwriteIBBLinks = 0.5f;
	}
	if ( z >= Info.Bounds.zMax-10.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.overwriteIBBLinks = 0.5f;
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
	float xCut = 0.f;
	exportSectionCutPlotZY( grids, xCut, iterationsFinished + 0 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	
	// ZY detail
	BoundsStruct Bounds;
	Bounds = grids[0].Info.Bounds;
	Bounds.zMin = -120.f;
	Bounds.yMin = -60.f;
	Bounds.yMax = 20.f;
	exportSectionCutPlotZY( grids, Bounds, xCut, iterationsFinished + 1 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	
	// XY section cut
	float zCut = 0.f;
	exportSectionCutPlotXY( grids, zCut, iterationsFinished + 2 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	
	// XY detail
	Bounds.xMin = -50.f;
	Bounds.xMax = 50.f;
	Bounds.yMin = -50.f;
	Bounds.yMax = 20.f;
	exportSectionCutPlotXY( grids, Bounds, zCut, iterationsFinished + 3 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 3 );
	readSTL( gridStaticSTLs[0], STLPathIntake );
	readSTL( gridStaticSTLs[1], STLPathPump );
	readSTL( gridStaticSTLs[2], STLPathShaft );
	
	std::vector<STLStruct> rotorSTLs( 1 );
	readSTL( rotorSTLs[0], STLPathImpeller );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMin = -200.f;
	DomainBounds.zMax = 77.5f;
	DomainBounds.xMin = -90.f;
	DomainBounds.xMax = 90.f;
	DomainBounds.yMin = -120.f;
	DomainBounds.yMax = 23.f;
	
	grids[0].Info.useRotors = false;
	grids[1].Info.useRotors = false;
	grids[2].Info.useRotors = false;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.rotateAlongZ = true;
	
	TrackerStruct Tracker;
	//Tracker.TRACK_CUSTOM_VARIABLES = true;
	//Tracker.customNames = { "Mass flow", "Head", "Hydraulic efficiency", "Hydraulic input power" };
	//Tracker.customUnits = { "kg/s", "m", "%", "kW" };
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
			/*
			const float torqueTotal = Tracker.rotorTz[0] + Tracker.wallTz[1] + Tracker.wallTz[2]; // rotor + shaft + shroud
			const float inputPower = torqueTotal * radiansPerSecond;
			const float outputPower = Tracker.pressurePower[0] + Tracker.normalKineticPower[0] + Tracker.pressurePower[1] + Tracker.normalKineticPower[1];
			const float etaPercent = 100.f * outputPower / inputPower;
			const float massFlow = 0.5f * ( - Tracker.massFlow[0] + Tracker.massFlow[1] );
			const float head = outputPower / ( massFlow * 9.81f );
			const float inputPowerKw = inputPower * 0.001f;
			trackCustomVariables( Tracker, { massFlow, head, etaPercent, inputPowerKw });
			*/
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
