// coarse
//constexpr float RES_GLOBAL = 4.f; 
//constexpr int GRID_LEVEL_COUNT = 4;
//constexpr int ITERATION_COUNT = 80000; 

// medium
constexpr float RES_GLOBAL = 3.2f; // 2.64f;
constexpr int GRID_LEVEL_COUNT = 4;
constexpr int ITERATION_COUNT = 80000; // 100000;

constexpr int PLOTTER_PERIOD = 2000;

constexpr int WALL_REFINEMENT_COUNT = 3;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = true;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;	// kg/m3 water
constexpr float NU_PHYS = 1e-6;		// m2/s water

constexpr float uzInlet = 0.01f; 															// also works as nominal LBM Mach number	
constexpr float massFlowPhys = 335.f;														// kg/s
constexpr float RInlet = 150.f;																// mm
constexpr float inletAreamm2 = 3.14159f * RInlet * RInlet;									// mm2
constexpr float uzInletPhys = massFlowPhys / ( RHO_PHYS * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float radiansPerSecond = -198.967f;												// rad/s
const float boundaryLayerThickness = 2.f;													// mm

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathStator = "../../../../ns300/ns300_STATOR.STL";
std::string STLPathRotorShaft = "../../../../ns300/ns300_ROTOR_SHAFT.STL";
std::string STLPathRotorShroud = "../../../../ns300/ns300_ROTOR_SHROUD.STL";
std::string STLPathRotorBlades = "../../../../ns300/ns300_ROTOR_BLADES.STL";
std::string STLPathTipGapBlocker = "../../../../ns300/ns300_TIP_GAP_BLOCKER.STL";

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
		if ( y <= 400.f && z <= 200.f ) refinementMarker = true;
	}
	if ( Info.gridID == 1 )
	{
		refinementMarker = false;
		if ( rz < 180.f && z < 150.f ) refinementMarker = true;
	}
	if ( Info.gridID == 2 )
	{
		float zMin = -3.f;
		float zMax = 124.f;
		float rzMax = 164.f;
		if ( z > 68.f ) rzMax = 127.f + ( 92.f - z);
		if ( z > 92.f ) rzMax = 127.f;
		float rzMin = 58.f;
		if ( z > 68.f ) rzMin = 58.f + ( z - 68.f );

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
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float wallDistancePhys = std::max(0.f, RInlet-r);
	const float delta = std::max( 0.f, std::min( 1.f, wallDistancePhys / boundaryLayerThickness ));
	const float velocityMultiplier = delta * delta * (3.0f - 2.0f * delta);
	if ( kCell == Info.cellCountZ-1 ) // Inlet
	{
		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = - ( uzInlet ) * velocityMultiplier;
		BC.openBCID = 0;
		BC.rhoReflectionTolerance = 3.8e-8f * RES_GLOBAL;
	}
	else if ( jCell == Info.cellCountY-1 ) // Outlet
	{
		BC.dirichletRho = true;
		BC.openBCID = 1;
		BC.dRho = 0.f;
		BC.rhoReflectionTolerance = 3.8e-8f * RES_GLOBAL;
	}
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
	if ( BC.wallID == 1 || BC.wallID == 2 ) // rotor shaft and shroud
	{
		BC.ux = - vt * (y / r);
		BC.uy = vt * (x / r);
		BC.uz = 0.f;
	}
	if ( BC.wallID == 3 ) // tip gap blocker -> interpolate rotation linearly
	{
		float rotationPart = ( r - 128.f ) / 2.3f;
		rotationPart = std::clamp( rotationPart, 0.f, 1.f );
		BC.ux = rotationPart * ( - vt * (y / r) );
		BC.uy = rotationPart * ( vt * (x / r) );
		BC.uz = 0.f;
	}
	if ( kCell >= Info.cellCountZ-20 || jCell >= Info.cellCountY-20 ) BC.collisionLimiter = 0.f;
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	int iCut, jCut, kCut;
	const float xTemp = 0.f; const float yTemp = 0.f; const float zTemp = 0.f;
	// ZY section cut shows the inlet pipe
	float xCut = 0.f;
	getIJKCellIndexFromXYZ( iCut, jCut, kCut, xCut, yTemp, zTemp, grids[GRID_LEVEL_COUNT-1].Info);
	exportSectionCutPlotZY( grids, iCut, iterationsFinished + 0 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	// Detail
	if ( GRID_LEVEL_COUNT >= 3 )
	{
		exportSectionCutPlotZY( grids, grids[2].Info.Bounds, iCut, iterationsFinished + 1 );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	// XY section cut shows the rotor and the outlet pipe
	float zCut = 32.5f;
	getIJKCellIndexFromXYZ( iCut, jCut, kCut, xTemp, yTemp, zCut, grids[GRID_LEVEL_COUNT-1].Info);
	exportSectionCutPlotXY( grids, kCut, iterationsFinished + 2 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	// Detail
	if ( GRID_LEVEL_COUNT >= 3 )
	{
		exportSectionCutPlotXY( grids, grids[2].Info.Bounds, kCut, iterationsFinished + 3 );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	// Detail in rotor frame
	if ( GRID_LEVEL_COUNT >= 3 )
	{
		exportSectionCutPlotXY( grids, grids[2].Info.Bounds, grids[GRID_LEVEL_COUNT-1].rotors[0].Info, kCut, iterationsFinished + 4 );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 4 );
	readSTL( gridStaticSTLs[0], STLPathStator );
	readSTL( gridStaticSTLs[1], STLPathRotorShaft );
	readSTL( gridStaticSTLs[2], STLPathRotorShroud );
	readSTL( gridStaticSTLs[3], STLPathTipGapBlocker );
	
	std::vector<STLStruct> rotorSTLs( 1 );
	readSTL( rotorSTLs[0], STLPathRotorBlades );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMax = 400.f;
	DomainBounds.yMax = 1000.f;
	
	grids[0].Info.useRotors = false;
	grids[1].Info.useRotors = false;
	grids[GRID_LEVEL_COUNT-2].Info.useRotors = false;
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.rotateAlongZ = true;
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	Tracker.customNames = { "Mass flow", "Head", "Hydraulic efficiency", "Hydraulic input power" };
	Tracker.customUnits = { "kg/s", "m", "%", "kW" };
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
			const float torqueTotal = Tracker.rotorTz[0] + Tracker.wallTz[1] + Tracker.wallTz[2]; // rotor + shaft + shroud
			const float inputPower = torqueTotal * radiansPerSecond;
			const float outputPower = Tracker.pressurePower[0] + Tracker.normalKineticPower[0] + Tracker.pressurePower[1] + Tracker.normalKineticPower[1];
			const float etaPercent = 100.f * outputPower / inputPower;
			const float massFlow = 0.5f * ( - Tracker.massFlow[0] + Tracker.massFlow[1] );
			const float head = outputPower / ( massFlow * 9.81f );
			const float inputPowerKw = inputPower * 0.001f;
			trackCustomVariables( Tracker, { massFlow, head, etaPercent, inputPowerKw });
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
