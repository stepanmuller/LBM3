// very coarse
//constexpr float RES_GLOBAL = 0.2f; 
//constexpr float uzInlet = 0.01f; 
//constexpr int ITERATION_COUNT = 40000; 

// coarse
// constexpr float RES_GLOBAL = 0.16f; 
// constexpr float uzInlet = 0.01f; 
// constexpr int ITERATION_COUNT = 50000; 

// medium, same as 35 bruteforce full reference case!
//constexpr float RES_GLOBAL = 0.125f; 
//constexpr float uzInlet = 0.01f; 
//constexpr int ITERATION_COUNT = 70000; 												

// fine
constexpr float RES_GLOBAL = 0.1f; 
constexpr float uzInlet = 0.01f; 
constexpr int ITERATION_COUNT = 60000; 												

// very fine
//constexpr float RES_GLOBAL = 0.08f; 
//constexpr float uzInlet = 0.01f; 
//constexpr int ITERATION_COUNT = 100000; 

// finest
//constexpr float RES_GLOBAL = 0.064f; 
//constexpr float uzInlet = 0.01f; 
//constexpr int ITERATION_COUNT = 130000; 

constexpr int PLOTTER_PERIOD = 20000;

constexpr int GRID_LEVEL_COUNT = 1;
constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr int TRACKER_PERIOD = 10;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = true;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;							// kg/m3 water
constexpr float NU_PHYS = 1e-6;								// m2/s water

constexpr float massFlowReference = 5.4f;														// kg/s
constexpr float RInlet = 16.5f;																// mm
constexpr float RInletShaft = 3.75f;														// mm
constexpr float inletAreamm2 = 3.14159f * ( RInlet * RInlet - RInletShaft * RInletShaft);	// mm2
constexpr float uzInletPhys = massFlowReference / ( RHO_PHYS * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float uzInletVariation = 0.6f; // from the nominal value, vary the uzInlet half up and half down to match the actual intake flow
constexpr float uzInletVariationDistance = 5.f;
constexpr float targetInletPowerNormalized = 786.f / 5.4f; // Watts per kg/s of mass flow
//constexpr float hullVelocityPhys = 20.f; // m/s reference hull velocity from which the inlet power was sampled

constexpr float rotationStartDistance = 10.f;				// mm, distance from inlet where the shaft starts rotating (avoid full rotation at the very start)
const float boundaryLayerThickness = 0.25f + RES_GLOBAL;	// mm

constexpr float radiansPerSecond = 2700.f;					// rad/s

constexpr float iRegulatorInletStrength = 200.f;				// this will regulate inlet velocity to converge towards target inlet power
constexpr float iRegulatorOutletStrength = 200000.f;			// this will regulate outlet pressure to achieve zero pressure at the actual outlet z coordinate

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathPumpShroud = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforcePumpShroudSTL.stl";
std::string STLPathOutlet = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforceOutletSTL.stl";
std::string STLPathShaft = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforceShaftSTL.stl";
std::string STLPathFirstImpeller = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforceFirstImpellerSTL.stl";
std::string STLPathSecondImpeller = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforceSecondImpellerSTL.stl";
std::string STLPathStator = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforceStatorSTL.stl";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	// no refinement is used in this case
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
	// global velocity modifier to achieve non uniform profile
	const float uzBase = uzInlet + Info.iRegulatorInlet;
	const float uzMin = uzBase * ( 1.f - uzInletVariation );
	const float uzMax = uzBase * ( 1.f + uzInletVariation );
	float uzResult = uzMax;
	if ( y > uzInletVariationDistance ) uzResult = uzMin;
	else if ( y > -uzInletVariationDistance )
	{
		const float t = ( y + uzInletVariationDistance ) / ( 2.f * uzInletVariationDistance );
		const float q = t * t * ( 3.f - 2.f * t );
		uzResult = uzMax + q * ( uzMin - uzMax );
	}
	// velocity multiplier near walls
	const float wallDistancePhys = std::min(std::max(0.f, RInlet-rz), std::max(0.f, rz-RInletShaft));
	const float delta = std::max( 0.f, std::min( 1.f, wallDistancePhys / boundaryLayerThickness ));
	const float velocityMultiplier = delta * delta * (3.0f - 2.0f * delta);
	uzResult *= velocityMultiplier;
	
	if ( kCell == 0 )
	{ 	// inlet
		BC.openBCID = 0;
		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = uzResult;
	}
	else if ( kCell == Info.cellCountZ-1 ) 
	{	// outlet
		BC.openBCID = 1;
		BC.dirichletRho = true;
		BC.dRho = Info.iRegulatorOutlet;
		BC.nonReflective = false;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	BC.collisionLimiter = 0.02f; // default global value
	const float rz = std::sqrt( x * x + y * y );
	const float vtPhys = radiansPerSecond * (rz / 1000.f);
	float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( z < Info.Bounds.zMin + rotationStartDistance ) vt *= ( ( z - Info.Bounds.zMin) / rotationStartDistance );
	if ( BC.wallID == 0 || BC.wallID == 1 || BC.wallID == 2 ) // shroud, outlet, stator
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( BC.wallID == 3 ) // Impeller shaft
	{
		BC.ux = - vt * (y / rz);
		BC.uy = vt * (x / rz);
		BC.uz = 0.f;
	}
	if ( z < Info.Bounds.zMin + 5.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.overwriteIBBLinks = 0.5f;
	}
	if ( z > Info.Bounds.zMax-19.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.nuMultiplier = 200.f;
		BC.overwriteIBBLinks = 0.5f;
	} 
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/flowReportFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/exportToiletPaperPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	BoundsStruct Bounds;
	Bounds = grids[0].Info.Bounds;
	int counter = 0;
	// ZY section cut
	for ( float xCut = 0.f; xCut < Bounds.xMax; xCut += 3.f )
	{
		exportSectionCutPlotZY( grids, xCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	// XY details
	for ( float zCut = Bounds.zMin+grids[0].Info.res*0.5f; zCut < Bounds.zMax-grids[0].Info.res*0.5f; zCut += 10.f )
	{
		exportSectionCutPlotXY( grids, zCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	// Toilet paper projection
	for ( float rz = 8.f; rz < 16.5f; rz += 1.f ) 
	{
		toiletPaperPlotZ(grids, rz, iterationsFinished + counter );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
		counter++;
		toiletPaperPlotZ(grids, grids[GRID_LEVEL_COUNT-1].rotors[0].Info, rz, iterationsFinished + counter );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
		counter++;
	}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 4 );
	readSTL( gridStaticSTLs[0], STLPathPumpShroud );
	readSTL( gridStaticSTLs[1], STLPathOutlet );
	readSTL( gridStaticSTLs[2], STLPathStator );
	readSTL( gridStaticSTLs[3], STLPathShaft );
	
	std::vector<STLStruct> rotorSTLs( 2 );
	readSTL( rotorSTLs[0], STLPathFirstImpeller );
	readSTL( rotorSTLs[1], STLPathSecondImpeller );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMin = rotorSTLs[0].Bounds.zMin - 20.f;
	DomainBounds.zMax = gridStaticSTLs[1].Bounds.zMax;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.rotateAlongZ = true;
	grids[GRID_LEVEL_COUNT-1].rotors[1].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[1].Info.rotateAlongZ = true;
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	Tracker.averagePercent = 25.9f; // at the selected res and iteration count this gives 1 revolution exactly
	
	Tracker.customNames = { "Inlet power residual", "Outlet pressure power residual", "Mass flow", "Shaft power", "Impeller power", "Impeller efficiency" };
	Tracker.customUnits = { "W", "W", "kg/s", "W", "W", "[1]" };
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
			// update tracker
			updateTracker( Tracker, grids );
			
			// get outlet flow report
			FlowReportStruct FlowReportOut;
			BoundsStruct Bounds; Bounds.xMin = -13.5f; Bounds.xMax = 13.5f; Bounds.yMin = -13.5f; Bounds.yMax = 13.5f;
			float zCut = grids[0].Info.Bounds.zMax - 20.f;
			getFlowReportXY( FlowReportOut, grids, Bounds, zCut );
			
			// now prepare all reported variables
			
			// 1) inlet power residual
			const float targetInletPower = targetInletPowerNormalized * (-Tracker.massFlow[0]);
			const float actualInletPower = - Tracker.pressurePower[0] - ( 0.5f * Tracker.normalVelocity[0] * Tracker.normalVelocity[0] * Tracker.massFlow[0] );
			const float inletPowerResidual = actualInletPower - targetInletPower;
			
			// 2) outlet pressure power residual
			const float outletPressurePowerResidual = FlowReportOut.pressurePower;
			
			// 3) mass flow
			const float massFlow = FlowReportOut.massFlow;
			
			// 4) shaft power
			const float shaftTorque = Tracker.rotorTz[0] + Tracker.rotorTz[1]; // + Tracker.wallTz[3]; // first impeller + second impeller + shaft ... shat can be neglected to stop tracking walls and make it faster
			const float shaftPower = shaftTorque * radiansPerSecond;
			
			// 5) impeller power (excludes variation in outlet velocity profile)
			const float usefulOutletPower = FlowReportOut.pressurePower + 0.5f * massFlow * FlowReportOut.normalVelocity * FlowReportOut.normalVelocity;
			const float impellerPower = usefulOutletPower - actualInletPower;
						
			// 6) impeller efficiency
			const float etaImpeller = impellerPower / shaftPower;
			
			// pass results to tracker
			trackCustomVariables( Tracker, { inletPowerResidual, outletPressurePowerResidual, massFlow, shaftPower, impellerPower, etaImpeller });
			
			// regulate pump outlet to achieve zero pressure power at the actual outlet coordinate
			const InfoStruct& coarseInfo = grids[0].Info;
			const float regulatorDt = static_cast<float>(TRACKER_PERIOD) * coarseInfo.dtPhys;
			float pressurePerDRho = 1.f; 
			convertToPhysicalPressure(pressurePerDRho, coarseInfo);
			
			grids[0].Info.iRegulatorOutlet -=  outletPressurePowerResidual * iRegulatorOutletStrength * regulatorDt / pressurePerDRho;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorOutlet = grids[0].Info.iRegulatorOutlet;
			
			// regulate pump inlet to achieve target inlet power
			const float inletPowerReference = targetInletPowerNormalized * massFlowReference;
			grids[0].Info.iRegulatorInlet -= inletPowerResidual * iRegulatorInletStrength * regulatorDt * uzInlet / inletPowerReference;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorInlet = grids[0].Info.iRegulatorInlet;
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
	
	// WRITE RESULT FOR OPTIMIZER
	
	// Match the percentage rounding used when calling plotTracker.py.
	const int count = Tracker.iterationsFinished;
	const double averagePercent = std::stod( std::to_string( static_cast<double>( Tracker.averagePercent ) ) );
	const int window = std::max( 1, static_cast<int>( count * averagePercent / 100.0 ) );
	auto historyMean = [&]( const FloatArray2DTypeCPU& history, int id )
	{
		double sum = 0.0;
		int finiteCount = 0;
		for ( int i = count - window; i < count; i++ )
		{
			const float value = history( id, i );
			if ( !std::isfinite( value ) ) continue;
			sum += static_cast<double>( value );
			finiteCount++;
		}
		// Keep the zero fallback when no usable samples remain.
		return finiteCount > 0 ? sum / finiteCount : 0.0;
	};
	std::ofstream result( "simulationResult.txt" );
	result.precision( 9 );
	// One value per line, in this order:
	result << historyMean( Tracker.customArray, 5 ) << '\n';  // Impeller eta [1]
	result << historyMean( Tracker.customArray, 2 ) << '\n';  // Mass flow [kg/s]
	result << historyMean( Tracker.customArray, 3 ) << '\n';  // Shaft power [W]
	result << historyMean( Tracker.rotorTzArray, 0 ) << '\n'; // Rotor 0 Tz [N m] = first impeller torque
	result << historyMean( Tracker.rotorTzArray, 1 ) << '\n'; // Rotor 1 Tz [N m] = second impeller torque
	result << historyMean( Tracker.wallTzArray, 2 ) << '\n'; // Wall 1 Tz [N m] = stator torque
	result << historyMean( Tracker.wallTzArray, 1 ) << '\n'; // Rotor 1 Tz [N m] = outlet torque
	result.close();
	
	// WRITE RESULT FOR OPTIMIZER END
	
	return EXIT_SUCCESS;
}
