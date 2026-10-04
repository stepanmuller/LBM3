constexpr float RES_GLOBAL = 1.f; 
constexpr float uzInlet = 0.01f; 
constexpr int ITERATION_COUNT = 200000; 												

constexpr int PLOTTER_PERIOD = 5000;

constexpr int GRID_LEVEL_COUNT = 1;
constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = true;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;							// kg/m3 water
constexpr float NU_PHYS = 1e-6;								// m2/s water

constexpr float massFlowPhys = 5.4f;														// kg/s
constexpr float RInlet = 16.5f;																// mm
constexpr float RInletShaft = 3.75f;														// mm
constexpr float inletAreamm2 = 3.14159f * ( RInlet * RInlet - RInletShaft * RInletShaft);	// mm2
constexpr float uzInletPhys = massFlowPhys / ( RHO_PHYS * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float uzInletVariation = 0.5f; // from the nominal value, vary the uzInlet half up and half down to match the actual intake flow
constexpr float uzInletVariationDistance = 5.f;
constexpr float targetInletPowerNormalized = 786.f / 5.4f; // Watts per kg/s of mass flow

constexpr float rotationStartDistance = 10.f;				// mm, distance from inlet where the shaft starts rotating (avoid full rotation at the very start)
const float boundaryLayerThickness = 0.25f;					// mm

constexpr float radiansPerSecond = 2700.f;					// rad/s

constexpr float iRegulatorInletStrength = 50000.f;			// this will regulate inlet velocity to converge towards target inlet power
constexpr float iRegulatorOutletStrength = 5000.f;			// this will regulate outlet pressure to achieve zero pressure at the actual outlet z coordinate

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
	float uzResult = uzMin;
	if ( y > uzInletVariationDistance ) uzResult = uzMax;
	else if ( y > -uzInletVariationDistance )
	{
		const float t = ( y + uzInletVariationDistance ) / ( 2.f * uzInletVariationDistance )
		const float q = t * t * ( 3.f - 2.f * t );
		uzResult = uzMin + q * ( uzMax - uzMin );
	}
	// velocity multiplier near walls
	const float wallDistancePhys = std::min(std::max(0.f, RInlet-rz), std::max(0.f, rz-RInletShaft))
	const float delta = std::max( 0.f, std::min( 1.f, wallDistancePhys / boundaryLayerThickness ));
	const float velocityMultiplier = delta * delta * (3.0f - 2.0f * delta);
	uzResult *= velocityModifier;
	
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
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( z < Info.Bounds.zMin + rotationStartDistance ) vt *= ( z / rotationStartDistance )
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
	DomainBounds.zMin = rotorSTLs[0].Bounds.zMin - 30.f;
	DomainBounds.zMax = gridStaticSTLs[1].Bounds.zMax;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[0].Info.rotateAlongZ = true;
	grids[GRID_LEVEL_COUNT-1].rotors[1].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[1].Info.rotateAlongZ = true;
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	
	Tracker.customNames = { "Inlet pressure power residual", "Outlet pressure power residual", "Mass flow", "Shaft power", "Useful power", "Impeller efficiency" };
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
			BoundsStruct Bounds; Bounds.xMin = -16.f; Bounds.xMax = 16.f; Bounds.yMin = -16.f; Bounds.yMax = 16.f;
			float zCut = grids[0].Info.Bounds.zMax - 20.f;
			getFlowReportXY( FlowReportOut, grids, Bounds, zCut );
			
			// get intake flow report
			FlowReportStruct FlowReportIntake;
			Bounds.xMin = -17.f; Bounds.xMax = 17.f; Bounds.yMin = -25.f; Bounds.yMax = 17.f;
			zCut = -10.f; // move the measurement plane more in front to avoid impeller effects
			getFlowReportXY( FlowReportIntake, grids, Bounds, zCut );
			
			// now prepare all reported variables
			
			// 1) outlet pressure power residual
			const float outletPressurePower = FlowReportOut.pressurePower;
			
			// 2) mass flow
			const float massFlow = FlowReportOut.massFlow;
			
			// 3) thrust
			const float thrust = FlowReportOut.momentumThrust - uzInletPhys * massFlow;
						
			// 4) intake power
			const float intakePower = FlowReportIntake.pressurePower + 0.5f * massFlow * FlowReportIntake.normalVelocity * FlowReportIntake.normalVelocity;
			
			// 5) impeller power
			const float impellerPower = FlowReportOut.normalKineticPower + FlowReportOut.pressurePower - intakePower;
			
			// 6) kinetic power
			const float kineticPower = FlowReportOut.normalKineticPower;
			
			// 7) shaft torque
			const float shaftTorque = Tracker.rotorTz[0] + Tracker.rotorTz[1] + Tracker.wallTz[3]; // first impeller + second impeller + shaft
			
			// 8) shaft power
			const float shaftPower = shaftTorque * radiansPerSecond;
			
			// 9) useful power
			const float usefulPower = thrust * uzInletPhys;
			
			// 10) intake efficiency
			const float lakePower = 0.5f * uzInletPhys * uzInletPhys * massFlow;
			const float etaIntake = intakePower / lakePower;
			
			// 11) impeller efficiency
			const float etaImpeller = ( kineticPower - intakePower ) / shaftPower;
			
			// 12) total efficiency
			const float etaTotal = usefulPower / shaftPower;
			
			// pass results to tracker
			trackCustomVariables( Tracker, { outletPressurePower, massFlow, thrust, intakePower, impellerPower, kineticPower, 
												shaftTorque, shaftPower, usefulPower, etaIntake, etaImpeller, etaTotal });
			
			// regulate pump outlet to achieve zero pressure power at the actual outlet coordinate
			const InfoStruct& coarseInfo = grids[0].Info;
			const float regulatorDt = static_cast<float>(TRACKER_PERIOD) * coarseInfo.dtPhys;
			float pressurePerDRho = 1.f; 
			convertToPhysicalPressure(pressurePerDRho, coarseInfo);
			grids[0].Info.iRegulatorOutlet -=  outletPressurePower * iRegulatorOutletStrength * regulatorDt / pressurePerDRho;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorOutlet = grids[0].Info.iRegulatorOutlet;
			
			// regulate lake outlet to achieve zero pressure power at lake inlet
			grids[0].Info.iRegulatorInlet += Tracker.pressurePower[0] * iRegulatorInletStrength * regulatorDt / pressurePerDRho;
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
	
	return EXIT_SUCCESS;
}
