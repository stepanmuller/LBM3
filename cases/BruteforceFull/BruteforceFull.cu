constexpr float RES_GLOBAL = 1.f; 
constexpr float uzInlet = 0.01f; 
constexpr int GRID_LEVEL_COUNT = 4;
constexpr int ITERATION_COUNT = 200000; 												

constexpr int PLOTTER_PERIOD = 5000;

constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = true;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;							// kg/m3 water
constexpr float NU_PHYS = 1e-6;								// m2/s water

constexpr float uzInletPhys = 20.f;							// m/s
constexpr float uyInlet = 0.0436 * uzInlet; 				// this is due to the 2.5 deg intake angle

constexpr float radiansPerSecond = 2700.f;					// rad/s

constexpr float iRegulatorOutletStrength = 50000.f;
constexpr float iRegulatorInletStrength = 100.f;

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathIntake = "../../../BruteforceOptimizer/BruteforceGeometry/BruteforceIntakeSTL7.stl";
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
		if ( x > -25.f && x < 25.f && y > -32.f ) refinementMarker = true;
	}
	if ( Info.gridID == 2 )
	{
		refinementMarker = false;
		if ( y > -22.f + 0.0436f * z && x > -22.f && x < 22.f && z > -95.f ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	BC.uz = uzInlet;
	BC.uy = uyInlet;
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
		BC.uy = uyInlet;
		BC.uz = uzInlet; 
		BC.nonReflective = false;
	}
	else if ( kCell == 0 ) 
	{	// lake inlet
		BC.openBCID = 0;
		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = uyInlet;
		BC.uz = uzInlet;
		BC.nonReflective = false;
	}
	else if ( kCell == Info.cellCountZ-1 && rz > 20.f ) 
	{	// lake outlet
		BC.openBCID = 1;
		BC.dirichletRho = true;
		BC.dRho = Info.iRegulatorInlet;
	}
	else if ( kCell == Info.cellCountZ-1 && rz <= 20.f ) 
	{	// pump outlet
		BC.openBCID = 2;
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
	const float rz = std::sqrt( x * x + y * y );
	const float vtPhys = radiansPerSecond * (rz / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( BC.wallID == 0 || BC.wallID == 1 || BC.wallID == 2 ) // intake, outlet, stator
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
	if ( z >= Info.Bounds.zMax-10.f && rz > 20.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.overwriteIBBLinks = 0.5f;
	} 
	if ( z >= Info.Bounds.zMax-19.f && rz <= 20.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.nuMultiplier = 200.f;
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
	int counter = 0;
	// ZY section cut
	for ( float xCut = 0.f; xCut <= 26.f; xCut += 5.f )
	{
		exportSectionCutPlotZY( grids, xCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	
	// XY details
	BoundsStruct Bounds;
	Bounds = grids[1].Info.Bounds;
	Bounds.yMax = grids[GRID_LEVEL_COUNT-1].Info.Bounds.yMax;
	for ( float zCut = -130.f; zCut < -50.f; zCut += 20.f )
	{
		exportSectionCutPlotXY( grids, Bounds, zCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	for ( float zCut = -50.f; zCut < 6.f; zCut += 5.f )
	{
		exportSectionCutPlotXY( grids, Bounds, zCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	
	// Toilet paper projection
	Bounds.zMin = -10.f;
	for ( float rz = 10.f; rz < 15.5f; rz += 1.f ) 
	{
		toiletPaperPlotZ(grids, Bounds, rz, iterationsFinished + counter );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
		counter++;
		toiletPaperPlotZ(grids, Bounds, grids[GRID_LEVEL_COUNT-1].rotors[0].Info, rz, iterationsFinished + counter );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
		counter++;
	}
	
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 4 );
	readSTL( gridStaticSTLs[0], STLPathIntake );
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
	DomainBounds.zMin = -200.f;
	DomainBounds.zMax = gridStaticSTLs[1].Bounds.zMax;
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
	grids[GRID_LEVEL_COUNT-1].rotors[1].Info.radiansPerSecond = radiansPerSecond;
	grids[GRID_LEVEL_COUNT-1].rotors[1].Info.rotateAlongZ = true;
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	
	Tracker.customNames = { "Outlet pressure power residual", "Mass flow", "Thrust", "Intake power", "Impeller power", "Normal kinetic power",
							"Shaft torque", "Shaft power", "Useful power", "Intake efficiency", "Impeller efficiency", "Total efficiency" };
	Tracker.customUnits = { "W", "kg/s", "N", "W", "W", "W", "Nm", "W", "W", "[1]", "[1]", "[1]" };
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
			float zCut = grids[0].Info.Bounds.zMax - 10.f;
			getFlowReportXY( FlowReportOut, grids, Bounds, zCut );
			
			// get intake flow report
			FlowReportStruct FlowReportIntake;
			Bounds.xMin = -17.f; Bounds.xMax = 17.f; Bounds.yMin = -25.f; Bounds.yMax = 17.f;
			zCut = 0.f;
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
