constexpr float RES_GLOBAL = 1.2f; 
constexpr float uzInlet = 0.01f; 
constexpr int GRID_LEVEL_COUNT = 3;
constexpr int ITERATION_COUNT = 100000; 												

constexpr int PLOTTER_PERIOD = 20000;

constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;								// kg/m3 water
constexpr float NU_PHYS = 1e-6;									// m2/s water

constexpr float uzInletPhys = 15.f;								// m/s
constexpr float uyInlet = 0.f; // 0.0436 * uzInlet; 			// this is due to the 2.5 deg intake angle
constexpr float targetMassFlow = 4.f; // kg/s

constexpr float iRegulatorOutletStrength = 250000000.f;
//constexpr float iRegulatorInletStrength = 500.f;

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathIntake = "BruteforceIntakeSTL_v16.STL";
std::string STLPathHull = "HullBlockSimpleSTL.STL";

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
		if ( TNL::abs(x) < 70.f && y > -40.f ) refinementMarker = true;
	}
	if ( Info.gridID == 1 )
	{
		refinementMarker = false;
		if ( TNL::abs(x) < 35.f && y > -15.f ) refinementMarker = true;
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
	else if ( kCell == Info.cellCountZ-1 && y < 1.f ) 
	{	// lake outlet
		BC.openBCID = 1;
		BC.dirichletRho = true;
		//BC.dRho = Info.iRegulatorInlet;
		BC.dRho = 0.f;
		BC.nonReflective = true;
	}
	else if ( kCell == Info.cellCountZ-1 && y >= 1.f ) 
	{	// pump outlet
		BC.openBCID = 2;
		BC.dirichletRho = true;
		BC.dRho = Info.iRegulatorOutlet;
		BC.nonReflective = true;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	if ( BC.wallID == 0 || BC.wallID == 1 ) // intake, hull
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	BC.collisionLimiter = 0.f; // switch to K15 by default
	if ( Info.gridID > 1 && TNL::abs(x) < 32.f && y > -12.f && z > -200.f ) BC.collisionLimiter = 0.01f; // K17 for the internals of the finest domain
	if ( z >= Info.Bounds.zMax-19.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.nuMultiplier = 200.f;
	}
	if ( z >= Info.Bounds.zMax-10.f && y < 1.f ) 
	{
		BC.useIBB = false;
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
	//BoundsStruct Bounds;
	//Bounds = grids[1].Info.Bounds;
	//Bounds.yMax = grids[GRID_LEVEL_COUNT-1].Info.Bounds.yMax;
	for ( float zCut = -130.f; zCut < 20.f; zCut += 10.f )
	{
		//exportSectionCutPlotXY( grids, Bounds, zCut, iterationsFinished + counter );
		exportSectionCutPlotXY( grids, zCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	
	// ZX section cut
	for ( float yCut = -50.f; yCut <= 1.f; yCut += 5.f )
	{
		exportSectionCutPlotZX( grids, yCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 2 );
	readSTL( gridStaticSTLs[0], STLPathIntake );
	readSTL( gridStaticSTLs[1], STLPathHull );
	
	std::vector<STLStruct> rotorSTLs( 0 );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMin = -250.f;
	DomainBounds.zMax = 40.f;
	DomainBounds.xMin = -150.f;
	DomainBounds.xMax = 150.f;
	DomainBounds.yMin = -200.f;
	DomainBounds.yMax = 45.f;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	
	Tracker.customNames = { "Mass flow", "Intake power", "Intake efficiency", "Intake drag", "Intake downforce", "Intake X moment" };
	Tracker.customUnits = { "kg/s", "W", "[1]", "N", "N", "Nm" };
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
			
			// get intake flow report
			FlowReportStruct FlowReportIntake;
			BoundsStruct Bounds;
			Bounds.xMin = -17.f; Bounds.xMax = 17.f; Bounds.yMin = 1.f; Bounds.yMax = 45.f;
			float zCut = 0.f;
			getFlowReportXY( FlowReportIntake, grids, Bounds, zCut );
			
			// now prepare all reported variables
			
			// 1) mass flow
			const float massFlow = FlowReportIntake.massFlow;
			
			// 2) intake momentum
			//const float intakeMomentum = FlowReportIntake.momentumThrust;
						
			// 3) intake power
			const float intakePower = FlowReportIntake.pressurePower + 0.5f * massFlow * FlowReportIntake.normalVelocity * FlowReportIntake.normalVelocity;
			
			// 4) intake efficiency
			const float lakePower = 0.5f * uzInletPhys * uzInletPhys * massFlow;
			const float etaIntake = intakePower / lakePower;
			
			// 5) intake drag
			const float exitAreaM2 = FlowReportIntake.areamm2 * 1e-6f; 
			// this area is at plane z=0, not at the actual outlet plane, but the intake continues as a straight cylinder so it equals
			const float intakeDrag =
				- Tracker.wallFz[0]
				+ ( Tracker.pressure[2] ) * exitAreaM2
				+ Tracker.momentumThrust[2]
				- FlowReportIntake.massFlow * uzInletPhys;
			
			const float intakeDownforce = Tracker.wallFy[0]; // Positive = downward on intake
			const float intakeXMoment = Tracker.wallTx[0];

			// pass results to tracker
			trackCustomVariables(Tracker, {	massFlow, intakePower, etaIntake, intakeDrag, intakeDownforce, intakeXMoment });
			
			// regulate pump outlet to achieve target mass flow
			const InfoStruct& coarseInfo = grids[0].Info;
			const float regulatorDt = static_cast<float>(TRACKER_PERIOD) * coarseInfo.dtPhys;
			float pressurePerDRho = 1.f; 
			convertToPhysicalPressure(pressurePerDRho, coarseInfo);
			grids[0].Info.iRegulatorOutlet +=  ( massFlow - targetMassFlow ) * iRegulatorOutletStrength * regulatorDt / pressurePerDRho;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorOutlet = grids[0].Info.iRegulatorOutlet;
			
			// regulate lake outlet to achieve zero pressure power at lake inlet
			/*
			FlowReportStruct lakeReference;
			BoundsStruct referenceBounds{};
			referenceBounds.xMin = -40.f;
			referenceBounds.xMax =  40.f;
			referenceBounds.yMin = -140.f;
			referenceBounds.yMax =  -80.f;
			getFlowReportXY(lakeReference, grids, referenceBounds, -200.f);
			const float inletVolumeFlow = -Tracker.massFlow[0] / RHO_PHYS;
			grids[0].Info.iRegulatorInlet -= lakeReference.pressure * inletVolumeFlow * iRegulatorInletStrength * regulatorDt / pressurePerDRho;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorInlet = grids[0].Info.iRegulatorInlet;
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
