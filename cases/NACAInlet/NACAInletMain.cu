constexpr float RES_GLOBAL = 2.f; 
constexpr float uzInlet = 0.04f; 
constexpr int GRID_LEVEL_COUNT = 4;
constexpr int ITERATION_COUNT = 20000; 												

constexpr int PLOTTER_PERIOD = 2000;

constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float RHO_PHYS = 997.0f;							// kg/m3 water
constexpr float NU_PHYS = 1e-6;								// m2/s water

constexpr float uzInletPhys = 20.f;							// m/s

constexpr float iRegulatorOutletStrength = 10000000.f;
constexpr float targetMassFlow = 10.2f;

constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); // s

#include "../../include/types.h"

std::string STLPathIntake = "../../../BruteforceOptimizer/NACA/NACA_7deg.STL";

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
		if ( x > -50.f && x < 50.f && y > -30.f ) refinementMarker = true;
	}
	if ( Info.gridID == 1 )
	{
		refinementMarker = false;
		if ( x > -40.f && x < 40.f && y > -20.f ) refinementMarker = true;
	}
	if ( Info.gridID == 2 )
	{
		refinementMarker = false;
		if ( x > -30.f && x < 30.f && y > -5.f ) refinementMarker = true;
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
	else if ( kCell == Info.cellCountZ-1 && y < 1.f ) 
	{	// lake outlet
		BC.openBCID = 1;
		BC.dirichletRho = true;
		BC.dRho = 0.f;
	}
	else if ( kCell == Info.cellCountZ-1 && y >= 1.f ) 
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
	if ( BC.wallID == 0 ) // intake
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( Info.gridID == 0 || Info.gridID == 1 ) 
	{
		BC.collisionLimiter = 0.f; // switch to K15 for coarsest levels
		BC.overwriteIBBLinks = 0.5f;
	}
	if ( Info.gridID == 2 ) BC.collisionLimiter = 0.01f;
	if ( Info.gridID == 3 ) BC.collisionLimiter = 0.01f;
	if ( z <= -170.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.overwriteIBBLinks = 0.5f;
	}
	if ( z >= Info.Bounds.zMax-10.f && y < 1.f ) 
	{
		BC.collisionLimiter = 0.f;
		BC.overwriteIBBLinks = 0.5f;
	} 
	if ( z >= Info.Bounds.zMax-19.f ) 
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
	for ( float xCut = 0.f; xCut <= 21.f; xCut += 5.f )
	{
		exportSectionCutPlotZY( grids, xCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	
	// XY details
	BoundsStruct Bounds;
	Bounds = grids[1].Info.Bounds;
	Bounds.yMax = 20.f;
	for ( float zCut = -160.f; zCut < 10.f; zCut += 20.f )
	{
		exportSectionCutPlotXY( grids, Bounds, zCut, iterationsFinished + counter );
		counter++;
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}

	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathIntake );
	
	std::vector<STLStruct> rotorSTLs( 0 );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMin = -200.f;
	DomainBounds.zMax = 20.f;
	DomainBounds.xMin = -90.f;
	DomainBounds.xMax = 90.f;
	DomainBounds.yMin = -100.f;
	DomainBounds.yMax = 25.f;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	
	Tracker.customNames = { "Mass flow", "Intake power", "Intake efficiency", "Intake induced drag" };
	Tracker.customUnits = { "kg/s", "W", "[1]", "N" };
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
			BoundsStruct Bounds; 
			Bounds = grids[GRID_LEVEL_COUNT-1].Info.Bounds;
			Bounds.yMin = 1.f;
			float zCut = 0.f;
			getFlowReportXY( FlowReportOut, grids, Bounds, zCut );
			
			// get reference flow report
			FlowReportStruct FlowReportRef; 
			Bounds = grids[0].Info.Bounds;
			Bounds.yMax = -30.f;
			Bounds.xMin += 10.f;
			Bounds.xMax -= 10.f;
			Bounds.yMin += 10.f;
			getFlowReportXY( FlowReportRef, grids, Bounds, zCut );
			
			// get drag flow report
			FlowReportStruct FlowReportDrag;
			Bounds = grids[1].Info.Bounds;
			Bounds.yMax =  1.f;
			Bounds.xMin += 2.f;
			Bounds.xMax -= 2.f;
			Bounds.yMin += 2.f;
			getFlowReportXY( FlowReportDrag, grids, Bounds, zCut );
			
			// 2) mass flow
			const float massFlow = FlowReportOut.massFlow;
						
			// 4) intake power
			const float intakePower = FlowReportOut.pressurePower + 0.5f * massFlow * FlowReportOut.normalVelocity * FlowReportOut.normalVelocity;
			
			// 10) intake efficiency
			const float lakePower = 0.5f * uzInletPhys * uzInletPhys * massFlow;
			const float etaIntake = intakePower / lakePower;
			
			// intake induced drag
			const float uzRef = FlowReportRef.normalVelocity;
			const float momentumRef = uzRef * uzRef * FlowReportDrag.areamm2 * (1.f / 1000000.f) * RHO_PHYS;
			const float drag = momentumRef - FlowReportDrag.momentumThrust;
			
			// pass results to tracker
			trackCustomVariables( Tracker, { massFlow, intakePower, etaIntake, drag });
			
			// regulate outlet to achieve target mass flow
			const InfoStruct& coarseInfo = grids[0].Info;
			const float regulatorDt = static_cast<float>(TRACKER_PERIOD) * coarseInfo.dtPhys;
			float pressurePerDRho = 1.f; 
			convertToPhysicalPressure(pressurePerDRho, coarseInfo);
			grids[0].Info.iRegulatorOutlet -=  ( targetMassFlow - massFlow ) * iRegulatorOutletStrength * regulatorDt / pressurePerDRho;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) grids[level].Info.iRegulatorOutlet = grids[0].Info.iRegulatorOutlet;
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
