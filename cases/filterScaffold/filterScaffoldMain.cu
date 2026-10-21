constexpr float uxInletPhys = 0.05305f;													// m/s
constexpr float NU_PHYS = 1.5e-5f;														// m2/s
constexpr float RHO_PHYS = 1.225f;														// kg/m3

// coarse settings
//constexpr float RES_GLOBAL = 0.175f;
//constexpr int GRID_LEVEL_COUNT = 2;
//constexpr float uxInlet = 0.002f; // in this case uzInlet has to be set very very low to reduce weakly compressible density error to acceptable level
//constexpr int ITERATION_COUNT = 50000;

// medium settings
constexpr float RES_GLOBAL = 0.5f * 0.175f;
constexpr int GRID_LEVEL_COUNT = 2;
constexpr float uxInlet = 0.002f; // in this case uzInlet has to be set very very low to reduce weakly compressible density error to acceptable level
constexpr int ITERATION_COUNT = 100000;

// fine settings
//constexpr float RES_GLOBAL = 0.12f;
//constexpr int GRID_LEVEL_COUNT = 3;
//constexpr float uxInlet = 0.002f; // in this case uzInlet has to be set very very low to reduce weakly compressible density error to acceptable level
//constexpr int ITERATION_COUNT = 30000;

constexpr int PLOTTER_PERIOD = 10000;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr int WALL_REFINEMENT_COUNT = 2;

constexpr float DT_PHYS_GLOBAL = (uxInlet / uxInletPhys) * (RES_GLOBAL/1000); 			// s

#include "../../include/types.h"

std::string STLPathFilter = "N2.stl";
constexpr int NCOUNT = 2;

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float refinementMin = 8.f;
	const float refinementMax = refinementMin + NCOUNT * 5.6f + 0.8f + 3.f;
	if ( Info.gridID == 0 )
	{
		refinementMarker = false;
		if ( x > refinementMin && x < refinementMax ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	BC.ux = uxInlet;
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( iCell == 0 ) // Inlet
	{
		BC.dirichletU = true;
		BC.ux = uxInlet;
		BC.uy = 0.f;
		BC.uz = 0.f;
		BC.nonReflective = false;
		BC.openBCID = 0;
	}
	else if ( iCell == Info.cellCountX-1  ) // Outlet
	{
		BC.dirichletRho = true;
		BC.dRho = 0.f;
		BC.nonReflective = true;
		BC.openBCID = 1;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	BC.collisionLimiter = 0.f; // use K15
	if ( BC.wallID >= 0 ) 
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/flowReportFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	// XY section cut
	const float zCut = 4.2f;
	exportSectionCutPlotXY( grids, zCut, iterationsFinished );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	
	// ZY section cuts
	//int counter = 1;
	//for ( float zCut = -25.f; zCut < 26.f; zCut += 25.f )
	//{
	//	exportSectionCutPlotXY( grids, zCut, iterationsFinished + counter );
	//	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	//	counter++;
	//}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathFilter );
	
	std::vector<STLStruct> rotorSTLs( 0 ); // there are no rotors
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	Tracker.customNames = { "Physical time", "Area averaged outlet pressure residual", "Area averaged pressure at measurement plane x=4.2mm", 
							"Area averaged pressure loss = measurement - outlet", "Uniform inlet velocity", "Weakly compressible density error at measurement plane", "Area averaged pressure at x=140mm" };
	Tracker.customUnits = { "s", "Pa", "Pa", "Pa", "m/s", "[1]", "Pa" };
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
			
			const float physicalTime = iterationsFinished * DT_PHYS_GLOBAL;
			
			// get pressure measurement flow report
			FlowReportStruct FlowReport; 
			BoundsStruct Bounds;
			Bounds = grids[0].Info.Bounds;
			float xCut = 4.2f;
			getFlowReportZY( FlowReport, grids, Bounds, xCut );
			
			// pressure loss
			const float pressureLoss = FlowReport.pressure - Tracker.pressure[1];
			
			// calculate density error
			const float soundspeedPhys = INVSQRT3 * (grids[0].Info.res/1000.f) / grids[0].Info.dtPhys;
			const float dRho = FlowReport.pressure / ( RHO_PHYS * soundspeedPhys * soundspeedPhys );
			const float densityError = dRho;
			
			// get pressure measurement flow report
			FlowReportStruct FlowReport2; 
			xCut = 140.f;
			getFlowReportZY( FlowReport2, grids, Bounds, xCut );
			
			// pass results to tracker
			trackCustomVariables( Tracker, { physicalTime, Tracker.pressure[1], FlowReport.pressure, pressureLoss, Tracker.normalVelocity[0], densityError, FlowReport2.pressure });
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
