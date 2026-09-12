constexpr float reynoldsNumber = 100000.f;
constexpr int cellsPerSphereDiameter = 512.f;

constexpr float sphereDiameterPhys = 1000.f;											// mm
constexpr float uxInlet = 0.015625f * 5.f; 												// also works as nominal LBM Mach number
constexpr float uxInletPhys = uxInlet; 													// m/s, physical velocity set to same as LBM velocity

constexpr int GRID_LEVEL_COUNT = 6;
constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr float RES_GLOBAL = (sphereDiameterPhys / cellsPerSphereDiameter) * (1u << (GRID_LEVEL_COUNT - 1));

constexpr float NU_PHYS = uxInletPhys * (sphereDiameterPhys / 1000.f) / reynoldsNumber;	// m2/s
constexpr float RHO_PHYS = 1.225f;														// kg/m3 air
constexpr float DT_PHYS_GLOBAL = (uxInlet / uxInletPhys) * (RES_GLOBAL/1000); 			// s

constexpr int ITERATION_COUNT = 60000 / 5;
constexpr int PLOTTER_PERIOD = 500;

#include "../../include/types.h"

std::string STLPathSphere = "sphere_D=1000mm.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"
#include "../../include/updateInterface.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r2 = y * y + z * z;
	const float xTransition = 1300.f;
	float xStart = 0.f; float xEnd = 0.f; float rLimit = 0.f;
	if ( Info.gridID == 0 )
	{
		xStart = -1300.f;
		xEnd = 3300.f;
		rLimit = 1500.f;
	}
	else if ( Info.gridID == 1 )
	{
		xStart = -950.f;
		xEnd = 3000.f;
		rLimit = 1100.f;
	}
	else if ( Info.gridID == 2 )
	{
		xStart = -750.f;
		xEnd = 2900.f;
		rLimit = 950.f;
	}
	else if ( Info.gridID == 3 )
	{
		xStart = -640.f;
		xEnd = 1300.f;
		rLimit = 850.f;
	}
	if ( Info.gridID < 4 )
	{
		refinementMarker = false;
		float r2Limit = rLimit * rLimit;
		if (x < xTransition)
		{
			const float s = (x - xTransition) / (xTransition - xStart);
			r2Limit *= 1.f - s * s;
		}
		if ( r2 < r2Limit && x > xStart && x < xEnd ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	//BC.ux = uxInlet;
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( iCell == Info.cellCountX-1 && jCell != 0 && jCell != Info.cellCountY-1 && kCell != 0 && kCell != Info.cellCountZ-1  ) // Outlet
	{
		BC.dirichletRho = true;
		BC.rho = 1.f;
		BC.openBCID = 1;
		BC.rhoReflectionTolerance = 0.00001f;
	}
	else if ( iCell == 0 && jCell != 0 && jCell != Info.cellCountY-1 && kCell != 0 && kCell != Info.cellCountZ-1  ) // Inlet
	{
		BC.dirichletU = true;
		BC.ux = uxInlet;
		BC.uy = 0.f;
		BC.uz = 0.f;
		BC.rhoReflectionTolerance = 1.f;
		BC.openBCID = 0;
	}
	else // Every other boundary cell is strict dirichlet velocity
	{
		BC.dirichletU = true;
		BC.ux = uxInlet;
		BC.uy = 0.f;
		BC.uz = 0.f;
		BC.rhoReflectionTolerance = 1.f;
		BC.openBCID = 2;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( BC.wallID == 0 ) 
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( Info.gridID == 0 ) BC.collisionLimiter = 0.f; // use more stable K15 collision for grid 0 where open BC happen
	//BC.collisionLimiter = 0.f;
}

void exportHistoryData( const std::vector<float>& historyDragCoefficient, 
                        const int &currentIteration, int fileNumber ) {
    FILE* fp = fopen("/dev/shm/historyData.bin", "wb");
    if (!fp) return;
    int count = std::min({ITERATION_COUNT-1, currentIteration}) + 1;
    fwrite(&count, sizeof(int), 1, fp);
    fwrite(historyDragCoefficient.data(), sizeof(float), count, fp);
    fclose(fp);
    std::string cmd = "python3 historyPlotter.py " + std::to_string(fileNumber) + " &";
    if (system(cmd.c_str()) != 0) {}
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/plotter/exportSectionCutPlot.h"

int main(int argc, char **argv)
{
	std::cout << RES_GLOBAL << std::endl;
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathSphere );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds.xMin = - 2000.f;
	DomainBounds.xMax =   9000.f;
	DomainBounds.yMin = - 5500.f;
	DomainBounds.yMax =   5500.f;
	DomainBounds.zMin = - 5500.f;
	DomainBounds.zMax =   5500.f;
	
	buildGrids( grids, gridStaticSTLs, DomainBounds );
	
	long long totalUpdatesPerIteration = 0LL;
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) totalUpdatesPerIteration += grids[level].Info.cellCount * std::pow( 2, grids[level].Info.gridID );
	
	std::vector<float> historyDragCoefficient( ITERATION_COUNT+1, 0.f );
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iteration = 0; iteration <= ITERATION_COUNT; iteration++ )
	{
		updateAllGrids( grids, 0 );
		
		float gxSum = TNL::sum( grids[GRID_LEVEL_COUNT-1].Wall.gxArray );
		grids[GRID_LEVEL_COUNT-1].Wall.gxArray.setValue( 0.f );
		gxSum /= (float)grids[GRID_LEVEL_COUNT-1].Info.updatesSinceTrackerReset;
		grids[GRID_LEVEL_COUNT-1].Info.updatesSinceTrackerReset = 0;
		float gy = 0.f; float gz = 0.f;
		convertToPhysicalForce( gxSum, gy, gz, grids[GRID_LEVEL_COUNT-1].Info );
		const float drag = - gxSum;
		const float dragCoefficient = - (8 * drag) / (RHO_PHYS * uxInletPhys * uxInletPhys * 3.14159f * (sphereDiameterPhys / 1000.f) * (sphereDiameterPhys / 1000.f));
		historyDragCoefficient[iteration] = dragCoefficient;
		/*
		OpenBCReportStruct InletReport;
		InletReport.uNormalPhys = TNL::sum( grids[0].openBCs[0].uNormalCumulativeArray ) / (float)grids[0].openBCs[0].openBCCount;
		grids[0].openBCs[0].uNormalCumulativeArray.setValue( 0.f );
		//historyDragCoefficient[iteration] = InletReport.uNormalPhys;
		
		OpenBCReportStruct OutletReport;
		OutletReport.rhoPhys = TNL::sum( grids[0].openBCs[1].rhoCumulativeArray ) / (float)grids[0].openBCs[1].openBCCount;
		grids[0].openBCs[1].rhoCumulativeArray.setValue( 0.f );
		historyDragCoefficient[iteration] = OutletReport.rhoPhys;
		*/		
		if ( iteration % PLOTTER_PERIOD == 0 )
		{
			lapTimer.stop();
			std::cout << std::endl;
			std::cout << "Finished iteration " << iteration << std::endl;
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)totalUpdatesPerIteration * (float)PLOTTER_PERIOD;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			if ( iteration > 0 ) exportHistoryData( historyDragCoefficient, iteration, 0 );
			
			// XY section cut
			const int kCut = grids[ GRID_LEVEL_COUNT-1 ].Info.cellCountZ / 2;
			exportSectionCutPlotXY( grids, kCut, iteration );
			if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			// Detail 1
			if ( GRID_LEVEL_COUNT > 1 )
			{
				exportSectionCutPlotXY( grids, grids[1].Info.Bounds, kCut, iteration + 1 );
				if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			}
			// Detail 2
			if ( GRID_LEVEL_COUNT > 2 )
			{
				exportSectionCutPlotXY( grids, grids[2].Info.Bounds, kCut, iteration + 2 );
				if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			}
			lapTimer.reset();
			lapTimer.start();
		}
	}
	
	return EXIT_SUCCESS;
}
