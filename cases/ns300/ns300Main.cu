constexpr float RES_GLOBAL = 2.0f; 	
constexpr int GRID_LEVEL_COUNT = 3;
constexpr int WALL_REFINEMENT_COUNT = 6;

int reportChunk = 31;
int plotterChunk = 200;
constexpr int iterationCount = 20000;

constexpr float uzInlet = 0.01f; 														// also works as nominal LBM Mach number	
constexpr float nuPhys = 1e-6;															// m2/s water
constexpr float rhoNominalPhys = 997.0f;												// kg/m3 water
constexpr float massFlowPhys = 335.f;													// kg/s
constexpr float RInlet = 150.f;															// mm
// constexpr float ROutlet = 175.f;														// mm
constexpr float inletAreamm2 = 3.14159f * RInlet * RInlet;								// mm2
constexpr float uzInletPhys = massFlowPhys / ( rhoNominalPhys * ( inletAreamm2 / 1000000.f) );	// m/s
constexpr float dtPhysGlobal = (uzInlet / uzInletPhys) * (RES_GLOBAL/1000.f); 			// s
// constexpr float soundspeedPhys = 0.577350269f * (RES_GLOBAL/1000.f) / dtPhysGlobal; 		// m/s (0.577350269f is 1/sqrt(3))
constexpr float angularVelocity = -198.967f;											// rad/s
const float boundaryLayerThickness = 2.f;												// mm

#include "../../include/types.h"

std::string STLPathStator = "../../../../ns300/ns300_STATOR_ENLARGED_TIP_GAP.STL";
//std::string STLPathStator = "../../../../ns300/ns300_STATOR.STL";
std::string STLPathRotorShaft = "../../../../ns300/ns300_ROTOR_SHAFT.STL";
std::string STLPathRotorBlades = "../../../../ns300/ns300_ROTOR_BLADES.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"
#include "../../include/updateInterface.h"

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
		if ( rz < 170.f && z < 140.f ) refinementMarker = true;
	}
	if ( Info.gridID == 2 ) // additional refinement for the tip gap
	{
		refinementMarker = false;
		if ( rz < 134.f && rz > 129.4f && z < 126.5f && z > 103.5f ) refinementMarker = true;
	}
	if ( Info.gridID == 3 ) // additional refinement for the tip gap
	{
		refinementMarker = false;
		if ( rz < 130.5f && rz > 129.8f && z < 125.5f && z > 104.5f ) refinementMarker = true;
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
	}
	else if ( jCell == Info.cellCountY-1 ) // Outlet
	{
		BC.dirichletRho = true;
		BC.openBCID = 1;
		BC.rho = 1.f;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r = std::sqrt( x * x + y * y );
	const float vtPhys = angularVelocity * (r / 1000.f);
	const float vt = vtPhys * ( uzInlet / uzInletPhys );
	if ( BC.wallID == 0 ) 
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( BC.wallID == 1 || BC.wallID == 2 ) 
	{
		BC.ux = - vt * (y / r);
		BC.uy = vt * (x / r);
		BC.uz = 0.f;
	}
	if ( kCell >= Info.cellCountZ-20 || jCell >= Info.cellCountY-20 ) BC.collisionLimiter = 0.f;
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/plotter/exportSectionCutPlot.h"

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 2 );
	readSTL( gridStaticSTLs[0], STLPathStator );
	readSTL( gridStaticSTLs[1], STLPathRotorShaft );
	// readSTL( gridStaticSTLs[2], STLPathRotorBlades );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMax = 400.f;
	DomainBounds.yMax = 1000.f;
	
	buildGrids( grids, gridStaticSTLs, DomainBounds );
	
	long long totalUpdatesPerIteration = 0LL;
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) totalUpdatesPerIteration += grids[level].Info.cellCount * std::pow( 2, grids[level].Info.gridID );
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iteration = 0; iteration <= iterationCount; iteration++ )
	{
		updateAllGrids( grids, 0 );
		
		if ( iteration % plotterChunk == 0 )
		{
			lapTimer.stop();
			std::cout << std::endl;
			std::cout << "Finished iteration " << iteration << std::endl;
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)totalUpdatesPerIteration * (float)plotterChunk;
			const float glups = updateCount / lapTime / 1000000000.f;
			if ( iteration > 0) std::cout << "GLUPS: " << glups << std::endl;
			
			int iCut, jCut, kCut;
			const float xTemp = 0.f; const float yTemp = 0.f; const float zTemp = 0.f;
			
			// ZY section cut shows the inlet pipe
			float xCut = 0.f;
			getIJKCellIndexFromXYZ( iCut, jCut, kCut, xCut, yTemp, zTemp, grids[GRID_LEVEL_COUNT-1].Info);
			exportSectionCutPlotZY( grids, iCut, iteration + 0 );
			if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			
			// Detail
			if ( GRID_LEVEL_COUNT >= 3 )
			{
				exportSectionCutPlotZY( grids, grids[2].Info.Bounds, iCut, iteration + 1 );
				if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			}
			// XY section cut shows the rotor and the outlet pipe
			float zCut = 32.5f;
			getIJKCellIndexFromXYZ( iCut, jCut, kCut, xTemp, yTemp, zCut, grids[GRID_LEVEL_COUNT-1].Info);
			exportSectionCutPlotXY( grids, kCut, iteration + 2 );
			if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			
			// Detail
			if ( GRID_LEVEL_COUNT >= 3 )
			{
				exportSectionCutPlotXY( grids, grids[2].Info.Bounds, kCut, iteration + 3 );
				if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
			}
			lapTimer.reset();
			lapTimer.start();
		}
	}
	
	return EXIT_SUCCESS;
}
