constexpr float RES_GLOBAL = 2.0f; 	
constexpr int GRID_LEVEL_COUNT = 4;
constexpr int WALL_REFINEMENT_COUNT = 6;

const float dtPhysGlobal = 1.f;
const float nuPhys = 1.f;

#include "../../include/types.h"

std::string STLPathStator = "../../../../ns300/ns300_STATOR_ENLARGED_TIP_GAP.STL";
std::string STLPathRotorShaft = "../../../../ns300/ns300_ROTOR_SHAFT.STL";
std::string STLPathRotorBlades = "../../../../ns300/ns300_ROTOR_BLADES.STL";

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
		if ( rz < 170.f && z < 140.f ) refinementMarker = true;
	}
	if ( Info.gridID == 2 ) // additional refinement for the tip gap
	{
		refinementMarker = false;
		if ( rz < 134.f && rz > 129.f && z < 126.5f && z > 103.5f ) refinementMarker = true;
	}
	if ( Info.gridID == 3 ) // additional refinement for the tip gap
	{
		refinementMarker = false;
		if ( rz < 134.f && rz > 129.5f && z < 125.5f && z > 104.5f ) refinementMarker = true;
	}
	return; // this just keeps the automatic default refinement setting
}

#include "../../include/gridGenerationFunctions.h"
#include "../../include/TEMPexportSectionCutPlot.h"

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 3 );
	readSTL( gridStaticSTLs[0], STLPathStator );
	readSTL( gridStaticSTLs[1], STLPathRotorShaft );
	readSTL( gridStaticSTLs[2], STLPathRotorBlades );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMax = 400.f;
	DomainBounds.yMax = 1000.f;
	initializeGridInfo( grids, DomainBounds, 0 );
	
	// Voxelizers
	std::vector<VoxelizerStruct> voxelizers( GRID_LEVEL_COUNT );
	initializeVoxelizers( voxelizers, grids, gridStaticSTLs, 0 );
	
	buildIJKFull( grids, voxelizers, 0 );
	
	deleteExcessCells( grids, voxelizers, 0 );
	
	buildWallMarkers( grids, voxelizers, 0 );
	
	int iCut, jCut, kCut;
	const float xTemp = 0.f; const float yTemp = 0.f; const float zTemp = 0.f;
	
	// ZY section cut shows the inlet pipe
	float xCut = 0.f;
	getIJKCellIndexFromXYZ( iCut, jCut, kCut, xCut, yTemp, zTemp, grids[GRID_LEVEL_COUNT-1].Info);
	exportSectionCutPlotZY( grids, iCut, 0 );
	if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
	
	// XY section cut shows the rotor and the outlet pipe
	float zCut = 32.5f;
	getIJKCellIndexFromXYZ( iCut, jCut, kCut, xTemp, yTemp, zCut, grids[GRID_LEVEL_COUNT-1].Info);
	exportSectionCutPlotXY( grids, kCut, 1 );
	if (system("python3 ../../include/plotter/OLDplotter.py") != 0) {}
		
	return EXIT_SUCCESS;
}
