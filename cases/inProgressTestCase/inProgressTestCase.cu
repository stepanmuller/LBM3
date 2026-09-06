constexpr float RES_GLOBAL = 2.0f; 	
constexpr int GRID_LEVEL_COUNT = 2;
constexpr int WALL_REFINEMENT_COUNT = 6;

const float dtPhysGlobal = 1.f;
const float nuPhys = 1.f;

#include "../../include/types.h"

std::string STLPathStator = "../../../../ns300/ns300_STATOR.STL";
std::string STLPathRotorShaft = "../../../../ns300/ns300_ROTOR_SHAFT.STL";
//std::string STLPathRotorBlades = "../../../../ns300/ns300_ROTOR_BLADES.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	return; // this just keeps the automatic default refinement setting
}

#include "../../include/gridGenerationFunctions.h"
#include "../../include/TEMPexportSectionCutPlot.h"

__host__ __device__ void getIJKCellIndexFromXYZ( int& iCell, int& jCell, int& kCell, const float &x, const float &y, const float &z, const InfoStruct &Info)
{
    iCell = (int)(( x - Info.ox ) / Info.res + 0.5f);
    jCell = (int)(( y - Info.oy ) / Info.res + 0.5f);
    kCell = (int)(( z - Info.oz ) / Info.res + 0.5f);
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 3 );
	readSTL( gridStaticSTLs[0], STLPathStator );
	readSTL( gridStaticSTLs[1], STLPathRotorShaft );
	
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
