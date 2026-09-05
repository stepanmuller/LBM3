constexpr float RES_GLOBAL = 3.0f; 	
constexpr int GRID_LEVEL_COUNT = 2;

const float dtPhysGlobal = 1.f;
const float nuPhys = 1.f;

#include "../../include/types.h"

std::string STLPathStator = "../../../../ns300/ns300_STATOR.STL";
std::string STLPathRotorShaft = "../../../../ns300/ns300_ROTOR_SHAFT.STL";
std::string STLPathRotorBlades = "../../../../ns300/ns300_ROTOR_BLADES.STL";

#include "../../include/STLFunctions.h"
#include "../../include/gridGenerationFunctions.h"
#include "../../include/voxelizerFunctions.h"

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 2 );
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
		
	return EXIT_SUCCESS;
}
