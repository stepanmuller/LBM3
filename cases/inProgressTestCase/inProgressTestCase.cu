constexpr float resGlobal = 0.08f; 														// mm

#include "../../include/types.h"

std::string STLPathStator = "M-Jet_35_pump_main.STL";
std::string STLPathRotor = "M-Jet_35_impeller.STL";
std::string STLPathShaft = "M-Jet_35_shaft.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"

int main(int argc, char **argv)
{
	// STLs
	STLStruct STLStator;
	readSTL( STLStator, STLPathStator );
	STLStruct STLRotor;
	readSTL( STLRotor, STLPathRotor );
	STLStruct STLShaft;
	readSTL( STLShaft, STLPathShaft );

	// Voxelizer 
	VoxelizerStruct Voxelizer;
	Voxelizer.Info.res = resGlobal;
	Voxelizer.Info.cellCountX = 370;
	Voxelizer.Info.cellCountY = 370;
	Voxelizer.Info.ox = -18.45f;
	Voxelizer.Info.oy = -18.45f;
	Voxelizer.Info.oz = -74.4f;
	Voxelizer.rayMaps.resize( 3 );
	voxelizeSTL( Voxelizer.rayMaps[0], STLStator, Voxelizer );
	voxelizeSTL( Voxelizer.rayMaps[1], STLRotor, Voxelizer );
	voxelizeSTL( Voxelizer.rayMaps[2], STLShaft, Voxelizer );
	Voxelizer.rayMapTotal = Voxelizer.rayMaps[0];
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMaps[1] );
	sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMaps[2] );
		
	return EXIT_SUCCESS;
}
