#pragma once

#include "./types.h"
#include "./markerFunctions.h"
#include "./voxelizerFunctions.h"

__cuda_callable__ inline void bitPackInterpolationCube( uint32_t& result, const uint8_t (&counter)[8] )
{
    result = 0u;
    for (unsigned int corner = 0; corner < 8; corner++ )
    {
        result |= static_cast<uint32_t>(counter[corner]) << (4u * corner);
    }
}

void buildRotors( GridStruct &Grid, std::vector<STLStruct> &rotorSTLs )
{
	// builds rotors for this grid level
	const int rotorCount = rotorSTLs.size();
	Grid.rotors.resize( rotorCount );
	for ( int rotorID = 0; rotorID < rotorCount; rotorID++ )
	{
		STLStruct &STL = rotorSTLs[ rotorID ];
		RotorStruct &Rotor = Grid.rotors[ rotorID ];
		VoxelizerStruct Voxelizer;
		
		// 1) set bounds
		Voxelizer.Info.ox = STL.Bounds.xMin;
		Voxelizer.Info.oy = STL.Bounds.yMin;
		Voxelizer.Info.oz = STL.Bounds.zMin;
		Voxelizer.Info.res = Grid.Info.res * 0.5f;
		Voxelizer.Info.cellCountX = (int)(( STL.Bounds.xMax - STL.Bounds.xMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.Info.cellCountY = (int)(( STL.Bounds.yMax - STL.Bounds.yMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.Info.cellCountZ = (int)(( STL.Bounds.zMax - STL.Bounds.zMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.rayMaps.resize( 1 );
		voxelizeSTL( Voxelizer.rayMaps[0], STL, Voxelizer );
		
		Rotor.rotorID = rotorID;
		Rotor.res = Voxelizer.Info.res;
		Rotor.Bounds.xMin = Voxelizer.Info.ox - 1.5f * Voxelizer.Info.res;
		Rotor.Bounds.yMin = Voxelizer.Info.oy - 1.5f * Voxelizer.Info.res;
		Rotor.Bounds.zMin = Voxelizer.Info.oz - 1.5f * Voxelizer.Info.res;
		// add overlap of 2 cells and then +3 /4 *4 to get closest upper multiple of 4
		Rotor.cellCountX = (( Voxelizer.Info.cellCountX + 2 + 3 ) / 4 ) * 4; // we want a multiple of 4 here
		Rotor.cellCountY = (( Voxelizer.Info.cellCountY + 2 + 3 ) / 4 ) * 4;
		Rotor.cellCountZ = (( Voxelizer.Info.cellCountZ + 2 + 3 ) / 4 ) * 4;
		Rotor.Bounds.xMax = Rotor.Bounds.xMin + Rotor.res * (float)Rotor.cellCountX;
		Rotor.Bounds.yMax = Rotor.Bounds.yMin + Rotor.res * (float)Rotor.cellCountY;
		Rotor.Bounds.zMax = Rotor.Bounds.zMin + Rotor.res * (float)Rotor.cellCountZ;
		
		const int blockCountX = Rotor.cellCountX/4;
		const int blockCountY = Rotor.cellCountY/4;
		const int blockCountZ = Rotor.cellCountZ/4;
		const int blockCountXY = blockCountX * blockCountY;
		const int blockCount = blockCountX * blockCountY * blockCountZ;
		const int voxelizerCountX = Voxelizer.Info.cellCountX;
		const int voxelizerCountY = Voxelizer.Info.cellCountY;
		
		// 2) Mark rotorMap blocks, if they contain any nonzero value
		BoolArrayType rotorMapMarkerArray( blockCount );
		rotorMapMarkerArray.setValue( false );
		// Each element in rotorMap tracks 4x4x4 interpolation cells
		// At vertices of those cells, there are counter points that count from 0 to 8
		// The count says how many points around the counter point are marked as 1 by the voxelizer
		// 4x4x4 interpolation cells -> 5x5x5 counter points -> 6x6x6 voxelizer points
		// First, mark elements of the rotorMap as 1 if any of the voxelizer points is 1
		auto rayMapView = Voxelizer.rayMaps[0].rayMapArray.getConstView();
		auto hitCounterScanView = Voxelizer.rayMaps[0].hitCounterScanArray.getConstView();
		auto rotorMapMarkerView = rotorMapMarkerArray.getView();
		auto rotorMapMarkerLambda = [=] __cuda_callable__ ( const int block ) mutable
		{
			const int kBlock = block / blockCountXY;
			const int remainder = block % blockCountXY;
			const int jBlock = remainder / blockCountX;
			const int iBlock = remainder % blockCountX;
			
			const int iVoxelizerStart = TNL::max( 0, iBlock * 4 - 2 );
			const int iVoxelizerEnd = TNL::min( voxelizerCountX, iBlock * 4 + 4 );
			const int jVoxelizerStart = TNL::max( 0, jBlock * 4 - 2);
			const int jVoxelizerEnd = TNL::min( voxelizerCountY, jBlock * 4 + 4 );
			const int kVoxelizerStart = kBlock * 4 - 2;
			const int kVoxelizerEnd = kBlock * 4 + 4;
			
			// here iCell, jCell, kCell refers to the voxelizer cells
			for ( int jCell = jVoxelizerStart; jCell < jVoxelizerEnd; jCell++ )
			{
				for ( int iCell = iVoxelizerStart; iCell < iVoxelizerEnd; iCell++ )
				{
					for ( int kCell = kVoxelizerStart; kCell < kVoxelizerEnd; kCell++ )
					{
						int kStart, kEnd;
						const int rayIndex = voxelizerCountX * jCell + iCell;
						const long long startingPoint = hitCounterScanView( rayIndex );
						const long long endingPoint = hitCounterScanView( rayIndex + 1 );
						for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
						{
							kStart = rayMapView( startIndex );
							if ( kStart > kCell ) break;
							kEnd = rayMapView( startIndex + 1LL );
							if ( kEnd > kCell )
							{
								rotorMapMarkerView( block ) = true;
								return;
							}
						}
					}
				}
			}			
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, blockCount, rotorMapMarkerLambda );
		
		// 3) Fill the rotorMap
		// -1 = all underlying interpolation cells are fully zero
		// >= 0 -> compact index of this non zero block 
		// also fill the indexArray
		const int nonZeroBlockCount = TNL::sum( rotorMapMarkerArray );
		Rotor.indexArray.setSize( nonZeroBlockCount );
		IntArrayType scanArray( blockCount );
		intArrayFromBoolArray( scanArray, rotorMapMarkerArray );
		TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, blockCount, TNL::Plus{} );
		Rotor.rotorMap.setSize( blockCount );
		auto rotorMapView = Rotor.rotorMap.getView();
		auto scanView = scanArray.getConstView();
		auto indexView = Rotor.indexArray.getView();
		
		auto scanLambda = [=] __cuda_callable__ ( const int block ) mutable
		{
			if ( rotorMapMarkerView( block ) ) 
			{
				const int scan = scanView( block );
				rotorMapView( block ) = scan;
				indexView( scan ) = block;
			}
			else rotorMapView( block ) = -1;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, blockCount, scanLambda );
		
		// 4) Build the interpolationArray
		Rotor.interpolationArray.setSize( nonZeroBlockCount * 64 );
		auto interpolationView = Rotor.interpolationArray.getView();
		// Loop over the non zero blocks captured in the indexArray
		// Inside the block, loop over the 6x6x6 voxelizer cells -> 216 bools
		// Then loop over the 5x5x5 counter points -> 125 uint8_t
		// Finally loop over the 4x4x4 interpolation cells
		// Bit pack the 8 relevant counter points into a single uint32_t
		// Write this uint as result
		auto resultLambda = [=] __cuda_callable__ ( const int index ) mutable
		{
			const int block = indexView( index );
			
			const int kBlock = block / blockCountXY;
			const int remainder = block % blockCountXY;
			const int jBlock = remainder / blockCountX;
			const int iBlock = remainder % blockCountX;
			
			const int iVoxelizerStart = TNL::max( 0, iBlock * 4 - 2 );
			const int iVoxelizerEnd = TNL::min( voxelizerCountX, iBlock * 4 + 4 );
			const int jVoxelizerStart = TNL::max( 0, jBlock * 4 - 2);
			const int jVoxelizerEnd = TNL::min( voxelizerCountY, jBlock * 4 + 4 );
			const int kVoxelizerStart = kBlock * 4 - 2;
			const int kVoxelizerEnd = kBlock * 4 + 4;
			
			bool voxelizerMark[216] = { false };
			
			// here iCell, jCell, kCell refers to the voxelizer cells
			for ( int jCell = jVoxelizerStart; jCell < jVoxelizerEnd; jCell++ )
			{
				for ( int iCell = iVoxelizerStart; iCell < iVoxelizerEnd; iCell++ )
				{
					for ( int kCell = kVoxelizerStart; kCell < kVoxelizerEnd; kCell++ )
					{
						int kStart, kEnd;
						const int rayIndex = voxelizerCountX * jCell + iCell;
						const long long startingPoint = hitCounterScanView( rayIndex );
						const long long endingPoint = hitCounterScanView( rayIndex + 1 );
						for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
						{
							kStart = rayMapView( startIndex );
							if ( kStart > kCell ) break;
							kEnd = rayMapView( startIndex + 1LL );
							if ( kEnd > kCell )
							{
								const int iLocal = iCell - (iBlock * 4 - 2);
								const int jLocal = jCell - (jBlock * 4 - 2);
								const int kLocal = kCell - (kBlock * 4 - 2);
								const int markIndex = kLocal * 36 + jLocal * 6 + iLocal;
								voxelizerMark[ markIndex ] = true;
							}
						}
					}
				}
			}		
			
			uint8_t counterPoint[125] = { 0u };
			for ( int kCounter = 0; kCounter < 5; kCounter++ )
			{
				for ( int jCounter = 0; jCounter < 5; jCounter++ )
				{
					for ( int iCounter = 0; iCounter < 5; iCounter++ )
					{
						int counter = 0;
						for ( int kMark = kCounter; kMark <= kCounter+1; kMark++ )
						{
							for ( int jMark = jCounter; jMark <= jCounter+1; jMark++ )
							{
								for ( int iMark = iCounter; iMark <= iCounter+1; iMark++ )
								{
									const int markIndex = kMark * 36 + jMark * 6 + iMark;
									if ( voxelizerMark[ markIndex ] ) counter++;
								}
							}
						}
						const int counterIndex = kCounter * 25 + jCounter * 5 + iCounter;
						counterPoint[ counterIndex ] = counter;
					}
				}
			}
			
			for ( int kInterpolation = 0; kInterpolation < 4; kInterpolation++ )
			{
				for ( int jInterpolation = 0; jInterpolation < 4; jInterpolation++ )
				{
					for ( int iInterpolation = 0; iInterpolation < 4; iInterpolation++ )
					{
						uint8_t smallCounter[8];
						int smallIndex = 0;
						for ( int kCounter = kInterpolation; kCounter <= kInterpolation+1; kCounter++ )
						{
							for ( int jCounter = jInterpolation; jCounter <= jInterpolation+1; jCounter++ )
							{
								for ( int iCounter = iInterpolation; iCounter <= iInterpolation+1; iCounter++ )
								{
									const int counterIndex = kCounter * 25 + jCounter * 5 + iCounter;
									smallCounter[ smallIndex ] = counterPoint[ counterIndex ];
									smallIndex++;
								}
							}
						}
						uint32_t result = 0u;
						bitPackInterpolationCube( result, smallCounter );
						const int interpolationIndex = index * 64 + kInterpolation * 16 + jInterpolation * 4 + iInterpolation;
						interpolationView( interpolationIndex ) = result;
					}
				}
			}
			
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, nonZeroBlockCount, resultLambda );
		
		Rotor.gxArray.setSize( nonZeroBlockCount * 27 );
		Rotor.gyArray.setSize( nonZeroBlockCount * 27 );
		Rotor.gzArray.setSize( nonZeroBlockCount * 27 );
		Rotor.gxArray.setValue( 0.f );
		Rotor.gyArray.setValue( 0.f );
		Rotor.gzArray.setValue( 0.f );
	}
	
	long long memoryBytes = 0LL;
	for ( int rotorID = 0; rotorID < (int)Grid.rotors.size(); rotorID++ )
	{
		memoryBytes += 1LL * (long long)Grid.rotors[ rotorID ].rotorMap.getSize() * 4LL; // rotorMap
		memoryBytes += 65LL * (long long)Grid.rotors[ rotorID ].indexArray.getSize() * 4LL; // indexArray, interpolationArray
		memoryBytes += 81LL * (long long)Grid.rotors[ rotorID ].indexArray.getSize() * 4LL; // rotor force tracker
	}
	std::cout << "	Allocated rotors for grid level " << Grid.Info.gridID << ", they take " << memoryBytes / 1048576.0 << " MiB" << std::endl;
}
