#pragma once

#include "./types.h"

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
		
		Voxelizer.Info.ox = STL.Bounds.xMin;
		Voxelizer.Info.oy = STL.Bounds.yMin;
		Voxelizer.Info.oz = STL.Bounds.zMin;
		Voxelizer.Info.res = Grid.Info.res * 0.5f;
		Voxelizer.Info.cellCountX = (int)(( STL.Bounds.xMax - STL.Bounds.xMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.Info.cellCountY = (int)(( STL.Bounds.yMax - STL.Bounds.yMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.Info.cellCountZ = (int)(( STL.Bounds.zMax - STL.Bounds.zMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.rayMaps.resize( 1 );
		voxelizeSTL( Voxelizer.rayMaps[0], STL, Voxelizer );
		
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
		Rotor.rotorMap.setSizes( rotorMapSize );
		BoolArrayType rotorMapMarker( rotorMapSize );
		// Each element in rotorMap tracks 4x4x4 interpolation cells
		// At vertices of those cells, there are counter points that count from 0 to 8
		// The count says how many points around the counter point are marked as 1 by the voxelizer
		// 4x4x4 interpolation cells -> 5x5x5 counter points
		// First, mark elements of the rotorMap as 1 if any of its counter points is > 0
		auto rotorMapMarkerLambda = [=] __cuda_callable__ ( const int index ) mutable
		{
			const int iBlock = iView[ cell ];
			const int jBlock = jView[ cell ];
			const int kBlock = kView[ cell ];
			int kStart, kEnd;
			const int rayIndex = cellCountX * jCell + iCell;
			const long long startingPoint = hitCounterScanView( rayIndex );
			const long long endingPoint = hitCounterScanView( rayIndex + 1 );
			for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
			{
				kStart = rayMapView( startIndex );
				if ( kStart > kCell ) break;
				kEnd = rayMapView( startIndex + 1LL );
				if ( kEnd > kCell )
				{
					markerView[ cell ] = true;
					return;
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, rotorMapSize, rotorMapMarkerLambda );	
	}
}
