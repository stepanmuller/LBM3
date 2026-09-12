#pragma once

#include "./types.h"
#include "./NBRFunctions.h"

void markWallCells( BoolArrayType &markerArray, const RayMapStruct &rayMap, const GridBuilderStruct &GridBuilder )
{
	const int &cellCount = GridBuilder.Info.cellCount;
	const int &cellCountX = GridBuilder.Info.cellCountX;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	const IntArrayType &rayMapArray = rayMap.rayMapArray;
	const LongLongArrayType &hitCounterScanArray = rayMap.hitCounterScanArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	auto hitCounterScanView = hitCounterScanArray.getConstView();
	
	markerArray.setValue( false );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
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
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}

void markSingleFinerFluid( BoolArrayType &markerArray, const RayMapStruct &rayMap, const SkeletonGridStruct &SkeletonGrid )
{
	// marks the skeleton grid based on a finer rayMapArray, result is 1 if at least one fine cell is 0 (fluid)
	const int cellCountX = SkeletonGrid.Info.cellCountX;
	const int cellCountY = SkeletonGrid.Info.cellCountY;
	// const int cellCountZ = SkeletonGrid.Info.cellCountZ; // this is not needed
	const int cellCount = SkeletonGrid.Info.cellCount;
	const IntArrayType &rayMapArray = rayMap.rayMapArray;
	const LongLongArrayType &hitCounterScanArray = rayMap.hitCounterScanArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	auto hitCounterScanView = hitCounterScanArray.getConstView();
	
	const int levelDifference = rayMap.gridID - (-1); // skeleton grid would have gridID = -1
	const int downsample = 1 << levelDifference;
	
	markerArray.setValue( true );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int kCoarse = cell / (cellCountX * cellCountY);
		const int remainder = cell % (cellCountX * cellCountY);
		const int jCoarse = remainder / cellCountX;
		const int iCoarse = remainder % cellCountX;
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				const int rayIndex = ( cellCountX * downsample ) * jFine + iFine;
				const long long startingPoint = hitCounterScanView( rayIndex );
				const long long endingPoint = hitCounterScanView( rayIndex + 1 );
				if (startingPoint == endingPoint) return; // No solid intervals: fluid exists
				for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
				{
					kEnd = rayMapView( startIndex + 1LL );
					if ( kEnd < kFineFirst )
					{
						if ( startIndex + 2LL < endingPoint ) continue; // continue browsing the next interval if there still is one
						else return; // if this was the last interval, return to mark this as fluid
					}
					else if ( kEnd >= kFineFirst && kEnd <= kFineLast ) return;
					kStart = rayMapView( startIndex );
					if ( kStart <= kFineFirst ) break;
					else return;
				}
			}
		}
		markerView[ cell ] = false;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}

void markSingleFinerFluid( BoolArrayType &markerArray, const RayMapStruct &rayMap, const GridBuilderStruct &GridBuilder )
{
	// marks a coarse grid based on a finer rayMapArray, result is 1 if at least one fine cell is 0 (fluid)
	const int cellCountX = GridBuilder.Info.cellCountX;
	const int cellCount = GridBuilder.Info.cellCount;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	const IntArrayType &rayMapArray = rayMap.rayMapArray;
	const LongLongArrayType &hitCounterScanArray = rayMap.hitCounterScanArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	auto hitCounterScanView = hitCounterScanArray.getConstView();
	
	const int levelDifference = rayMap.gridID - (GridBuilder.Info.gridID);
	const int downsample = 1 << levelDifference;
	
	markerArray.setValue( true );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCoarse = iView[ cell ];
		const int jCoarse = jView[ cell ];
		const int kCoarse = kView[ cell ];
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				const int rayIndex = ( cellCountX * downsample ) * jFine + iFine;
				const long long startingPoint = hitCounterScanView( rayIndex );
				const long long endingPoint = hitCounterScanView( rayIndex + 1 );
				if (startingPoint == endingPoint) return; // No solid intervals: fluid exists
				for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
				{
					kEnd = rayMapView( startIndex + 1 );
					if ( kEnd < kFineFirst ) 
					{
						if ( startIndex + 2LL < endingPoint ) continue; // continue browsing the next interval if there still is one
						else return; // if this was the last interval, return to mark this as fluid
					}
					else if ( kEnd >= kFineFirst && kEnd <= kFineLast ) return;
					kStart = rayMapView( startIndex );
					if ( kStart <= kFineFirst ) break;
					else return;
				}
			}
		}
		markerView[ cell ] = false;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}

void markSingleFinerWall( BoolArrayType &markerArray, const RayMapStruct &rayMap, const GridBuilderStruct &GridBuilder )
{
	// marks a coarse grid based on a fine rayMapArray, result is 1 if at least one fine cell is 1 (wall)
	const int cellCountX = GridBuilder.Info.cellCountX;
	const int cellCount = GridBuilder.Info.cellCount;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	const IntArrayType &rayMapArray = rayMap.rayMapArray;
	const LongLongArrayType &hitCounterScanArray = rayMap.hitCounterScanArray;
	auto markerView = markerArray.getView();
	auto rayMapView = rayMapArray.getConstView();
	auto hitCounterScanView = hitCounterScanArray.getConstView();
	
	const int levelDifference = rayMap.gridID - (GridBuilder.Info.gridID);
	const int downsample = 1 << levelDifference;
	
	markerArray.setValue( false );

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCoarse = iView[ cell ];
		const int jCoarse = jView[ cell ];
		const int kCoarse = kView[ cell ];
		const int iFineFirst = iCoarse * downsample;
		const int jFineFirst = jCoarse * downsample;
		const int kFineFirst = kCoarse * downsample;
		const int kFineLast = kFineFirst + downsample - 1;
		int iFine, jFine, kStart, kEnd;
		for ( int jAdd = 0; jAdd < downsample; jAdd++ )
		{
			jFine = jFineFirst + jAdd; 
			for ( int iAdd = 0; iAdd < downsample; iAdd++ )
			{
				iFine = iFineFirst + iAdd;
				const int rayIndex = ( cellCountX * downsample ) * jFine + iFine;
				const long long startingPoint = hitCounterScanView( rayIndex );
				const long long endingPoint = hitCounterScanView( rayIndex + 1 );
				for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
				{
					kStart = rayMapView( startIndex );
					if ( kStart > kFineLast ) break;
					else if ( kStart >= kFineFirst ) 
					{
						markerView[ cell ] = true;
						return;
					}
					kEnd = rayMapView( startIndex + 1 );
					if ( kEnd <= kFineFirst ) continue;
					else 
					{
						markerView[ cell ] = true;
						return;
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}

void markAllFinerFluids( BoolArrayType &resultArray, const std::vector<VoxelizerStruct> &voxelizers, const SkeletonGridStruct &SkeletonGrid )
{
	resultArray.setValue( false );
	BoolArrayType markerArray( resultArray.getSize() );
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
	{
		markSingleFinerFluid( markerArray, voxelizers[level].rayMapTotal, SkeletonGrid );
		resultArray += markerArray;
	}
}

void markAllFinerFluids( BoolArrayType &resultArray, const std::vector<VoxelizerStruct> &voxelizers, const GridBuilderStruct &GridBuilder )
{
	resultArray.setValue( false );
	BoolArrayType markerArray( resultArray.getSize() );
	for ( int level = GridBuilder.Info.gridID; level < GRID_LEVEL_COUNT; level++ )
	{
		markSingleFinerFluid( markerArray, voxelizers[level].rayMapTotal, GridBuilder );
		resultArray += markerArray;
	}
}

void applyUserRefinementModification( BoolArrayType &markerArray, const GridBuilderStruct &GridBuilder )
{
	// uses the getRefinementModifier function defined in the main file to adjust refinement area
	const InfoStruct &Info = GridBuilder.Info;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	auto markerView = markerArray.getView();

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		bool refinementMarker = markerView( cell );		
		getRefinementModifier( iCell, jCell, kCell, refinementMarker, Info );
		markerView( cell ) = refinementMarker;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
}

void markAllFinerWalls( BoolArrayType &resultArray, const std::vector<VoxelizerStruct> &voxelizers, const GridBuilderStruct &GridBuilder )
{
	resultArray.setValue( false );
	BoolArrayType markerArray( resultArray.getSize() );
	for ( int level = GridBuilder.Info.gridID; level < GRID_LEVEL_COUNT; level++ )
	{
		markSingleFinerWall( markerArray, voxelizers[level].rayMapTotal, GridBuilder );
		resultArray += markerArray;
	}
}

void spreadMarkers( BoolArrayType &targetMarkerArray, const BoolArrayType &sourceMarkerArray, SkeletonGridStruct &SkeletonGrid )
{
	auto targetMarkerView = targetMarkerArray.getView();
	auto sourceMarkerView = sourceMarkerArray.getConstView();
	const int cellCountX = SkeletonGrid.Info.cellCountX;
	const int cellCountY = SkeletonGrid.Info.cellCountY;
	const int cellCountZ = SkeletonGrid.Info.cellCountZ;
	const int cellCount = SkeletonGrid.Info.cellCount;

	targetMarkerArray = sourceMarkerArray; // initialize as source
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		bool marker = sourceMarkerView[ cell ];
		if ( marker ) return; // only continue if the marker is not already 1
		const int kCell = cell / (cellCountX * cellCountY);
		const int remainder = cell % (cellCountX * cellCountY);
		const int jCell = remainder / cellCountX;
		const int iCell = remainder % cellCountX;
		int nbr, iNbr, jNbr, kNbr;
		for ( int kAdd = -1; kAdd <= 1; kAdd++ )
		{
			kNbr = kCell + kAdd;
			if ( kNbr >= 0 && kNbr < cellCountZ )
			{
				for ( int jAdd = -1; jAdd <= 1; jAdd++ )
				{
					jNbr = jCell + jAdd;
					if ( jNbr >= 0 && jNbr < cellCountY )
					{
						for ( int iAdd = -1; iAdd <= 1; iAdd++ )
						{
							if ( kAdd!=0 || jAdd!=0 || iAdd!=0 )
							{
								iNbr = iCell + iAdd;
								if ( iNbr >= 0 && iNbr < cellCountX )
								{
									nbr = kNbr * (cellCountX * cellCountY) + jNbr * cellCountX + iNbr;
									if ( sourceMarkerView[ nbr ] )
									{
										targetMarkerView[ cell ] = true;
										return;
									}
								}
							}
						}
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}

void spreadMarkers( BoolArrayType &targetMarkerArray, const BoolArrayType &sourceMarkerArray, GridBuilderStruct &GridBuilder )
{
	// The way this is written creates a race condition, one that is harmless because all threads write the same 1
	const int &cellCount = GridBuilder.Info.cellCount;
	auto targetMarkerView = targetMarkerArray.getView();
	auto sourceMarkerView = sourceMarkerArray.getConstView();
	auto jPlusView = GridBuilder.NBR.jPlusArray.getConstView();
	auto kPlusView = GridBuilder.NBR.kPlusArray.getConstView();
	auto isGeometricBitPackedMarkerView = GridBuilder.NBR.isGeometricBitPackedMarkerArray.getView();
	
	targetMarkerArray = sourceMarkerArray; // initialize as source

	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		int nbrPlus[7];
		nbrPlus[0] = (cell + 1 < cellCount) ? cell + 1 : 0; 					// iPlus
		nbrPlus[1] = jPlusView[ cell ];											// jPlus
		nbrPlus[2] = (nbrPlus[1] + 1 < cellCount) ? nbrPlus[1] + 1 : 0;		// ijPlus
		nbrPlus[3] = kPlusView[ cell ];											// kPlus
		nbrPlus[4] = (nbrPlus[3] + 1 < cellCount) ? nbrPlus[3] + 1 : 0;		// ikPlus
		nbrPlus[5] = jPlusView[ nbrPlus[3] ];									// jkPlus
		nbrPlus[6] = (nbrPlus[5] + 1 < cellCount) ? nbrPlus[5] + 1 : 0;		// ijkPlus
		
		bool isGeometricMarker[8] = {false};
		const uint8_t isGeometricBitPack = isGeometricBitPackedMarkerView( cell );
		byteToBools( isGeometricBitPack, isGeometricMarker );
		
		bool marker = sourceMarkerView[ cell ];
		if ( !marker )
		{
			for ( int q = 0; q < 7; q++ )
			{
				if ( isGeometricMarker[q] )
				{
					if ( sourceMarkerView[nbrPlus[q]] )
					{
						marker = true;
						break;
					}
				}
			}
		}
		if ( marker )
		{
			targetMarkerView[ cell ] = true; // <- race condition here
			for ( int q = 0; q < 7; q++ )
			{
				if ( isGeometricMarker[q] )
				{
					targetMarkerView[nbrPlus[q]] = true; // <- race condition here too <3
				}
			}
		}		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridBuilder.Info.cellCount, cellLambda );	
}

void markKeepCells( SkeletonGridStruct &SkeletonGrid, const std::vector<VoxelizerStruct> &voxelizers )
{
	markAllFinerFluids( SkeletonGrid.keepCellMarkerArray, voxelizers, SkeletonGrid );
	BoolArrayType markerSource;
	markerSource = SkeletonGrid.keepCellMarkerArray;
	spreadMarkers( SkeletonGrid.keepCellMarkerArray, markerSource, SkeletonGrid );
}

void markKeepCells( GridBuilderStruct &GridBuilder, const std::vector<VoxelizerStruct> &voxelizers )
{
	markAllFinerFluids( GridBuilder.keepCellMarkerArray, voxelizers, GridBuilder );
	BoolArrayType markerSource;
	markerSource = GridBuilder.keepCellMarkerArray;
	spreadMarkers( GridBuilder.keepCellMarkerArray, markerSource, GridBuilder );
	// A fine cell located at the parent interface is blocked from getting deleted later
	if ( GridBuilder.Info.gridID > 0 )	GridBuilder.keepCellMarkerArray += GridBuilder.parentInterfaceMarkerArray; 
}

void markRefinementCells( GridBuilderStruct &GridBuilder, const std::vector<VoxelizerStruct> &voxelizers )
{
	markKeepCells( GridBuilder, voxelizers );
	// search deep refinement area
	markAllFinerWalls( GridBuilder.deepRefinementMarkerArray, voxelizers, GridBuilder );
	
	BoolArrayType markerBuffer( GridBuilder.deepRefinementMarkerArray.getSize() );
	for ( int spread = 0; spread < WALL_REFINEMENT_COUNT; spread++ )
	{
		GridBuilder.deepRefinementMarkerArray.swap( markerBuffer );
		spreadMarkers( GridBuilder.deepRefinementMarkerArray, markerBuffer, GridBuilder );
	}
	applyUserRefinementModification( GridBuilder.deepRefinementMarkerArray, GridBuilder );
	// A fine cell located at the parent interface is blocked from getting deeply refined (interface with finer grid is still allowed)
	if ( GridBuilder.Info.gridID > 0 )	GridBuilder.deepRefinementMarkerArray *= !GridBuilder.parentInterfaceMarkerArray;
	GridBuilder.deepRefinementMarkerArray = GridBuilder.deepRefinementMarkerArray * GridBuilder.keepCellMarkerArray;
	// search fine to coarse interface
	GridBuilder.fineToCoarseMarkerArray = GridBuilder.deepRefinementMarkerArray;
	GridBuilder.fineToCoarseMarkerArray.swap( markerBuffer );
	spreadMarkers( GridBuilder.fineToCoarseMarkerArray, markerBuffer, GridBuilder );
	GridBuilder.fineToCoarseMarkerArray = GridBuilder.fineToCoarseMarkerArray * GridBuilder.keepCellMarkerArray * !GridBuilder.deepRefinementMarkerArray;
	// search coarse to fine interface
	GridBuilder.coarseToFineMarkerArray = GridBuilder.fineToCoarseMarkerArray;
	GridBuilder.coarseToFineMarkerArray.swap( markerBuffer );
	spreadMarkers( GridBuilder.coarseToFineMarkerArray, markerBuffer, GridBuilder );
	GridBuilder.coarseToFineMarkerArray = GridBuilder.coarseToFineMarkerArray * GridBuilder.keepCellMarkerArray * !GridBuilder.deepRefinementMarkerArray * !GridBuilder.fineToCoarseMarkerArray;
	// mark refinement all together
	GridBuilder.refinementMarkerArray = GridBuilder.deepRefinementMarkerArray + GridBuilder.fineToCoarseMarkerArray + GridBuilder.coarseToFineMarkerArray;
}
