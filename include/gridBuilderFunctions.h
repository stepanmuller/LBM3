#pragma once

#include "./types.h"
#include "./markerFunctions.h"
#include "./IBBLinkBuilderFunctions.h"

void initializeGridInfo( std::vector<GridBuilderStruct> &gridBuilders, const BoundsStruct &Bounds, const int level )
{
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridBuilderStruct &GridBuilder = gridBuilders[ level ];
	InfoStruct &Info = GridBuilder.Info;
	
	if ( iAmCoarsest )
	{
		Info.res = RES_GLOBAL;
		
		SkeletonGridStruct &SkeletonGrid = GridBuilder.SkeletonGrid;
		InfoStruct &SkeletonInfo = SkeletonGrid.Info;
		SkeletonInfo.res = Info.res * 2.f;
		SkeletonInfo.cellCountX = static_cast<int>((Bounds.xMax - Bounds.xMin) / SkeletonInfo.res);
		SkeletonInfo.cellCountY = static_cast<int>((Bounds.yMax - Bounds.yMin) / SkeletonInfo.res);
		SkeletonInfo.cellCountZ = static_cast<int>((Bounds.zMax - Bounds.zMin) / SkeletonInfo.res);
		SkeletonInfo.cellCount = SkeletonInfo.cellCountX * SkeletonInfo.cellCountY * SkeletonInfo.cellCountZ;
		SkeletonInfo.ox = Bounds.xMin + ((Bounds.xMax - Bounds.xMin) - (SkeletonInfo.cellCountX * SkeletonInfo.res) + SkeletonInfo.res) / 2.0f;
		SkeletonInfo.oy = Bounds.yMin + ((Bounds.yMax - Bounds.yMin) - (SkeletonInfo.cellCountY * SkeletonInfo.res) + SkeletonInfo.res) / 2.0f;
		SkeletonInfo.oz = Bounds.zMin + ((Bounds.zMax - Bounds.zMin) - (SkeletonInfo.cellCountZ * SkeletonInfo.res) + SkeletonInfo.res) / 2.0f;
			
		Info.cellCountX = SkeletonInfo.cellCountX * 2;
		Info.cellCountY = SkeletonInfo.cellCountY * 2;
		Info.cellCountZ = SkeletonInfo.cellCountZ * 2;
		Info.ox = SkeletonInfo.ox - Info.res * 0.5f;
		Info.oy = SkeletonInfo.oy - Info.res * 0.5f;
		Info.oz = SkeletonInfo.oz - Info.res * 0.5f;
		
		Info.dtPhys = dtPhysGlobal;
		Info.nu = (Info.dtPhys * nuPhys) / ((Info.res/1000.f) * (Info.res/1000.f));
	}
	
	else
	{
		Info.gridID = level;
		GridBuilderStruct &GridBuilderCoarse = gridBuilders[ level-1 ];
		Info.res = GridBuilderCoarse.Info.res * 0.5f;
		Info.cellCountX = GridBuilderCoarse.Info.cellCountX * 2;
		Info.cellCountY = GridBuilderCoarse.Info.cellCountY * 2;
		Info.cellCountZ = GridBuilderCoarse.Info.cellCountZ * 2;
		Info.ox = GridBuilderCoarse.Info.ox - Info.res * 0.5f;
		Info.oy = GridBuilderCoarse.Info.oy - Info.res * 0.5f;
		Info.oz = GridBuilderCoarse.Info.oz - Info.res * 0.5f;
		Info.dtPhys = GridBuilderCoarse.Info.dtPhys * 0.5f;
		Info.nu = (Info.dtPhys * nuPhys) / ((Info.res/1000.f) * (Info.res/1000.f));
	}
	
	if ( !iAmFinest ) initializeGridInfo( gridBuilders, Bounds, level+1 );
}

void intArrayFromBoolArray( IntArrayType &intArray, const BoolArrayType &boolArray )
{
	auto intView = intArray.getView();
	auto boolView = boolArray.getConstView();
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		if ( boolView[ cell ] ) intView[ cell ] = 1;
		else intView[ cell ] = 0;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, intArray.getSize(), cellLambda );
}

void buildFinerGridBuilder( SkeletonGridStruct &SkeletonGrid, GridBuilderStruct &GridBuilderFine )
{
	// label stuff for SkeletonGrid
	const int cellCountSkeleton = SkeletonGrid.Info.cellCount;
	const int cellCountXSkeleton = SkeletonGrid.Info.cellCountX;
	const int cellCountYSkeleton = SkeletonGrid.Info.cellCountY;
	// const int cellCountZSkeleton = SkeletonGrid.Info.cellCountZ; // not needed here
	const int cellCountXYSkeleton = cellCountXSkeleton * cellCountYSkeleton;
	BoolArrayType &refinementMarkerArraySkeleton = SkeletonGrid.keepCellMarkerArray;
	// Some SkeletonGrid Views
	auto refinementMarkerViewSkeleton = refinementMarkerArraySkeleton.getConstView();
	
	// label stuff for GridBuilderFine
	const int cellCountFine = GridBuilderFine.Info.cellCount;
	IntArrayType &iArrayFine = GridBuilderFine.IJK.iArray;
	IntArrayType &jArrayFine = GridBuilderFine.IJK.jArray;
	IntArrayType &kArrayFine = GridBuilderFine.IJK.kArray;
	// Some GridBuilderFine Views
	auto iViewFine = iArrayFine.getView();
	auto jViewFine = jArrayFine.getView();
	auto kViewFine = kArrayFine.getView();
	
	// 1) build refinedParentList
	IntArrayType scanArray( cellCountSkeleton );
	IntArrayType refinedParentList( cellCountFine / 8 );
	intArrayFromBoolArray( scanArray, refinementMarkerArraySkeleton );
	TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, cellCountSkeleton, TNL::Plus{} );
	auto scanView = scanArray.getConstView();
	auto refinedParentListView = refinedParentList.getView();
	auto refinedParentListLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( refinementMarkerViewSkeleton[ cell ] )
		{
			const int index = scanView[ cell ];
			refinedParentListView[ index ] = cell;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountSkeleton, refinedParentListLambda );
	
	// 2) loop through the refined parents and write fine IJK
	auto refinedCellLambda = [=] __cuda_callable__ ( const int refinedIndex ) mutable
	{	
		const int cellSkeleton = refinedParentListView[ refinedIndex ];
		const int kSkeleton = cellSkeleton / cellCountXYSkeleton;
		const int remainder = cellSkeleton % cellCountXYSkeleton;
		const int jSkeleton = remainder / cellCountXSkeleton;
		const int iSkeleton = remainder % cellCountXSkeleton;
		int cellFine = 8 * refinedIndex;
		for ( int kAdd = 0; kAdd < 2; kAdd++ )
		{
			for ( int jAdd = 0; jAdd < 2; jAdd++ )
			{
				for ( int iAdd = 0; iAdd < 2; iAdd++ )
				{
					const int iFine = 2 * iSkeleton + iAdd;
					const int jFine = 2 * jSkeleton + jAdd;
					const int kFine = 2 * kSkeleton + kAdd;
					iViewFine[ cellFine ] = iFine;
					jViewFine[ cellFine ] = jFine;
					kViewFine[ cellFine ] = kFine;
					cellFine++;
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountFine / 8, refinedCellLambda );
}

void buildFinerGridBuilder( GridBuilderStruct &GridBuilderCoarse, GridBuilderStruct &GridBuilderFine )
{
	// label stuff for GridBuilderCoarse
	const int cellCountCoarse = GridBuilderCoarse.Info.cellCount;
	const IntArrayType &iArrayCoarse = GridBuilderCoarse.IJK.iArray;
	const IntArrayType &jArrayCoarse = GridBuilderCoarse.IJK.jArray;
	const IntArrayType &kArrayCoarse = GridBuilderCoarse.IJK.kArray;
	BoolArrayType &refinementMarkerArrayCoarse = GridBuilderCoarse.refinementMarkerArray;
	// Some GridBuilderCoarse Views
	auto iViewCoarse = iArrayCoarse.getConstView();
	auto jViewCoarse = jArrayCoarse.getConstView();
	auto kViewCoarse = kArrayCoarse.getConstView();
	auto refinementMarkerViewCoarse = refinementMarkerArrayCoarse.getConstView();
	
	// label stuff for GridBuilderFine
	const int cellCountFine = GridBuilderFine.Info.cellCount;
	IntArrayType &iArrayFine = GridBuilderFine.IJK.iArray;
	IntArrayType &jArrayFine = GridBuilderFine.IJK.jArray;
	IntArrayType &kArrayFine = GridBuilderFine.IJK.kArray;
	// Some GridBuilderFine Views
	auto iViewFine = iArrayFine.getView();
	auto jViewFine = jArrayFine.getView();
	auto kViewFine = kArrayFine.getView();
	
	// 1) build refinedParentList
	IntArrayType scanArray( cellCountCoarse );
	IntArrayType refinedParentList( cellCountFine / 8 );
	intArrayFromBoolArray( scanArray, refinementMarkerArrayCoarse );
	TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, cellCountCoarse, TNL::Plus{} );
	auto scanView = scanArray.getConstView();
	auto refinedParentListView = refinedParentList.getView();
	auto refinedParentListLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( refinementMarkerViewCoarse[ cell ] )
		{
			const int index = scanView[ cell ];
			refinedParentListView[ index ] = cell;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountCoarse, refinedParentListLambda );
	
	// 2) loop through the refined parents and write fine IJK
	auto refinedCellLambda = [=] __cuda_callable__ ( const int refinedIndex ) mutable
	{	
		const int cellCoarse = refinedParentListView[ refinedIndex ];
		const int iCoarse = iViewCoarse[ cellCoarse ];
		const int jCoarse = jViewCoarse[ cellCoarse ];
		const int kCoarse = kViewCoarse[ cellCoarse ];
		int cellFine = 8 * refinedIndex;
		for ( int kAdd = 0; kAdd < 2; kAdd++ )
		{
			for ( int jAdd = 0; jAdd < 2; jAdd++ )
			{
				for ( int iAdd = 0; iAdd < 2; iAdd++ )
				{
					const int iFine = 2 * iCoarse + iAdd;
					const int jFine = 2 * jCoarse + jAdd;
					const int kFine = 2 * kCoarse + kAdd;
					iViewFine[ cellFine ] = iFine;
					jViewFine[ cellFine ] = jFine;
					kViewFine[ cellFine ] = kFine;
					cellFine++;
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountFine / 8, refinedCellLambda );
}

// sort IJK arrays as ascending k, j, i so that k changes the slowest
void sortIJK( IJKArrayStruct &IJK )
{
	const int cellCount = IJK.iArray.getSize();
	auto iView = IJK.iArray.getView();
	auto jView = IJK.jArray.getView();
	auto kView = IJK.kArray.getView();

    auto comparisonLambda = [=] __cuda_callable__ ( const size_t a, const size_t b )
	{
		if ( kView[ a ] < kView[ b ] ) return true;
		else if ( kView[ a ] > kView[ b ] ) return false;
		else
		{
			if ( jView[ a ] < jView[ b ] ) return true;
			else if ( jView[ a ] > jView[ b ] ) return false;
			else
			{
				if ( iView[ a ] < iView[ b ] ) return true;
				else if ( iView[ a ] > iView[ b ] ) return false;
				else return false;
			}
		}
	};
	auto swapLambda = [=] __cuda_callable__ ( const size_t a, const size_t b ) mutable
	{
		TNL::swap( iView[ a ], iView[ b ] );
		TNL::swap( jView[ a ], jView[ b ] );
		TNL::swap( kView[ a ], kView[ b ] );
	};
	TNL::Algorithms::sort<TNL::Devices::Cuda, size_t>( 0, cellCount, comparisonLambda, swapLambda );
}

// For each Wanted, find a matching cell in Source. If its found, write its index to resultArray. If there is no such cell, write -1.
// Source IJK must be already sorted		
void binarySearchIJK( IJKArrayStruct &Wanted, IJKArrayStruct &Source, IntArrayType &resultArray )
{
	const int wantedCellCount = Wanted.iArray.getSize();
	const int sourceCellCount = Source.iArray.getSize();
	
	auto iWantedView = Wanted.iArray.getConstView();
	auto jWantedView = Wanted.jArray.getConstView();
	auto kWantedView = Wanted.kArray.getConstView();
	auto iSourceView = Source.iArray.getConstView();
	auto jSourceView = Source.jArray.getConstView();
	auto kSourceView = Source.kArray.getConstView();
	auto resultView = resultArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cellWanted ) mutable
	{
		const int iWanted = iWantedView[ cellWanted ];
		const int jWanted = jWantedView[ cellWanted ];
		const int kWanted = kWantedView[ cellWanted ];
		
		int start = 0;
		int end = sourceCellCount;
		int result = -1;
		
		// Search for k, j, and i in a single binary search
		while ( end > start )
		{
			int half = start + ( end - start ) / 2;
			int kHalf = kSourceView[ half ];
			int jHalf = jSourceView[ half ];
			int iHalf = iSourceView[ half ];
			
			if ( kHalf == kWanted && jHalf == jWanted && iHalf == iWanted )
			{
				result = half;
				break;
			}
			
			// Move the end bound if k is too big, OR if k matches but j is too big, OR if k and j match but i is too big
			if ( kHalf > kWanted || 
			   ( kHalf == kWanted && jHalf > jWanted ) || 
			   ( kHalf == kWanted && jHalf == jWanted && iHalf > iWanted ) ) 
			{
				end = half;
			}
			else 
			{
				start = half + 1;
			}
		}
		if ( result == -1 && start < sourceCellCount )
		{
			if ( kSourceView[ start ] == kWanted && jSourceView[ start ] == jWanted && iSourceView[ start ] == iWanted )
			{
				result = start;
			}
		}
		resultView[ cellWanted ] = result;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, wantedCellCount, cellLambda );
}

void buildIJKFull( std::vector<GridBuilderStruct> &gridBuilders, const std::vector<VoxelizerStruct> &voxelizers, const int level )
// Builds uncompressed full IJK for all grid levels recursively. 
// "Full" IJK means that the area on coarse grid that has a finer grid on top does not get deleted (yet)
{
	if ( level == 0 ) std::cout << "Building full IJK for all grid levels" << std::endl; 
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridBuilderStruct &GridBuilder = gridBuilders[ level ];	
	InfoStruct &Info = GridBuilder.Info;	
	
	SkeletonGridStruct &SkeletonGrid = GridBuilder.SkeletonGrid;
	InfoStruct &SkeletonInfo = SkeletonGrid.Info;
	
	static GridBuilderStruct dummyGridBuilder; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridBuilderStruct &GridBuilderCoarse = iAmCoarsest ? dummyGridBuilder : gridBuilders[ level - 1 ];
	
	// 1) On coarser grid, mark refinement area where our level will be built, from that calculate our cellCount
	if ( iAmCoarsest )
	{
		SkeletonGrid.keepCellMarkerArray.setSize( SkeletonInfo.cellCount );
		markKeepCells( SkeletonGrid, voxelizers );
		Info.cellCount = 8 * TNL::sum( SkeletonGrid.keepCellMarkerArray );
	}
	else
	{
		markRefinementCells( GridBuilderCoarse, voxelizers );
		Info.cellCount = 8 * TNL::sum( GridBuilderCoarse.refinementMarkerArray );
	}
	
	// 2) Set size of our arrays to cellCount
	GridBuilder.IJK.iArray.setSize( Info.cellCount );
	GridBuilder.IJK.jArray.setSize( Info.cellCount );
	GridBuilder.IJK.kArray.setSize( Info.cellCount );
	GridBuilder.NBR.jPlusArray.setSize( Info.cellCount );
	GridBuilder.NBR.kPlusArray.setSize( Info.cellCount );
	GridBuilder.NBR.isGeometricBitPackedMarkerArray.setSize( Info.cellCount );
	GridBuilder.keepCellMarkerArray.setSize( Info.cellCount );	
	if ( !iAmFinest )
	{
		GridBuilder.refinementMarkerArray.setSize( Info.cellCount );
		GridBuilder.deepRefinementMarkerArray.setSize( Info.cellCount );
		GridBuilder.fineToCoarseMarkerArray.setSize( Info.cellCount );
		GridBuilder.coarseToFineMarkerArray.setSize( Info.cellCount );
	}
	if ( !iAmCoarsest )
	{
		GridBuilder.parentMapArray.setSize( Info.cellCount );
		GridBuilder.parentInterfaceMarkerArray.setSize( Info.cellCount );
	}
	
	// 3) Build our grid = fill our IJK (we are the "finer grid" with respect to the grid we are taking spatial information from)
	if ( iAmCoarsest ) buildFinerGridBuilder( SkeletonGrid, GridBuilder );
	else buildFinerGridBuilder( GridBuilderCoarse, GridBuilder );
	
	// 4) Sort our IJK so that k changes the slowest
	sortIJK( GridBuilder.IJK );
	
	// 5) Build our NBR Plus and mark geometric validity
	buildNBRPlus( GridBuilder );
	markGeometricNBRPlus( GridBuilder );
	
	// 6) Build parentMapArray
	if ( !iAmCoarsest )
	{
		IJKArrayStruct IJKWanted;
		IJKWanted.iArray = GridBuilder.IJK.iArray / 2;
		IJKWanted.jArray = GridBuilder.IJK.jArray / 2;
		IJKWanted.kArray = GridBuilder.IJK.kArray / 2;
		binarySearchIJK( IJKWanted, GridBuilderCoarse.IJK, GridBuilder.parentMapArray );
	}
	
	// 7) Mark our cells that are part of the parent interface. A fine cell located at the parent interface:
	//    - is blocked from getting deleted later
	//	  - is blocked from getting deeply refined (interface with finer grid is still allowed)
	//	  - inherits fluid / wall state from the parent, even if the voxelizer says otherwise
	if ( !iAmCoarsest )
	{
		auto parentInterfaceMarkerView = GridBuilder.parentInterfaceMarkerArray.getView();
		auto parentMapView = GridBuilder.parentMapArray.getConstView();
		auto parentCoarseToFineMarkerView = GridBuilderCoarse.coarseToFineMarkerArray.getConstView();
		auto parentFineToCoarseMarkerView = GridBuilderCoarse.fineToCoarseMarkerArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			const int parentCell = parentMapView( cell );
			const bool parentInterfaceMarker = (parentCoarseToFineMarkerView( parentCell ) + parentFineToCoarseMarkerView( parentCell ));
			parentInterfaceMarkerView( cell ) = parentInterfaceMarker;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
	}
	
	// std::cout << "	Level " << level << " done, initial cellCount " << Info.cellCount << std::endl;
	
	// 8) Recursion
	if ( !iAmFinest ) buildIJKFull( gridBuilders, voxelizers, level + 1 );
	// else std::cout << std::endl;
}

void deleteExcessCells( std::vector<GridBuilderStruct> &gridBuilders, const std::vector<VoxelizerStruct> &voxelizers, const int level )
// Delete cells from each level that are deep inside a wall or deeply refined
// Enforce keep cells that are part of the parent interface
{
	if ( level == 0 ) std::cout << "Deleting excess cells for all grid levels" << std::endl; 
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridBuilderStruct &GridBuilder = gridBuilders[ level ];	
	InfoStruct &Info = GridBuilder.Info;	
	const VoxelizerStruct &Voxelizer = voxelizers[ level ];
	
	static GridBuilderStruct dummyGridBuilder; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridBuilderStruct &GridBuilderCoarse = iAmCoarsest ? dummyGridBuilder : gridBuilders[ level - 1 ];
	
	// 1) Mark fluid cells and add one layer of walls
	markWallCells( GridBuilder.keepCellMarkerArray, Voxelizer.rayMapTotal, GridBuilder );
	GridBuilder.keepCellMarkerArray = !GridBuilder.keepCellMarkerArray; // now keepCellMarkerArray marks fluid cells only
	BoolArrayType markerSource;
	markerSource = GridBuilder.keepCellMarkerArray;
	spreadMarkers( GridBuilder.keepCellMarkerArray, markerSource, GridBuilder ); // added one layer of walls
	
	// 2) Remove deeply refined cells
	if ( !iAmFinest ) GridBuilder.keepCellMarkerArray = GridBuilder.keepCellMarkerArray * !GridBuilder.deepRefinementMarkerArray; 
	
	// 3) Because we changed the coarser grid, we must rebuild the parentMapArray and parentInterfaceMarkerArray
	if ( !iAmCoarsest )
	{
		IJKArrayStruct IJKWanted;
		IJKWanted.iArray = GridBuilder.IJK.iArray / 2;
		IJKWanted.jArray = GridBuilder.IJK.jArray / 2;
		IJKWanted.kArray = GridBuilder.IJK.kArray / 2;
		binarySearchIJK( IJKWanted, GridBuilderCoarse.IJK, GridBuilder.parentMapArray );
	}
	if ( !iAmCoarsest )
	{
		auto parentInterfaceMarkerView = GridBuilder.parentInterfaceMarkerArray.getView();
		auto parentMapView = GridBuilder.parentMapArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			const int parentCell = parentMapView( cell );
			if ( parentCell < 0 ) parentInterfaceMarkerView( cell ) = false; // parent cell does not exist here so there is certainly no interface
			else parentInterfaceMarkerView( cell ) = true; // because we deleted deep refinement cells from the parent, this must only be the interface
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
	}
	
	// 4) Enforce keep cells that are part of the parent interface. 
	if ( !iAmCoarsest ) GridBuilder.keepCellMarkerArray += GridBuilder.parentInterfaceMarkerArray; 
	
	// 5) Build fullToKeep map
	IntArrayType fullToKeepMapArray( Info.cellCount );
	intArrayFromBoolArray( fullToKeepMapArray, GridBuilder.keepCellMarkerArray );
	TNL::Algorithms::inplaceExclusiveScan( fullToKeepMapArray, 0, Info.cellCount, TNL::Plus{} );
	
	// 6) Transform necessary information from full grid to the keep grid
	// We need IJK, parentMapArray, fineToCoarseInterfaceMarkerArray, coarseToFineInterfaceMarkerArray	
	// starting a scope so that temporary arrays then go out of scope
	{ 	
		IJKArrayStruct IJKFull = GridBuilder.IJK;
		IntArrayType parentMapArrayFull;
		parentMapArrayFull = GridBuilder.parentMapArray;
		BoolArrayType fineToCoarseMarkerArrayFull;
		fineToCoarseMarkerArrayFull = GridBuilder.fineToCoarseMarkerArray;
		BoolArrayType coarseToFineMarkerArrayFull;
		coarseToFineMarkerArrayFull = GridBuilder.coarseToFineMarkerArray;
		
		auto keepCellMarkerView = GridBuilder.keepCellMarkerArray.getConstView();
		auto fullToKeepMapView = fullToKeepMapArray.getConstView();
		
		auto iFullView = IJKFull.iArray.getConstView();
		auto jFullView = IJKFull.jArray.getConstView();
		auto kFullView = IJKFull.kArray.getConstView();
		auto parentMapFullView = parentMapArrayFull.getConstView();
		auto fineToCoarseMarkerFullView = fineToCoarseMarkerArrayFull.getConstView();
		auto coarseToFineMarkerFullView = coarseToFineMarkerArrayFull.getConstView();
		auto iView = GridBuilder.IJK.iArray.getView();
		auto jView = GridBuilder.IJK.jArray.getView();
		auto kView = GridBuilder.IJK.kArray.getView();
		auto parentMapView = GridBuilder.parentMapArray.getView();
		auto fineToCoarseMarkerView = GridBuilder.fineToCoarseMarkerArray.getView();
		auto coarseToFineMarkerView = GridBuilder.coarseToFineMarkerArray.getView();
		
		auto fullToKeepLambda = [=] __cuda_callable__ ( const int cellFull ) mutable
		{
			if ( !keepCellMarkerView( cellFull ) ) return;
			const int cell = fullToKeepMapView( cellFull );
			iView( cell ) = iFullView( cellFull );
			jView( cell ) = jFullView( cellFull );
			kView( cell ) = kFullView( cellFull );
			if (!iAmCoarsest) parentMapView( cell ) = parentMapFullView( cellFull );
			if (!iAmFinest) fineToCoarseMarkerView( cell ) = fineToCoarseMarkerFullView( cellFull );
			if (!iAmFinest) coarseToFineMarkerView( cell ) = coarseToFineMarkerFullView( cellFull );
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, fullToKeepLambda );	
	}
	
	// 7) set new cellCount
	Info.cellCount = TNL::sum( GridBuilder.keepCellMarkerArray );
	
	// 8) Resize the necessary arrays. Note that NBR is now broken and will have to be rebuilt again
	GridBuilder.IJK.iArray.resize( Info.cellCount );
	GridBuilder.IJK.jArray.resize( Info.cellCount );
	GridBuilder.IJK.kArray.resize( Info.cellCount );
	GridBuilder.NBR.jPlusArray.resize( Info.cellCount );
	GridBuilder.NBR.kPlusArray.resize( Info.cellCount );
	GridBuilder.NBR.jMinusArray.resize( Info.cellCount );
	GridBuilder.NBR.kMinusArray.resize( Info.cellCount );
	GridBuilder.NBR.isGeometricBitPackedMarkerArray.resize( Info.cellCount );
	GridBuilder.wallMarkerArray.resize( Info.cellCount );
	if (!iAmCoarsest) GridBuilder.parentMapArray.resize( Info.cellCount );
	if (!iAmFinest) GridBuilder.fineToCoarseMarkerArray.resize( Info.cellCount );
	if (!iAmFinest) GridBuilder.coarseToFineMarkerArray.resize( Info.cellCount );
	
	// 9) Forget the no longer necessary arrays
	GridBuilder.SkeletonGrid.keepCellMarkerArray.resize( 0 );
	GridBuilder.deepRefinementMarkerArray.resize( 0 );
	GridBuilder.refinementMarkerArray.resize( 0 );
	
	// 10) Rebuild our NBR Plus
	buildNBRPlus( GridBuilder );
	markGeometricNBRPlus( GridBuilder );
	
	// 11) Build our NBR Minus
	if ( !iAmCoarsest )
	{
		auto jPlusView = GridBuilder.NBR.jPlusArray.getConstView();
		auto kPlusView = GridBuilder.NBR.kPlusArray.getConstView();
		auto jMinusView = GridBuilder.NBR.jMinusArray.getView();
		auto kMinusView = GridBuilder.NBR.kMinusArray.getView();
		auto NBRMinusLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			jMinusView[ jPlusView[ cell ] ] = cell;
			kMinusView[ kPlusView[ cell ] ] = cell;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, NBRMinusLambda );
	}
	
	// 12) fill grid bounds
	Info.Bounds.xMin = Info.ox + Info.res * TNL::min( GridBuilder.IJK.iArray ) - 0.5f * Info.res;
	Info.Bounds.xMax = Info.ox + Info.res * TNL::max( GridBuilder.IJK.iArray ) + 0.5f * Info.res;
	Info.Bounds.yMin = Info.oy + Info.res * TNL::min( GridBuilder.IJK.jArray ) - 0.5f * Info.res;
	Info.Bounds.yMax = Info.oy + Info.res * TNL::max( GridBuilder.IJK.jArray ) + 0.5f * Info.res;
	Info.Bounds.zMin = Info.oz + Info.res * TNL::min( GridBuilder.IJK.kArray ) - 0.5f * Info.res;
	Info.Bounds.zMax = Info.oz + Info.res * TNL::max( GridBuilder.IJK.kArray ) + 0.5f * Info.res;
	
	std::cout << "	Level " << level << " done, final cellCount " << Info.cellCount << std::endl;
	
	// 13) Recursion
	if ( !iAmFinest ) deleteExcessCells( gridBuilders, voxelizers, level + 1 );
	// else std::cout << std::endl;
}

void buildWallMarkers( std::vector<GridBuilderStruct> &gridBuilders, const std::vector<VoxelizerStruct> &voxelizers, std::vector<STLStruct> &gridStaticSTLs, const int level )
// Delete cells from each level that are deep inside a wall or deeply refined
// Enforce keep cells that are part of the parent interface
{
	if ( level == 0 ) std::cout << "Marking walls for all grid levels" << std::endl; 
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridBuilderStruct &GridBuilder = gridBuilders[ level ];	
	InfoStruct &Info = GridBuilder.Info;	
	const VoxelizerStruct &Voxelizer = voxelizers[ level ];
	
	static GridBuilderStruct dummyGridBuilder; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridBuilderStruct &GridBuilderCoarse = iAmCoarsest ? dummyGridBuilder : gridBuilders[ level - 1 ];
	
	// 1) Final marking of the wall
	markWallCells( GridBuilder.wallMarkerArray, Voxelizer.rayMapTotal, GridBuilder );
	if ( !iAmCoarsest ) // if we are not coarsest, inherit wall state at parent interface from the parent
	{
		auto parentMapView = GridBuilder.parentMapArray.getConstView();
		auto parentWallMarkerView = GridBuilderCoarse.wallMarkerArray.getConstView();
		auto wallMarkerView = GridBuilder.wallMarkerArray.getView();
		auto parentWallLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			const int parentCell = parentMapView( cell );
			if ( parentCell < 0 ) return; 
			wallMarkerView( cell ) = parentWallMarkerView( parentCell );
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, parentWallLambda );
	}
	
	// 2) For all fluid cells adjacent to a wall, mark which voxelized body they belong to
	// The last body has higher priority
	// At the parent interface, inherit bodyID from the parent, this has the highest priority
	GridBuilder.wallIDArray.setSize( Info.cellCount );
	GridBuilder.wallIDArray.setValue( -1 );
	
	BoolArrayType markerSourceArray( Info.cellCount );
	BoolArrayType bodyWallMarkerArray( Info.cellCount );
	
	for ( int wallID = 0; wallID < (int)Voxelizer.rayMaps.size(); wallID++ )
	{
		const RayMapStruct &RayMap = Voxelizer.rayMaps[ wallID ];
		markWallCells( bodyWallMarkerArray, RayMap, GridBuilder );
		// At the interface, we must overwrite marker for this wall by the parent cells
		// Generate temporary parent wall marker
		BoolArrayType parentBodyWallMarkerArray( GridBuilderCoarse.Info.cellCount );
		if (!iAmCoarsest) 
		{
			markWallCells( parentBodyWallMarkerArray, voxelizers[level-1].rayMaps[ wallID ], GridBuilderCoarse );
			auto bodyWallMarkerView = bodyWallMarkerArray.getView();
			auto parentMapView = GridBuilder.parentMapArray.getConstView();
			auto parentBodyWallMarkerView = parentBodyWallMarkerArray.getConstView();
			auto wallIDLambda = [=] __cuda_callable__ ( const int cell ) mutable
			{	
				const int parentCell = parentMapView( cell );
				if ( parentCell >= 0 ) bodyWallMarkerView( cell ) = parentBodyWallMarkerView( parentCell );
			};
			TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, wallIDLambda );
		}
		markerSourceArray = bodyWallMarkerArray;
		spreadMarkers( bodyWallMarkerArray, markerSourceArray, GridBuilder );
		bodyWallMarkerArray = bodyWallMarkerArray * !GridBuilder.wallMarkerArray;
		auto bodyWallMarkerView = bodyWallMarkerArray.getConstView();
		auto wallIDView = GridBuilder.wallIDArray.getView();
		auto wallIDLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( bodyWallMarkerView( cell ) ) wallIDView( cell ) = wallID;	
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, wallIDLambda );
	}
	
	// 3) Build scan over wall adjacent cells
	IntArrayType wallAdjacentScanArray( Info.cellCount );
	wallAdjacentScanArray.setValue( 0 );
	auto wallIDView = GridBuilder.wallIDArray.getConstView();
	auto wallAdjacentScanView = wallAdjacentScanArray.getView();
	auto wallAdjacentLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( wallIDView( cell ) >= 0 ) wallAdjacentScanView( cell ) = 1;	
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, wallAdjacentLambda );
	const int wallAdjacentCellCount = TNL::sum( wallAdjacentScanArray );
	TNL::Algorithms::inplaceExclusiveScan( wallAdjacentScanArray, 0, Info.cellCount, TNL::Plus{} );
	
	// 4) Build compact wall adjacent list and compact the wallID array
	GridBuilder.wallAdjacentCellList.setSize( wallAdjacentCellCount );
	IntArrayType wallIDArrayCopy;
	wallIDArrayCopy = GridBuilder.wallIDArray;
	GridBuilder.wallIDArray.resize( wallAdjacentCellCount );
	auto wallAdjacentCellListView = GridBuilder.wallAdjacentCellList.getView();
	auto wallIDCopyView = wallIDArrayCopy.getConstView();
	auto wallIDCompactView = GridBuilder.wallIDArray.getView();
	auto compactionLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( wallIDCopyView( cell ) >= 0 ) 
		{
			const int compactIndex = wallAdjacentScanView( cell );
			wallAdjacentCellListView( compactIndex ) = cell;	
			wallIDCompactView( compactIndex ) = wallIDCopyView( cell );
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, compactionLambda );
	
	// 5) Fill linkExistenceMarkerArray
	GridBuilder.linkExistenceMarkerArray.setSizes( 27, wallAdjacentCellCount );
	buildLinkExistenceMarkerArray( GridBuilder );
	
	// 6) Fill link lengths
	GridBuilder.linkLengthArray.setSizes( 27, wallAdjacentCellCount );
	buildLinkLengthArray( GridBuilder, gridStaticSTLs );
	
	// std::cout << "	Level " << level << " done" << std::endl;
	// 3) Recursion
	if ( !iAmFinest ) buildWallMarkers( gridBuilders, voxelizers, gridStaticSTLs, level + 1 );
	// else std::cout << std::endl;
}	

void buildGrids( std::vector<GridStruct> &grids, std::vector<STLStruct> &gridStaticSTLs, BoundsStruct DomainBounds )
{
	// call the temporary GridBuilders and Voxelizers - these will go out of scope
	// start the GridBuilder and Voxelizer scope
	{
		std::vector<GridBuilderStruct> gridBuilders( GRID_LEVEL_COUNT );
		initializeGridInfo( gridBuilders, DomainBounds, 0 );
		
		// Voxelizers
		std::vector<VoxelizerStruct> voxelizers( GRID_LEVEL_COUNT );
		initializeVoxelizers( voxelizers, gridBuilders, gridStaticSTLs, 0 );
		
		// Build grids
		buildIJKFull( gridBuilders, voxelizers, 0 );
		deleteExcessCells( gridBuilders, voxelizers, 0 );
		buildWallMarkers( gridBuilders, voxelizers, gridStaticSTLs, 0 );
		
		// Pass the information to the actual grids, but in a compressed form
	}
}
