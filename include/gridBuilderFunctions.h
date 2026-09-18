#pragma once

#include "./types.h"
#include "./markerFunctions.h"
#include "./boundaryConditions/interpolatedBouncebackFunctions.h"
#include "./boundaryConditions/applyInitialCondition.h"
#include "./rotorFunctions.h"

void initializeGridInfo( std::vector<GridBuilderStruct> &gridBuilders, const BoundsStruct &Bounds, const int level )
{
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	if ( iAmCoarsest ) 
	{
		std::cout << "Initializing Info for all grid levels" << std::endl;
		std::cout << std::endl;
	}
	
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
		
		Info.dtPhys = DT_PHYS_GLOBAL;
		Info.nu = (Info.dtPhys * NU_PHYS) / ((Info.res/1000.f) * (Info.res/1000.f));
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
		Info.nu = (Info.dtPhys * NU_PHYS) / ((Info.res/1000.f) * (Info.res/1000.f));
	}
	
	if ( !iAmFinest ) initializeGridInfo( gridBuilders, Bounds, level+1 );
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
void binarySearchIJK( const IJKArrayStruct &Wanted, const IJKArrayStruct &Source, IntArrayType &resultArray )
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
		GridBuilder.parentFineToCoarseMarkerArray.setSize( Info.cellCount );
		GridBuilder.needValuesFromCoarseMarkerArray.setSize( Info.cellCount );
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
	//    - is blocked from getting deleted later (unless in the outermost layer)
	//	  - is blocked from getting deeply refined (interface with finer grid is still allowed)
	//	  - inherits fluid / wall state from the parent, even if the voxelizer says otherwise
	// Also, in another array mark cells that are specifically under the fineToCoarse interface of our parent
	if ( !iAmCoarsest )
	{
		auto parentInterfaceMarkerView = GridBuilder.parentInterfaceMarkerArray.getView();
		auto parentFineToCoarseMarkerView = GridBuilder.parentFineToCoarseMarkerArray.getView();
		auto parentMapView = GridBuilder.parentMapArray.getConstView();
		auto coarseToFineMarkerViewCoarse = GridBuilderCoarse.coarseToFineMarkerArray.getConstView();
		auto fineToCoarseMarkerViewCoarse = GridBuilderCoarse.fineToCoarseMarkerArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			const int parentCell = parentMapView( cell );
			const bool parentInterfaceMarker = (coarseToFineMarkerViewCoarse( parentCell ) + fineToCoarseMarkerViewCoarse( parentCell ));
			parentInterfaceMarkerView( cell ) = parentInterfaceMarker;
			const bool parentFineToCoarseMarker = fineToCoarseMarkerViewCoarse( parentCell );
			parentFineToCoarseMarkerView( cell ) = parentFineToCoarseMarker;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
	}
	
	// 8) Recursion
	if ( !iAmFinest ) buildIJKFull( gridBuilders, voxelizers, level + 1 );
	else std::cout << std::endl;
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
	
	// 5) Delete the outermost layer of interface fine cells
	// This is because Geier's interpolation cube sends information from 8 coarse cells
	// to 8 fine cells in their center. The outermost layer is excessive.
	// We identify this by spreading from the parentFineToCoarse array
	// At the same time, mark the needValuesFromCoarseMarkerArray
	if ( !iAmCoarsest )
	{
		markerSource = GridBuilder.parentFineToCoarseMarkerArray;
		BoolArrayType coarseToFine3LayerArray( Info.cellCount );
		spreadMarkers( coarseToFine3LayerArray, markerSource, GridBuilder );
		coarseToFine3LayerArray.swap( markerSource );
		spreadMarkers( coarseToFine3LayerArray, markerSource, GridBuilder );
		coarseToFine3LayerArray.swap( markerSource );
		spreadMarkers( coarseToFine3LayerArray, markerSource, GridBuilder );
		// now the result is in the coarseToFine3LayerArray
		// delete cells that are part of the parent interface but not in this 3Layer
		BoolArrayType markToDelete( Info.cellCount );
		markToDelete = GridBuilder.parentInterfaceMarkerArray * !coarseToFine3LayerArray;
		GridBuilder.keepCellMarkerArray *= !markToDelete;
		// now mark the needValuesFromCoarseMarkerArray
		GridBuilder.needValuesFromCoarseMarkerArray = GridBuilder.parentInterfaceMarkerArray * GridBuilder.keepCellMarkerArray;
		// use markerSource as temporary target
		spreadMarkers( markerSource, GridBuilder.parentFineToCoarseMarkerArray, GridBuilder );
		GridBuilder.needValuesFromCoarseMarkerArray *= !markerSource;
	}
	
	// 6) Build fullToKeep map
	IntArrayType fullToKeepMapArray( Info.cellCount );
	intArrayFromBoolArray( fullToKeepMapArray, GridBuilder.keepCellMarkerArray );
	TNL::Algorithms::inplaceExclusiveScan( fullToKeepMapArray, 0, Info.cellCount, TNL::Plus{} );
	
	// 7) Transform necessary information from full grid to the keep grid
	// We need IJK, parentMapArray, needValuesFromCoarseMarkerArray,
	// fineToCoarseInterfaceMarkerArray, coarseToFineInterfaceMarkerArray	
	// starting a scope so that temporary arrays then go out of scope
	{ 	
		IJKArrayStruct IJKFull = GridBuilder.IJK;
		IntArrayType parentMapArrayFull;
		parentMapArrayFull = GridBuilder.parentMapArray;
		BoolArrayType needValuesFromCoarseMarkerArrayFull;
		needValuesFromCoarseMarkerArrayFull = GridBuilder.needValuesFromCoarseMarkerArray;
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
		auto needValuesFromCoarseMarkerFullView = needValuesFromCoarseMarkerArrayFull.getConstView();
		auto fineToCoarseMarkerFullView = fineToCoarseMarkerArrayFull.getConstView();
		auto coarseToFineMarkerFullView = coarseToFineMarkerArrayFull.getConstView();
		auto iView = GridBuilder.IJK.iArray.getView();
		auto jView = GridBuilder.IJK.jArray.getView();
		auto kView = GridBuilder.IJK.kArray.getView();
		auto parentMapView = GridBuilder.parentMapArray.getView();
		auto needValuesFromCoarseMarkerView = GridBuilder.needValuesFromCoarseMarkerArray.getView();
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
			if (!iAmCoarsest) needValuesFromCoarseMarkerView( cell ) = needValuesFromCoarseMarkerFullView( cellFull );
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
	GridBuilder.interfaceOverlapMarkerArray.setSize( Info.cellCount );
	GridBuilder.interfaceOverlapMarkerArray.setValue( false );
	if (!iAmCoarsest) GridBuilder.parentMapArray.resize( Info.cellCount );
	if (!iAmCoarsest) GridBuilder.needValuesFromCoarseMarkerArray.resize( Info.cellCount );
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
	
	// 12) fill grid bounds
	Info.Bounds.xMin = Info.ox + Info.res * TNL::min( GridBuilder.IJK.iArray ) - 0.5f * Info.res;
	Info.Bounds.xMax = Info.ox + Info.res * TNL::max( GridBuilder.IJK.iArray ) + 0.5f * Info.res;
	Info.Bounds.yMin = Info.oy + Info.res * TNL::min( GridBuilder.IJK.jArray ) - 0.5f * Info.res;
	Info.Bounds.yMax = Info.oy + Info.res * TNL::max( GridBuilder.IJK.jArray ) + 0.5f * Info.res;
	Info.Bounds.zMin = Info.oz + Info.res * TNL::min( GridBuilder.IJK.kArray ) - 0.5f * Info.res;
	Info.Bounds.zMax = Info.oz + Info.res * TNL::max( GridBuilder.IJK.kArray ) + 0.5f * Info.res;
	
	// 13) Recursion
	if ( !iAmFinest ) deleteExcessCells( gridBuilders, voxelizers, level + 1 );
	else std::cout << std::endl;
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
	
	// 5) Fill linkExistenceMarkerArray and linkPiercesInterfaceMarkerArray
	GridBuilder.linkExistenceMarkerArray.setSizes( 27, wallAdjacentCellCount );
	GridBuilder.linkPiercesInterfaceMarkerArray.setSizes( 27, wallAdjacentCellCount );
	buildLinkExistenceMarkerArray( GridBuilder );
	
	// 6) Fill link lengths
	GridBuilder.linkLengthArray.setSizes( 27, wallAdjacentCellCount );
	buildLinkLengthArray( GridBuilder, gridStaticSTLs );
	
	// 7) Fill interfaceOverlapMarker
	// mark our fineToCoarse interface if applicable
	if ( !iAmFinest ) GridBuilder.interfaceOverlapMarkerArray += GridBuilder.fineToCoarseMarkerArray;
	// mark the parent interface but only the coarseToFine part
	if ( !iAmCoarsest )
	{
		auto interfaceOverlapMarkerView = GridBuilder.interfaceOverlapMarkerArray.getView();
		auto parentMapView = GridBuilder.parentMapArray.getConstView();
		auto coarseCoarseToFineMarkerView = GridBuilderCoarse.coarseToFineMarkerArray.getConstView();
		auto overlapLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			const int parentCell = parentMapView( cell );
			if ( parentCell >= 0 ) 
			{
				const bool coarseToFineParentMarker = coarseCoarseToFineMarkerView( parentCell );
				if ( coarseToFineParentMarker ) interfaceOverlapMarkerView( cell ) = true;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, overlapLambda );
	}

	// 3) Recursion
	if ( !iAmFinest ) buildWallMarkers( gridBuilders, voxelizers, gridStaticSTLs, level + 1 );
	else std::cout << std::endl;
}	

void fillFineToCoarseInterface( InterfaceStruct &Interface, const BoolArrayType &markerArray, 
					const IntArrayType &childMapArrayGlobal, const GridBuilderStruct &GridBuilder )
{
	const int cellCountTotal = markerArray.getSize();
	Interface.interfaceCount = TNL::sum( markerArray );
	Interface.indexArray.setSize( Interface.interfaceCount );
	Interface.childMapArray.setSize( Interface.interfaceCount );
	
	IntArrayType scanArray( markerArray.getSize() );
	intArrayFromBoolArray( scanArray, markerArray );
	TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, cellCountTotal, TNL::Plus{} );
	
	auto scanView = scanArray.getConstView();
	auto indexView = Interface.indexArray.getView();
	auto childMapView = Interface.childMapArray.getView();
	auto markerView = markerArray.getConstView();
	auto childMapGlobalView = childMapArrayGlobal.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( !markerView( cell ) ) return;
		const int child = childMapGlobalView( cell );
		const int index = scanView( cell );
		
		indexView( index ) = cell;
		childMapView( index ) = child;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountTotal, cellLambda );
}

void fillCoarseToFineInterface( InterfaceStruct &Interface, const BoolArrayType &markerArray, const IntArrayType &childMapArrayGlobal, 
								const GridBuilderStruct &GridBuilder, const GridBuilderStruct &GridBuilderFine )
{
	const InfoStruct &Info = GridBuilder.Info;
	const InfoStruct &InfoFine = GridBuilderFine.Info;
	const IJKArrayStruct &IJK = GridBuilder.IJK;
	const NBRArrayStruct &NBR = GridBuilder.NBR;
	const BoolArrayType &wallMarkerArray = GridBuilder.wallMarkerArray;
	const int cellCountTotal = markerArray.getSize();
	
	auto iView = IJK.iArray.getConstView();
	auto jView = IJK.jArray.getConstView();
	auto kView = IJK.kArray.getConstView();
	auto iViewFine = GridBuilderFine.IJK.iArray.getConstView();
	auto jViewFine = GridBuilderFine.IJK.jArray.getConstView();
	auto kViewFine = GridBuilderFine.IJK.kArray.getConstView();
	auto jPlusGlobalView = NBR.jPlusArray.getConstView();
	auto kPlusGlobalView = NBR.kPlusArray.getConstView();
	auto jMinusGlobalView = NBR.jMinusArray.getConstView();
	auto kMinusGlobalView = NBR.kMinusArray.getConstView();
	auto jPlusGlobalViewFine = GridBuilderFine.NBR.jPlusArray.getConstView();
	auto kPlusGlobalViewFine = GridBuilderFine.NBR.kPlusArray.getConstView();
	auto jMinusGlobalViewFine = GridBuilderFine.NBR.jMinusArray.getConstView();
	auto kMinusGlobalViewFine = GridBuilderFine.NBR.kMinusArray.getConstView();
	auto wallMarkerView = wallMarkerArray.getConstView();
	auto markerView = markerArray.getConstView();
	auto childMapGlobalView = childMapArrayGlobal.getConstView();
	
	BoolArrayType iSeeFullBlockMarkerArray( cellCountTotal );
	iSeeFullBlockMarkerArray.setValue( false );
	auto iSeeFullBlockMarkerView = iSeeFullBlockMarkerArray.getView();
	
	auto iSeeFullBlockLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( !markerView( cell ) ) return;
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusGlobalView( cell );
		NBR.kPlus = kPlusGlobalView( cell );
		NBR.jkPlus = jPlusGlobalView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		// check if this is part of the interface, this includes a wall check already (the supplied markerArray does not contain walls)
		if ( !markerView( NBR.iPlus ) || !markerView( NBR.jPlus ) || !markerView( NBR.ijPlus ) || !markerView( NBR.kPlus ) 
			|| !markerView( NBR.ikPlus ) || !markerView( NBR.jkPlus ) || !markerView( NBR.ijkPlus ) ) return;
		// check for position
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		if ( iView( NBR.iPlus ) != iCell+1 || iView( NBR.jPlus ) != iCell || iView( NBR.ijPlus ) != iCell+1 || iView( NBR.kPlus ) != iCell 
			|| iView( NBR.ikPlus ) != iCell+1 || iView( NBR.jkPlus ) != iCell || iView( NBR.ijkPlus ) != iCell+1 ) return;
		if ( jView( NBR.iPlus ) != jCell || jView( NBR.jPlus ) != jCell+1 || jView( NBR.ijPlus ) != jCell+1 || jView( NBR.kPlus ) != jCell 
			|| jView( NBR.ikPlus ) != jCell || jView( NBR.jkPlus ) != jCell+1 || jView( NBR.ijkPlus ) != jCell+1 ) return;
		if ( kView( NBR.iPlus ) != kCell || kView( NBR.jPlus ) != kCell || kView( NBR.ijPlus ) != kCell || kView( NBR.kPlus ) != kCell+1 
			|| kView( NBR.ikPlus ) != kCell+1 || kView( NBR.jkPlus ) != kCell+1 || kView( NBR.ijkPlus ) != kCell+1 ) return;
		
		const int cellFineTop = childMapGlobalView( NBR.ijkPlus );
		// check if the child exists
		if ( cellFineTop < 0 ) return;
		NBRStruct NBRFine;
		NBRFine.ijkPlus = cellFineTop;
		NBRFine.jkPlus = NBRFine.ijkPlus - 1; if ( NBRFine.jkPlus < 0 ) NBRFine.jkPlus = InfoFine.cellCount - 1;
		NBRFine.kPlus = jMinusGlobalViewFine( NBRFine.jkPlus );
		NBRFine.self = kMinusGlobalViewFine( NBRFine.kPlus );
		NBRFine.jPlus = jPlusGlobalViewFine( NBRFine.self );
		finishNBRPlus( NBRFine, InfoFine );
		// check position of all children
		if ( iViewFine( NBRFine.self ) != 2*iCell+1 || jViewFine( NBRFine.self ) != 2*jCell+1 || kViewFine( NBRFine.self ) != 2*kCell+1 ) return;
		if ( iViewFine( NBRFine.iPlus ) != 2*iCell+2 || iViewFine( NBRFine.jPlus ) != 2*iCell+1 || iViewFine( NBRFine.ijPlus ) != 2*iCell+2 || iViewFine( NBRFine.kPlus ) != 2*iCell+1 
			|| iViewFine( NBRFine.ikPlus ) != 2*iCell+2 || iViewFine( NBRFine.jkPlus ) != 2*iCell+1 || iViewFine( NBRFine.ijkPlus ) != 2*iCell+2 ) return;
		if ( jViewFine( NBRFine.iPlus ) != 2*jCell+1 || jViewFine( NBRFine.jPlus ) != 2*jCell+2 || jViewFine( NBRFine.ijPlus ) != 2*jCell+2 || jViewFine( NBRFine.kPlus ) != 2*jCell+1 
			|| jViewFine( NBRFine.ikPlus ) != 2*jCell+1 || jViewFine( NBRFine.jkPlus ) != 2*jCell+2 || jViewFine( NBRFine.ijkPlus ) != 2*jCell+2 ) return;
		if ( kViewFine( NBRFine.iPlus ) != 2*kCell+1 || kViewFine( NBRFine.jPlus ) != 2*kCell+1 || kViewFine( NBRFine.ijPlus ) != 2*kCell+1 || kViewFine( NBRFine.kPlus ) != 2*kCell+2 
			|| kViewFine( NBRFine.ikPlus ) != 2*kCell+2 || kViewFine( NBRFine.jkPlus ) != 2*kCell+2 || kViewFine( NBRFine.ijkPlus ) != 2*kCell+2 ) return;
		// all checks passed -> this is a full Geier block		
		iSeeFullBlockMarkerView( cell ) = true;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountTotal, iSeeFullBlockLambda );
	
	Interface.interfaceCount = TNL::sum( iSeeFullBlockMarkerArray );
	Interface.indexArray.setSize( Interface.interfaceCount );
	Interface.childMapArray.setSize( Interface.interfaceCount );
	
	IntArrayType scanArray( iSeeFullBlockMarkerArray.getSize() );
	intArrayFromBoolArray( scanArray, iSeeFullBlockMarkerArray );
	TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, cellCountTotal, TNL::Plus{} );
	
	auto scanView = scanArray.getConstView();
	auto indexView = Interface.indexArray.getView();
	auto childMapView = Interface.childMapArray.getView();
	
	auto indexListLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( !iSeeFullBlockMarkerView( cell ) ) return;
		const int index = scanView( cell );
		indexView( index ) = cell;
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusGlobalView( cell );
		NBR.kPlus = kPlusGlobalView( cell );
		NBR.jkPlus = jPlusGlobalView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		const int cellFineTop = childMapGlobalView( NBR.ijkPlus );
		NBRStruct NBRFine;
		NBRFine.ijkPlus = cellFineTop;
		NBRFine.jkPlus = NBRFine.ijkPlus - 1; if ( NBRFine.jkPlus < 0 ) NBRFine.jkPlus = InfoFine.cellCount - 1;
		NBRFine.kPlus = jMinusGlobalViewFine( NBRFine.jkPlus );
		NBRFine.self = kMinusGlobalViewFine( NBRFine.kPlus );
		
		childMapView( index ) = NBRFine.self;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCountTotal, indexListLambda );
	
	BoolArrayType valueReceivedMarkerArrayFine( GridBuilderFine.Info.cellCount );
	valueReceivedMarkerArrayFine.setValue( false );
	auto valueReceivedMarkerViewFine = valueReceivedMarkerArrayFine.getView();
	auto jPlusViewFine = GridBuilderFine.NBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridBuilderFine.NBR.kPlusArray.getConstView();
	
	auto valueReceivedLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int cellFine = childMapView( index );
		NBRStruct NBR;
		NBR.self = cellFine;
		NBR.jPlus = jPlusViewFine( cellFine );
		NBR.kPlus = kPlusViewFine( cellFine );
		NBR.jkPlus = jPlusViewFine( NBR.kPlus );
		finishNBRPlus( NBR, InfoFine );
		valueReceivedMarkerViewFine( NBR.self ) = true;
		valueReceivedMarkerViewFine( NBR.iPlus ) = true;
		valueReceivedMarkerViewFine( NBR.jPlus ) = true;
		valueReceivedMarkerViewFine( NBR.ijPlus ) = true;
		valueReceivedMarkerViewFine( NBR.kPlus ) = true;
		valueReceivedMarkerViewFine( NBR.ikPlus ) = true;
		valueReceivedMarkerViewFine( NBR.jkPlus ) = true;
		valueReceivedMarkerViewFine( NBR.ijkPlus ) = true;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Interface.interfaceCount, valueReceivedLambda );
	
	BoolArrayType leftoverMarkerArray( GridBuilderFine.Info.cellCount );
	leftoverMarkerArray = GridBuilderFine.needValuesFromCoarseMarkerArray
						* !GridBuilderFine.wallMarkerArray
						* !valueReceivedMarkerArrayFine;
	
	Interface.leftoverCount = TNL::sum( leftoverMarkerArray );
	Interface.leftoverIndexArray.resize( Interface.leftoverCount );
	Interface.leftoverParentMapArray.resize( Interface.leftoverCount );
	Interface.leftoverNbrIArray.resize( Interface.leftoverCount );
	Interface.leftoverNbrJArray.resize( Interface.leftoverCount );
	Interface.leftoverNbrKArray.resize( Interface.leftoverCount );
	
	scanArray.resize( GridBuilderFine.Info.cellCount );
	intArrayFromBoolArray( scanArray, leftoverMarkerArray );
	TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, GridBuilderFine.Info.cellCount, TNL::Plus{} );
	
	auto leftoverMarkerView = leftoverMarkerArray.getConstView();
	auto scanView2 = scanArray.getConstView();
	auto leftoverIndexView = Interface.leftoverIndexArray.getView();
	auto leftoverParentMapView = Interface.leftoverParentMapArray.getView();
	auto parentMapGlobalView = GridBuilderFine.parentMapArray.getConstView();
	auto leftoverNbrIView = Interface.leftoverNbrIArray.getView();
	auto leftoverNbrJView = Interface.leftoverNbrJArray.getView();
	auto leftoverNbrKView = Interface.leftoverNbrKArray.getView();
	
	auto leftoverLambda = [=] __cuda_callable__ ( const int cellFine ) mutable
	{	
		if ( !leftoverMarkerView( cellFine ) ) return;
		const int index = scanView2( cellFine );
		leftoverIndexView( index ) = cellFine;
		const int cellCoarse = parentMapGlobalView( cellFine );
		leftoverParentMapView( index ) = cellCoarse;
		const int iFine = iViewFine( cellFine );
		const int jFine = jViewFine( cellFine );
		const int kFine = kViewFine( cellFine );
		const int iCoarse = iView( cellCoarse );
		const int jCoarse = jView( cellCoarse );
		const int kCoarse = kView( cellCoarse );
		int nbrI = cellCoarse; int nbrJ = cellCoarse; int nbrK = cellCoarse;
		if ( iFine % 2 == 1 ) // iPlus direction
		{
			int candidateI = cellCoarse + 1; if ( candidateI >= Info.cellCount ) candidateI = 0;
			if ( markerView( candidateI ) && iView( candidateI ) == iCoarse+1 
			&& jView( candidateI ) == jCoarse && kView( candidateI ) == kCoarse ) nbrI = candidateI;
			else // positive candidate is not available -> try negative one
			{
				candidateI = cellCoarse - 1; if ( candidateI < 0 ) candidateI = Info.cellCount-1;
				if ( markerView( candidateI ) && iView( candidateI ) == iCoarse-1 
				&& jView( candidateI ) == jCoarse && kView( candidateI ) == kCoarse ) nbrI =  - candidateI - 1; 
				// coding the inverse side as negative index
			}
		}
		else // iMinus direction
		{
			int candidateI = cellCoarse - 1; if ( candidateI < 0 ) candidateI = Info.cellCount-1;
			if ( markerView( candidateI ) && iView( candidateI ) == iCoarse-1 
			&& jView( candidateI ) == jCoarse && kView( candidateI ) == kCoarse ) nbrI = candidateI;
			else // negative candidate is not available -> try positive one
			{
				candidateI = cellCoarse + 1; if ( candidateI >= Info.cellCount ) candidateI = 0;
				if ( markerView( candidateI ) && iView( candidateI ) == iCoarse+1 
				&& jView( candidateI ) == jCoarse && kView( candidateI ) == kCoarse ) nbrI = - candidateI - 1; 
				// coding the inverse side as negative index
			}
		}
		if ( jFine % 2 == 1 ) // jPlus direction
		{
			int candidateJ = jPlusGlobalView( cellCoarse );
			if ( markerView( candidateJ ) && iView( candidateJ ) == iCoarse 
			&& jView( candidateJ ) == jCoarse+1 && kView( candidateJ ) == kCoarse ) nbrJ = candidateJ;
			else // positive candidate is not available -> try negative one
			{
				candidateJ = jMinusGlobalView( cellCoarse );
				if ( markerView( candidateJ ) && iView( candidateJ ) == iCoarse 
				&& jView( candidateJ ) == jCoarse-1 && kView( candidateJ ) == kCoarse ) nbrJ = - candidateJ - 1;
			}
		}
		else // jMinus direction
		{
			int candidateJ = jMinusGlobalView( cellCoarse );
			if ( markerView( candidateJ ) && iView( candidateJ ) == iCoarse 
			&& jView( candidateJ ) == jCoarse-1 && kView( candidateJ ) == kCoarse ) nbrJ = candidateJ;
			else // negative candidate is not available -> try positive one
			{
				candidateJ = jPlusGlobalView( cellCoarse );
				if ( markerView( candidateJ ) && iView( candidateJ ) == iCoarse 
				&& jView( candidateJ ) == jCoarse+1 && kView( candidateJ ) == kCoarse ) nbrJ = - candidateJ - 1;
			}
		}
		if ( kFine % 2 == 1 ) // kPlus direction
		{
			int candidateK = kPlusGlobalView( cellCoarse );
			if ( markerView( candidateK ) && iView( candidateK ) == iCoarse 
			&& jView( candidateK ) == jCoarse && kView( candidateK ) == kCoarse+1 ) nbrK = candidateK;
			else // positive candidate is not available -> try negative one
			{
				candidateK = kMinusGlobalView( cellCoarse );
				if ( markerView( candidateK ) && iView( candidateK ) == iCoarse 
				&& jView( candidateK ) == jCoarse && kView( candidateK ) == kCoarse-1 ) nbrK = - candidateK - 1;
			}
		}
		else // kMinus direction
		{
			int candidateK = kMinusGlobalView( cellCoarse );
			if ( markerView( candidateK ) && iView( candidateK ) == iCoarse 
			&& jView( candidateK ) == jCoarse && kView( candidateK ) == kCoarse-1 ) nbrK = candidateK;
			else // negative candidate is not available -> try positive one
			{
				candidateK = kPlusGlobalView( cellCoarse );
				if ( markerView( candidateK ) && iView( candidateK ) == iCoarse 
				&& jView( candidateK ) == jCoarse && kView( candidateK ) == kCoarse+1 ) nbrK = - candidateK - 1;
			}
		}
		leftoverNbrIView( index ) = nbrI;
		leftoverNbrJView( index ) = nbrJ;
		leftoverNbrKView( index ) = nbrK;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridBuilderFine.Info.cellCount, leftoverLambda );
}

void gridBuilderToGrid( std::vector<GridBuilderStruct> &gridBuilders, std::vector<GridStruct> &grids, const int level )
{
	if ( level == 0 ) std::cout << "Passing grid data from GridBuilder to Grid for all levels" << std::endl; 
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridBuilderStruct &GridBuilder = gridBuilders[ level ];	
	GridStruct &Grid = grids[ level ];	
	
	// 1) copy Info, that can be copied directly
	Grid.Info = GridBuilder.Info;
	InfoStruct &Info = Grid.Info;
	
	// 2) build scans to be able to build compressed IJKNBR
	IntArrayType firstInRowArray( Info.cellCount );
	IntArrayType scanArray( Info.cellCount );
	auto firstInRowView = firstInRowArray.getView();
	auto scanView = scanArray.getView();
	auto iBuilderView = GridBuilder.IJK.iArray.getConstView();
	auto jBuilderView = GridBuilder.IJK.jArray.getConstView();
	auto kBuilderView = GridBuilder.IJK.kArray.getConstView();
	auto jPlusBuilderView = GridBuilder.NBR.jPlusArray.getConstView();
	auto kPlusBuilderView = GridBuilder.NBR.kPlusArray.getConstView();
	auto firstInRowMarkerLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( cell == 0 ) 
		{
			firstInRowView( cell ) = cell;
			scanView( cell ) = 1;
			return;
		}
		const int iCell = iBuilderView( cell );
		const int jCell = jBuilderView( cell );
		const int kCell = kBuilderView( cell );
		const int jPlus = jPlusBuilderView( cell );
		const int kPlus = kPlusBuilderView( cell );
		const int jkPlus = jPlusBuilderView( kPlus );
		
		const int iPrev = iBuilderView( cell-1 );
		const int jPrev = jBuilderView( cell-1 );
		const int kPrev = kBuilderView( cell-1 );
		const int jPlusPrev = jPlusBuilderView( cell-1 );
		const int kPlusPrev = kPlusBuilderView( cell-1 );
		const int jkPlusPrev = jPlusBuilderView( kPlusPrev );
		if ( 	iCell != iPrev+1 	 || jCell != jPrev 		 || kCell  != kPrev || 
				jPlus != jPlusPrev+1 || kPlus != kPlusPrev+1 || jkPlus != jkPlusPrev+1 ) 
		{
			firstInRowView( cell ) = cell;
			scanView( cell ) = 1;
		}
		else 
		{
			firstInRowView( cell ) = 0;
			scanView( cell ) = 0;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, firstInRowMarkerLambda );
	TNL::Algorithms::inplaceInclusiveScan( firstInRowArray, 0, Info.cellCount, TNL::Max{} );
	const int compressedIJKCount = TNL::sum( scanArray );
	TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, Info.cellCount, TNL::Plus{} );
	
	// 3) build compressed IJKNBR
	Grid.IJKNBR.shifterArray.setSize( Info.cellCount );
	Grid.IJKNBR.iArray.setSize( compressedIJKCount );
	Grid.IJKNBR.jArray.setSize( compressedIJKCount );
	Grid.IJKNBR.kArray.setSize( compressedIJKCount );
	Grid.IJKNBR.jPlusArray.setSize( compressedIJKCount );
	Grid.IJKNBR.kPlusArray.setSize( compressedIJKCount );
	Grid.IJKNBR.jkPlusArray.setSize( compressedIJKCount );
	auto shifterView = Grid.IJKNBR.shifterArray.getView();
	auto iView = Grid.IJKNBR.iArray.getView();
	auto jView = Grid.IJKNBR.jArray.getView();
	auto kView = Grid.IJKNBR.kArray.getView();
	auto jPlusView = Grid.IJKNBR.jPlusArray.getView();
	auto kPlusView = Grid.IJKNBR.kPlusArray.getView();
	auto jkPlusView = Grid.IJKNBR.jkPlusArray.getView();
	auto compressedIJKLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		const int firstInRow = firstInRowView( cell );
		if ( cell == firstInRow ) // we are first in row
		{
			const int compressedIndex = scanView( cell );
			shifterView( cell ) = compressedIndex;
			iView( compressedIndex ) = iBuilderView( cell );
			jView( compressedIndex ) = jBuilderView( cell );
			kView( compressedIndex ) = kBuilderView( cell );
			jPlusView( compressedIndex ) = jPlusBuilderView( cell );
			kPlusView( compressedIndex ) = kPlusBuilderView( cell );
			jkPlusView( compressedIndex ) = jPlusBuilderView( kPlusBuilderView( cell ) );
		}
		else
		{
			shifterView( cell ) = firstInRow - cell; // now this tells how much I need to shift backwards to find firstInRow
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, compressedIJKLambda );
	
	// 4) Build wallMap
	Grid.Wall.wallMapArray.setSize( Info.cellCount );
	Grid.Wall.wallMapArray.setValue( -1 ); // set to "free fluid" as default
	Grid.Wall.wallAdjacentCount = GridBuilder.wallAdjacentCellList.getSize();
	Grid.Wall.indexArray.setSize( Grid.Wall.wallAdjacentCount );
	Grid.Wall.indexArray = GridBuilder.wallAdjacentCellList;
	auto wallMapView = Grid.Wall.wallMapArray.getView();
	auto wallMarkerView = GridBuilder.wallMarkerArray.getConstView();
	auto wallAdjacentCellListView = GridBuilder.wallAdjacentCellList.getConstView();
	auto interfaceOverlapView = GridBuilder.interfaceOverlapMarkerArray.getConstView();
	auto wallMarkerLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( wallMarkerView( cell ) ) wallMapView( cell ) = -3; // overwrite where wall is
		else if ( interfaceOverlapView( cell ) ) wallMapView( cell ) = -2; // free fluid under a parent interface -> dont track force
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, wallMarkerLambda );
	auto wallAdjacentLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int cell = wallAdjacentCellListView( index );
		wallMapView( cell ) = index; // overwrite where wall adjacent fluid is
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Wall.wallAdjacentCount, wallAdjacentLambda );
	
	// 5) Build wallData
	Grid.Wall.wallDataArray.setSize( Grid.Wall.wallAdjacentCount );
	Grid.Wall.linkLengthArray.setSizes( 7, Grid.Wall.wallAdjacentCount );
	auto wallDataView = Grid.Wall.wallDataArray.getView();
	auto linkExistenceMarkerView = GridBuilder.linkExistenceMarkerArray.getConstView();
	auto linkLengthBuilderView = GridBuilder.linkLengthArray.getConstView();
	auto linkLengthView = Grid.Wall.linkLengthArray.getView();
	auto wallIDView = GridBuilder.wallIDArray.getConstView();
	auto wallDataLambda = [=] __cuda_callable__ ( const int index ) mutable
	{	
		const int cell = wallAdjacentCellListView( index );
		bool interfaceOverlapMarker = interfaceOverlapView( cell );
		bool linkExistenceMarker[26];
		float linkLength[26];
		for ( int direction = 1; direction < 27; direction++ ) 
		{
			linkExistenceMarker[direction-1] = linkExistenceMarkerView( direction, index );
			linkLength[direction-1] = linkLengthBuilderView( direction, index );
		}
		const int wallID = wallIDView( index );
		// now take wallID, linkExistenceMarker, inverfaceOverlapMarker and pack it into wallData
		// also take linkLength, pack those into 7 uints, only write those that exist
		uint32_t wallData;
		uint32_t packed[7];
		// call the packing function
		packWallData( wallData, packed, wallID, interfaceOverlapMarker, linkExistenceMarker, linkLength );
		wallDataView( index ) = wallData;
		for ( int i = 0; i < 7; i++ ) linkLengthView( i, index ) = packed[i];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Wall.wallAdjacentCount, wallDataLambda );
	// allocate force tracker
	Grid.Wall.gxArray.setSize( Grid.Wall.wallAdjacentCount );
	Grid.Wall.gyArray.setSize( Grid.Wall.wallAdjacentCount );
	Grid.Wall.gzArray.setSize( Grid.Wall.wallAdjacentCount );
	Grid.Wall.gxArray.setValue( 0.f );
	Grid.Wall.gyArray.setValue( 0.f );
	Grid.Wall.gzArray.setValue( 0.f );
	
	if ( !iAmFinest )
	{
		// 6) Build interface
		// Build the lists we will loop over to pass information
		// On wall cells, no information is exchanged so do not include wall cells even if they are marked as interface
		// First, prepare a full childMapArrayGlobal
		IntArrayType childMapArrayGlobal( Info.cellCount );
		childMapArrayGlobal.setValue( -1 );
		auto childMapGlobalView = childMapArrayGlobal.getView();
		// Fill it using information from the finer level
		GridBuilderStruct &GridBuilderFiner = gridBuilders[ level + 1 ];
		auto iFinerView = GridBuilderFiner.IJK.iArray.getConstView();
		auto jFinerView = GridBuilderFiner.IJK.jArray.getConstView();
		auto kFinerView = GridBuilderFiner.IJK.kArray.getConstView();
		auto parentMapFinerView = GridBuilderFiner.parentMapArray.getConstView();
		auto childFillLambda = [=] __cuda_callable__ ( const int cellFine ) mutable
		{	
			const int cellCoarse = parentMapFinerView( cellFine );
			if ( cellCoarse < 0 ) return;
			const int iCellFine = iFinerView( cellFine );
			const int jCellFine = jFinerView( cellFine );
			const int kCellFine = kFinerView( cellFine );
			if ( iCellFine % 2 == 0 && jCellFine % 2 == 0 && kCellFine % 2 == 0 )
			{
				childMapGlobalView( cellCoarse ) = cellFine;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridBuilderFiner.Info.cellCount, childFillLambda );
		
		BoolArrayType markerArray( Info.cellCount );
		// Fine to coarse
		markerArray = GridBuilder.fineToCoarseMarkerArray * !GridBuilder.wallMarkerArray;
		fillFineToCoarseInterface( Grid.FineToCoarseInterface, markerArray, childMapArrayGlobal, GridBuilder );
		// Coarse to fine
		markerArray = GridBuilder.coarseToFineMarkerArray * !GridBuilder.wallMarkerArray;
		fillCoarseToFineInterface( Grid.CoarseToFineInterface, markerArray, childMapArrayGlobal, GridBuilder, GridBuilderFiner );
		std::cout << "	Leftover fine cells on interface between grids " << level << ", " << level+1 << ": " 
					<< Grid.CoarseToFineInterface.leftoverCount << " out of " 
					<< Grid.CoarseToFineInterface.interfaceCount * 8 + Grid.CoarseToFineInterface.leftoverCount << std::endl;
	}
	
	// 6) Build OpenBCs
	IntArrayType BCIDArray( Grid.Info.cellCount );
	BCIDArray.setValue( -1 );
	auto BCIDView = BCIDArray.getView();
	auto BCIDLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( wallMarkerView( cell ) ) return;
		const int iCell = iBuilderView( cell );
		const int jCell = jBuilderView( cell );
		const int kCell = kBuilderView( cell );
		if ( iCell == 0 || iCell == Info.cellCountX - 1 || jCell == 0 || jCell == Info.cellCountY - 1 || kCell == 0 || kCell == Info.cellCountZ - 1 )
		{
			BCStruct BC;
			getOpenBC( BC, iCell, jCell, kCell, Info );
			BCIDView( cell ) = BC.openBCID;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Info.cellCount, BCIDLambda );
	const int BCIDMax = TNL::max( BCIDArray );
	const int BCIDCount = BCIDMax+1;
	// Now we know how long the openBCList should be
	scanArray.resize( Info.cellCount );
	BoolArrayType BCIDMarkerArray( Grid.Info.cellCount );
	auto BCIDMarkerView = BCIDMarkerArray.getView();
	Grid.openBCs.resize( BCIDCount );
	for ( int BCID = 0; BCID < BCIDCount; BCID++ )
	{
		Grid.openBCs[ BCID ].openBCID = BCID;
		BCIDMarkerArray.setValue( false );
		auto BCIDMarkerLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( BCIDView( cell ) == BCID ) BCIDMarkerView( cell ) = true;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Info.cellCount, BCIDMarkerLambda );
		Grid.openBCs[ BCID ].openBCCount = TNL::sum( BCIDMarkerArray );
		Grid.openBCs[ BCID ].indexArray.setSize( Grid.openBCs[ BCID ].openBCCount );
		Grid.openBCs[ BCID ].dRhoPrevArray.setSize( Grid.openBCs[ BCID ].openBCCount );
		Grid.openBCs[ BCID ].uNormalPrevArray.setSize( Grid.openBCs[ BCID ].openBCCount );
		Grid.openBCs[ BCID ].dRhoCumulativeArray.setSize( Grid.openBCs[ BCID ].openBCCount );
		Grid.openBCs[ BCID ].dRhoCumulativeArray.setValue( 0.f );
		Grid.openBCs[ BCID ].uNormalCumulativeArray.setSize( Grid.openBCs[ BCID ].openBCCount );
		Grid.openBCs[ BCID ].uNormalCumulativeArray.setValue( 0.f );
		intArrayFromBoolArray( scanArray, BCIDMarkerArray );
		TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, Info.cellCount, TNL::Plus{} );
		auto scanView2 = scanArray.getConstView();
		auto indexView = Grid.openBCs[ BCID ].indexArray.getView();
		
		auto indexArrayLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( !BCIDMarkerView( cell ) ) return;
			const int index = scanView2( cell );
			indexView( index ) = cell;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, indexArrayLambda );
		
		// identify how many cells are tracked
		BoolArrayType trackFlowMarkerArray( Grid.openBCs[ BCID ].openBCCount );
		auto trackFlowMarkerView = trackFlowMarkerArray.getView();
		auto flowTrackerLambda = [=] __cuda_callable__ ( const int index ) mutable
		{	
			const int cell = indexView( index );
			// read wallMap, this is to find if we should track flow through this cell
			// we dont track flow if the cell is under an interface overlap
			const int wallMap = wallMapView( cell );
			bool trackFlow = true;
			if ( wallMap == -2 ) trackFlow = false; // fluid cell under an interface overlap -> dont track flow
			// read wallData if this is a wall adjacent cell. So far only unpack wallID and interfaceOverlapMarker
			// read wallData if this is a wall adjacent cell. So far only unpack wallID and interfaceOverlapMarker
			int wallID = -1; 
			if ( wallMap >= 0 )
			{
				const uint32_t wallData = wallDataView( wallMap );
				bool interfaceOverlapMarker;
				unpackWallID( wallData, wallID, interfaceOverlapMarker );
				if ( interfaceOverlapMarker ) trackFlow = false;
			}
			trackFlowMarkerView( index ) = trackFlow;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.openBCs[ BCID ].openBCCount, flowTrackerLambda );
		Grid.openBCs[ BCID ].trackFlowCount = TNL::sum( trackFlowMarkerArray );
	}
		
	// 7) Recursion
	if ( !iAmFinest ) gridBuilderToGrid( gridBuilders, grids, level + 1 );
	else std::cout << std::endl;
}

void allocateFArray( GridStruct &Grid )
{
	InfoStruct &Info = Grid.Info;
	Grid.fArray.setSizes( 27, Info.cellCount );
}

long long buildGrids( std::vector<GridStruct> &grids, std::vector<STLStruct> &gridStaticSTLs, std::vector<STLStruct> &rotorSTLs, BoundsStruct DomainBounds )
{
	// call the temporary GridBuilders and Voxelizers - these will go out of scope
	// start the GridBuilder and Voxelizer scope
	long long fluidUpdatesPerIteration = 0LL;
	{
		std::vector<GridBuilderStruct> gridBuilders( GRID_LEVEL_COUNT );
		initializeGridInfo( gridBuilders, DomainBounds, 0 );
		
		// Pass the useRotors bool which is received with grids
		for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) gridBuilders[ level ].Info.useRotors = grids[ level ].Info.useRotors;
		
		// Voxelizers
		std::vector<VoxelizerStruct> voxelizers( GRID_LEVEL_COUNT + 1 ); 
		initializeVoxelizers( voxelizers, gridBuilders, gridStaticSTLs, 0 );
		
		// Build grids
		buildIJKFull( gridBuilders, voxelizers, 0 );
		deleteExcessCells( gridBuilders, voxelizers, 0 );
		buildWallMarkers( gridBuilders, voxelizers, gridStaticSTLs, 0 );
		
		// Pass the information to the actual grids, but in a compressed form
		gridBuilderToGrid( gridBuilders, grids, 0 );
		
		// while the GridBuilders exist, calculate fluid updates per iteration (do not include wall cells)
		for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) 
		{
			GridBuilderStruct &GridBuilder = gridBuilders[level];
			InfoStruct &Info = GridBuilder.Info;
			const int wallCount = TNL::sum( GridBuilder.wallMarkerArray );
			std::cout << " wall count " << wallCount <<  std::endl;
			fluidUpdatesPerIteration += (long long)( Info.cellCount - wallCount ) * (long long)std::pow( 2, grids[level].Info.gridID );
		}
	} // here GridBuilders and Voxelizers go out of scope
	
	// build rotors
	if ( rotorSTLs.size() > 0 )
	{
		std::cout << "Building rotors for selected grid levels" << std::endl;
		for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) 
		{
			if ( grids[ level ].Info.useRotors ) buildRotors( grids[ level ], rotorSTLs );
		}
		std::cout << std::endl;
	}
	
	std::cout << "Allocating fArray for all grid levels. Printed memory is total per level" << std::endl;
	long long totalMemoryBytes = 0LL;
	long long totalCells = 0LL;
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) 
	{
		GridStruct &Grid = grids[level];
		InfoStruct &Info = Grid.Info;
		
		Info.gridMemoryBytes = 27LL * (long long)Info.cellCount * 4LL; // fArray
		Info.gridMemoryBytes += 2LL * (long long)Info.cellCount * 4LL; // IJK shifter, wallMap
		Info.gridMemoryBytes += 6LL * (long long)Grid.IJKNBR.iArray.getSize() * 4LL; // compressed iArray, jArray, kArray, jPlusArray, kPlusArray, jkPlusArray
		Info.gridMemoryBytes += 12LL * (long long)Grid.Wall.wallAdjacentCount * 4LL; // index array + wall data + linkLengths + wall force tracker
		Info.gridMemoryBytes += 2LL * (long long)Grid.CoarseToFineInterface.interfaceCount * 4LL; // indexArray, childMap
		Info.gridMemoryBytes += 2LL * (long long)Grid.CoarseToFineInterface.leftoverCount * 4LL; // leftoverIndexArray, leftoverParentMap
		Info.gridMemoryBytes += 2LL * (long long)Grid.FineToCoarseInterface.interfaceCount * 4LL; // indexArray, childMap
		for ( int rotorID = 0; rotorID < (int)Grid.rotors.size(); rotorID++ )
		{
			Info.gridMemoryBytes += 1LL * (long long)Grid.rotors[ rotorID ].rotorMapArray.getSize() * 4LL; // rotorMap
			Info.gridMemoryBytes += 65LL * (long long)Grid.rotors[ rotorID ].indexArray.getSize() * 4LL; // indexArray, interpolationArray
			Info.gridMemoryBytes += 81LL * (long long)Grid.rotors[ rotorID ].indexArray.getSize() * 4LL; // rotor force tracker
		}
		for ( int openBCID = 0; openBCID < (int)Grid.openBCs.size(); openBCID++ )
		{
			Info.gridMemoryBytes += 5LL * (long long)Grid.openBCs[ openBCID ].openBCCount * 4LL; // indexList, rhoPrev, uNormalPrev, rhoCumulative, uNormalCumulative
		}
		std::cout << "	Level " << level << " with " << Info.cellCount << "	cells requires	" << Info.gridMemoryBytes / 1048576.0 << "	MiB ... " << std::flush;;
		
		allocateFArray( Grid );
		
		std::cout << "Done" << std::endl;
		
		totalMemoryBytes += grids[level].Info.gridMemoryBytes;
		totalCells += grids[level].Info.cellCount;
	}
	std::cout << std::endl;
	std::cout << "Total cells: " << totalCells << std::endl;
	std::cout << "Total GPU memory: " << totalMemoryBytes / 1048576.0 << " MiB"  << std::endl;
	std::cout << "Total fluid cell updates per iteration: " << fluidUpdatesPerIteration << std::endl;
	std::cout << std::endl;
	std::cout << "Applying initial condition" << std::endl;
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ ) applyInitialCondition( grids[ level ] );
	std::cout << std::endl;
	return fluidUpdatesPerIteration;
}
