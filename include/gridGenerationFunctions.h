#pragma once

#include "./types.h"
#include "./markerFunctions.h"

void initializeGridInfo( std::vector<GridStruct> &grids, const BoundsStruct &Bounds, const int level )
{
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];
	InfoStruct &Info = Grid.Info;
	
	if ( iAmCoarsest )
	{
		Info.res = RES_GLOBAL;
		
		SkeletonGridStruct &SkeletonGrid = Grid.SkeletonGrid;
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
		GridStruct &GridCoarse = grids[ level-1 ];
		Info.res = GridCoarse.Info.res * 0.5f;
		Info.cellCountX = GridCoarse.Info.cellCountX * 2;
		Info.cellCountY = GridCoarse.Info.cellCountY * 2;
		Info.cellCountZ = GridCoarse.Info.cellCountZ * 2;
		Info.ox = GridCoarse.Info.ox - Info.res * 0.5f;
		Info.oy = GridCoarse.Info.oy - Info.res * 0.5f;
		Info.oz = GridCoarse.Info.oz - Info.res * 0.5f;
		Info.dtPhys = GridCoarse.Info.dtPhys * 0.5f;
		Info.nu = (Info.dtPhys * nuPhys) / ((Info.res/1000.f) * (Info.res/1000.f));
	}
	
	if ( !iAmFinest ) initializeGridInfo( grids, Bounds, level+1 );
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

void buildFinerGrid( SkeletonGridStruct &SkeletonGrid, GridStruct &GridFine )
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
	
	// label stuff for GridFine
	const int cellCountFine = GridFine.Info.cellCount;
	IntArrayType &iArrayFine = GridFine.IJK.iArray;
	IntArrayType &jArrayFine = GridFine.IJK.jArray;
	IntArrayType &kArrayFine = GridFine.IJK.kArray;
	// Some GridFine Views
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

void buildFinerGrid( GridStruct &GridCoarse, GridStruct &GridFine )
{
	// label stuff for GridCoarse
	const int cellCountCoarse = GridCoarse.Info.cellCount;
	const IntArrayType &iArrayCoarse = GridCoarse.IJK.iArray;
	const IntArrayType &jArrayCoarse = GridCoarse.IJK.jArray;
	const IntArrayType &kArrayCoarse = GridCoarse.IJK.kArray;
	BoolArrayType &refinementMarkerArrayCoarse = GridCoarse.refinementMarkerArray;
	// Some GridCoarse Views
	auto iViewCoarse = iArrayCoarse.getConstView();
	auto jViewCoarse = jArrayCoarse.getConstView();
	auto kViewCoarse = kArrayCoarse.getConstView();
	auto refinementMarkerViewCoarse = refinementMarkerArrayCoarse.getConstView();
	
	// label stuff for GridFine
	const int cellCountFine = GridFine.Info.cellCount;
	IntArrayType &iArrayFine = GridFine.IJK.iArray;
	IntArrayType &jArrayFine = GridFine.IJK.jArray;
	IntArrayType &kArrayFine = GridFine.IJK.kArray;
	// Some GridFine Views
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

void buildIJKFull( std::vector<GridStruct> &grids, const std::vector<VoxelizerStruct> &voxelizers, const int level )
// Builds uncompressed full IJK for all grid levels recursively. 
// "Full" IJK means that the area on coarse grid that has a finer grid on top does not get deleted (yet)
{
	std::cout << "Building full IJK for grid level " << level << std::endl;
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];	
	InfoStruct &Info = Grid.Info;	
	
	SkeletonGridStruct &SkeletonGrid = Grid.SkeletonGrid;
	InfoStruct &SkeletonInfo = SkeletonGrid.Info;
	
	static GridStruct dummyGrid; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridStruct &GridCoarse = iAmCoarsest ? dummyGrid : grids[ level - 1 ];
	
	// 1) On coarser grid, mark refinement area where our level will be built, from that calculate our cellCount
	if ( iAmCoarsest )
	{
		SkeletonGrid.keepCellMarkerArray.setSize( SkeletonInfo.cellCount );
		markKeepCells( SkeletonGrid, voxelizers );
		Info.cellCount = 8 * TNL::sum( SkeletonGrid.keepCellMarkerArray );
	}
	else
	{
		markRefinementCells( GridCoarse, voxelizers );
		Info.cellCount = 8 * TNL::sum( GridCoarse.refinementMarkerArray );
	}
	std::cout << "	Initial cellCount set to " << Info.cellCount << std::endl;
	
	// 2) Set size of our arrays to cellCount
	Grid.IJK.iArray.setSize( Info.cellCount );
	Grid.IJK.jArray.setSize( Info.cellCount );
	Grid.IJK.kArray.setSize( Info.cellCount );
	Grid.NBR.jPlusArray.setSize( Info.cellCount );
	Grid.NBR.kPlusArray.setSize( Info.cellCount );
	Grid.NBR.isGeometricBitPackedMarkerArray.setSize( Info.cellCount );
	Grid.keepCellMarkerArray.setSize( Info.cellCount );	
	Info.gridMemoryBytes += (long long)(5 * 4 + 1 * 1) * (long long)(Info.cellCount); // 5 int arrays, 1 uint8_t
	if ( !iAmFinest )
	{
		Grid.refinementMarkerArray.setSize( Info.cellCount );
		Grid.deepRefinementMarkerArray.setSize( Info.cellCount );
		Grid.fineToCoarseMarkerArray.setSize( Info.cellCount );
		Grid.coarseToFineMarkerArray.setSize( Info.cellCount );
		Info.gridMemoryBytes += (long long)(4 * 1) * (long long)(Info.cellCount); // 4 bool arrays
	}
	if ( !iAmCoarsest )
	{
		Grid.parentMapArray.setSize( Info.cellCount );
		Grid.parentInterfaceMarkerArray.setSize( Info.cellCount );
		Info.gridMemoryBytes += (long long)(1 * 4 + 1 * 1) * (long long)(Info.cellCount); // 1 int array, 1 bool array
	}
	std::cout << "	Initial arrays allocated on GPU, it takes " << Info.gridMemoryBytes / 1048576.0 << " MiB" << std::endl;
	
	// 3) Build our grid = fill our IJK (we are the "finer grid" with respect to the grid we are taking spatial information from)
	if ( iAmCoarsest ) buildFinerGrid( SkeletonGrid, Grid );
	else buildFinerGrid( GridCoarse, Grid );
	
	// 4) Sort our IJK so that k changes the slowest
	std::cout << "	Sorting IJK" << std::endl;
	sortIJK( Grid.IJK );
	
	// 5) Build our NBR Plus and mark geometric validity
	std::cout << "	Building NBR Plus" << std::endl;
	buildNBRPlus( Grid );
	markGeometricNBRPlus( Grid );
	
	// 6) Build parentMapArray
	if ( !iAmCoarsest )
	{
		std::cout << "	Building parentMapArray" << std::endl;
		IJKArrayStruct IJKWanted;
		IJKWanted.iArray = Grid.IJK.iArray / 2;
		IJKWanted.jArray = Grid.IJK.jArray / 2;
		IJKWanted.kArray = Grid.IJK.kArray / 2;
		binarySearchIJK( IJKWanted, GridCoarse.IJK, Grid.parentMapArray );
	}
	
	// 7) Mark our cells that are part of the parent interface. A fine cell located at the parent interface:
	//    - is blocked from getting deleted later
	//	  - is blocked from getting deeply refined (interface with finer grid is still allowed)
	//	  - inherits fluid / wall state from the parent, even if the voxelizer says otherwise
	if ( !iAmCoarsest )
	{
		std::cout << "	Marking parent interface" << std::endl;
		auto parentInterfaceMarkerView = Grid.parentInterfaceMarkerArray.getView();
		auto parentMapView = Grid.parentMapArray.getConstView();
		auto parentCoarseToFineMarkerView = GridCoarse.coarseToFineMarkerArray.getConstView();
		auto parentFineToCoarseMarkerView = GridCoarse.fineToCoarseMarkerArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			const int parentCell = parentMapView( cell );
			const bool parentInterfaceMarker = (parentCoarseToFineMarkerView( parentCell ) + parentFineToCoarseMarkerView( parentCell ));
			parentInterfaceMarkerView( cell ) = parentInterfaceMarker;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
	}
	
	// 8) Recursion
	if ( !iAmFinest ) buildIJKFull( grids, voxelizers, level + 1 );
}

void deleteExcessCells( std::vector<GridStruct> &grids, const std::vector<VoxelizerStruct> &voxelizers, const int level )
// Delete cells from each level that are deep inside a wall or deeply refined
// Enforce keep cells that are part of the parent interface
{
	std::cout << "Deleting excess cells for grid level " << level << std::endl;
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];	
	InfoStruct &Info = Grid.Info;	
	const VoxelizerStruct &Voxelizer = voxelizers[ level ];
	
	static GridStruct dummyGrid; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridStruct &GridCoarse = iAmCoarsest ? dummyGrid : grids[ level - 1 ];
	
	// 1) Mark fluid cells and add one layer of walls
	markWallCells( Grid.keepCellMarkerArray, Voxelizer.rayMapTotal, Grid );
	Grid.keepCellMarkerArray = !Grid.keepCellMarkerArray; // now keepCellMarkerArray marks fluid cells only
	BoolArrayType markerSource;
	markerSource = Grid.keepCellMarkerArray;
	spreadMarkers( Grid.keepCellMarkerArray, markerSource, Grid ); // added one layer of walls
	
	// 2) Remove deeply refined cells
	if ( !iAmFinest ) Grid.keepCellMarkerArray = Grid.keepCellMarkerArray * !Grid.deepRefinementMarkerArray; 
	
	// 3) Because we changed the coarser grid, we must rebuild the parentMapArray and parentInterfaceMarkerArray
	if ( !iAmCoarsest )
	{
		std::cout << "	Rebuilding parentMapArray" << std::endl;
		IJKArrayStruct IJKWanted;
		IJKWanted.iArray = Grid.IJK.iArray / 2;
		IJKWanted.jArray = Grid.IJK.jArray / 2;
		IJKWanted.kArray = Grid.IJK.kArray / 2;
		binarySearchIJK( IJKWanted, GridCoarse.IJK, Grid.parentMapArray );
	}
	if ( !iAmCoarsest )
	{
		std::cout << "	Rebuilding parentInterfaceMarkerArray" << std::endl;
		auto parentInterfaceMarkerView = Grid.parentInterfaceMarkerArray.getView();
		auto parentMapView = Grid.parentMapArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			const int parentCell = parentMapView( cell );
			if ( parentCell < 0 ) parentInterfaceMarkerView( cell ) = false; // parent cell does not exist here so there is certainly no interface
			else parentInterfaceMarkerView( cell ) = true; // because we deleted deep refinement cells from the parent, this must only be the interface
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );	
	}
	
	// 4) Enforce keep cells that are part of the parent interface. 
	if ( !iAmCoarsest ) Grid.keepCellMarkerArray += Grid.parentInterfaceMarkerArray; 
	
	// 5) Build fullToKeep map
	IntArrayType fullToKeepMapArray( Info.cellCount );
	intArrayFromBoolArray( fullToKeepMapArray, Grid.keepCellMarkerArray );
	TNL::Algorithms::inplaceExclusiveScan( fullToKeepMapArray, 0, Info.cellCount, TNL::Plus{} );
	
	// 6) Transform necessary information from full grid to the keep grid
	// We need IJK, parentMapArray, fineToCoarseInterfaceMarkerArray, coarseToFineInterfaceMarkerArray	
	// starting a scope so that temporary arrays then go out of scope
	{ 	
		IJKArrayStruct IJKFull = Grid.IJK;
		IntArrayType parentMapArrayFull;
		parentMapArrayFull = Grid.parentMapArray;
		BoolArrayType fineToCoarseMarkerArrayFull;
		fineToCoarseMarkerArrayFull = Grid.fineToCoarseMarkerArray;
		BoolArrayType coarseToFineMarkerArrayFull;
		coarseToFineMarkerArrayFull = Grid.coarseToFineMarkerArray;
		
		auto keepCellMarkerView = Grid.keepCellMarkerArray.getConstView();
		auto fullToKeepMapView = fullToKeepMapArray.getConstView();
		
		auto iFullView = IJKFull.iArray.getConstView();
		auto jFullView = IJKFull.jArray.getConstView();
		auto kFullView = IJKFull.kArray.getConstView();
		auto parentMapFullView = parentMapArrayFull.getConstView();
		auto fineToCoarseMarkerFullView = fineToCoarseMarkerArrayFull.getConstView();
		auto coarseToFineMarkerFullView = coarseToFineMarkerArrayFull.getConstView();
		auto iView = Grid.IJK.iArray.getView();
		auto jView = Grid.IJK.jArray.getView();
		auto kView = Grid.IJK.kArray.getView();
		auto parentMapView = Grid.parentMapArray.getView();
		auto fineToCoarseMarkerView = Grid.fineToCoarseMarkerArray.getView();
		auto coarseToFineMarkerView = Grid.coarseToFineMarkerArray.getView();
		
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
	Info.cellCount = TNL::sum( Grid.keepCellMarkerArray );
	std::cout << "	Final cellCount set to " << Info.cellCount << std::endl;
	
	// 8) Resize the necessary arrays. Note that NBR is now broken and will have to be rebuilt again
	Grid.IJK.iArray.resize( Info.cellCount );
	Grid.IJK.jArray.resize( Info.cellCount );
	Grid.IJK.kArray.resize( Info.cellCount );
	Grid.NBR.jPlusArray.resize( Info.cellCount );
	Grid.NBR.kPlusArray.resize( Info.cellCount );
	Grid.NBR.jMinusArray.resize( Info.cellCount );
	Grid.NBR.kMinusArray.resize( Info.cellCount );
	Grid.NBR.isGeometricBitPackedMarkerArray.resize( Info.cellCount );
	Grid.wallMarkerArray.resize( Info.cellCount );
	if (!iAmCoarsest) Grid.parentMapArray.resize( Info.cellCount );
	if (!iAmFinest) Grid.fineToCoarseMarkerArray.resize( Info.cellCount );
	if (!iAmFinest) Grid.coarseToFineMarkerArray.resize( Info.cellCount );
	
	// 9) Forget the no longer necessary arrays
	Grid.SkeletonGrid.keepCellMarkerArray.resize( 0 );
	Grid.deepRefinementMarkerArray.resize( 0 );
	Grid.refinementMarkerArray.resize( 0 );
	
	// 10) Rebuild our NBR Plus
	std::cout << "	Rebuilding NBR Plus" << std::endl;
	buildNBRPlus( Grid );
	markGeometricNBRPlus( Grid );
	
	// 11) Build our NBR Minus
	if ( !iAmCoarsest )
	{
		std::cout << "	Building NBR Minus" << std::endl;
		auto jPlusView = Grid.NBR.jPlusArray.getConstView();
		auto kPlusView = Grid.NBR.kPlusArray.getConstView();
		auto jMinusView = Grid.NBR.jMinusArray.getView();
		auto kMinusView = Grid.NBR.kMinusArray.getView();
		auto NBRMinusLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			jMinusView[ jPlusView[ cell ] ] = cell;
			kMinusView[ kPlusView[ cell ] ] = cell;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, NBRMinusLambda );
	}
	
	// 12) Recursion
	if ( !iAmFinest ) deleteExcessCells( grids, voxelizers, level + 1 );
}

void buildWallMarkers( std::vector<GridStruct> &grids, const std::vector<VoxelizerStruct> &voxelizers, const int level )
// Delete cells from each level that are deep inside a wall or deeply refined
// Enforce keep cells that are part of the parent interface
{
	std::cout << "Building wall markers for grid level " << level << std::endl;
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];	
	InfoStruct &Info = Grid.Info;	
	const VoxelizerStruct &Voxelizer = voxelizers[ level ];
	
	static GridStruct dummyGrid; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridStruct &GridCoarse = iAmCoarsest ? dummyGrid : grids[ level - 1 ];
	
	// 1) Final marking of the wall
	std::cout << "	Final wall marking" << std::endl;
	markWallCells( Grid.wallMarkerArray, Voxelizer.rayMapTotal, Grid );
	if ( !iAmCoarsest ) // if we are not coarsest, inherit wall state at parent interface from the parent
	{
		auto parentMapView = Grid.parentMapArray.getConstView();
		auto parentWallMarkerView = GridCoarse.wallMarkerArray.getConstView();
		auto wallMarkerView = Grid.wallMarkerArray.getView();
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
	Grid.wallIDArray.setSize( Info.cellCount );
	Grid.wallIDArray.setValue( -1 );
	
	BoolArrayType markerSourceArray( Info.cellCount );
	BoolArrayType bodyWallMarkerArray( Info.cellCount );
	
	for ( int wallID = 0; wallID < (int)Voxelizer.rayMaps.size(); wallID++ )
	{
		const RayMapStruct &RayMap = Voxelizer.rayMaps[ wallID ];
		markWallCells( bodyWallMarkerArray, RayMap, Grid );
		// At the interface, we must overwrite marker for this wall by the parent cells
		// Generate temporary parent wall marker
		BoolArrayType parentBodyWallMarkerArray( GridCoarse.Info.cellCount );
		if (!iAmCoarsest) 
		{
			markWallCells( parentBodyWallMarkerArray, voxelizers[level-1].rayMaps[ wallID ], GridCoarse );
			auto bodyWallMarkerView = bodyWallMarkerArray.getView();
			auto parentMapView = Grid.parentMapArray.getConstView();
			auto parentBodyWallMarkerView = parentBodyWallMarkerArray.getConstView();
			auto wallIDLambda = [=] __cuda_callable__ ( const int cell ) mutable
			{	
				const int parentCell = parentMapView( cell );
				if ( parentCell >= 0 ) bodyWallMarkerView( cell ) = parentBodyWallMarkerView( parentCell );
			};
			TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, wallIDLambda );
		}
		markerSourceArray = bodyWallMarkerArray;
		spreadMarkers( bodyWallMarkerArray, markerSourceArray, Grid );
		bodyWallMarkerArray = bodyWallMarkerArray * !Grid.wallMarkerArray;
		auto bodyWallMarkerView = bodyWallMarkerArray.getConstView();
		auto wallIDView = Grid.wallIDArray.getView();
		auto wallIDLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( bodyWallMarkerView( cell ) ) wallIDView( cell ) = wallID;	
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, wallIDLambda );
	}
	
	// 3) Recursion
	if ( !iAmFinest ) buildWallMarkers( grids, voxelizers, level + 1 );
}

/*
#include "./genericArrayFunctions.h"
#include "./voxelizerFunctions.h"
#include "./NBRFunctions.h"
#include "./markerFunctions.h"
#include "./updateInterface.h"
#include "./updateForcedVelocity.h"
#include "./boundaryConditions/applyInitialCondition.h"

void rebuildGrids( std::vector<GridStruct> &grids, const VoxelizerStruct &Voxelizer, const int level )
// Consider grids 0, 1, 2, 3 where 3 is the finest. We want to rebuild grids 2, 3 -> we call this function on 2 (level=2) which recursively calls it on all levels below.
{
	const bool iAmCoarsest = ( level == 0 );
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	GridStruct &Grid = grids[ level ];	
	const bool initPass = ( Grid.fArray.getSizes()[0] < 1 );
	
	InfoStruct &Info = Grid.Info;	
	Info.cellCountOld = Info.cellCount;
	
	Info.updatesSinceRebuild = 0; 
	Info.updatesSinceMovingBouncebackUpdate = 0;
	
	SkeletonGridStruct &SkeletonGrid = Grid.SkeletonGrid;
	InfoStruct &SkeletonInfo = SkeletonGrid.Info;
	
	static GridStruct dummyGrid; // if I am the coarsest grid myself, here Im fooling C++ to think there is a coarser grid than me, muhehe
    GridStruct &GridCoarse = iAmCoarsest ? dummyGrid : grids[ level - 1 ];
	InfoStruct &InfoCoarse = GridCoarse.Info;
	
	// 0) sum fArray[ 27, all ] which tracks torque, add it to the cumulative tracker
	auto fView  = Grid.fArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int cell ) { return fView( 27, cell ); };
	auto reduction = [] __cuda_callable__( const float& a, const float& b ) { return a + b; };
	float TzSum = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Info.cellCountOld, fetch, reduction, 0.f );
	Info.torqueReportCumulative += ( TzSum / 1000.f ); // converting from Nmm to Nm
	
	// 1) Pull fArray into the correct cells to be able to forget NBR
	if ( !initPass ) pullFArrayIntoCells( Grid );
	// 2) Mark refinement area
	if ( iAmCoarsest )
	{
		markKeepCells( SkeletonGrid, Voxelizer );
		SkeletonInfo.refinementCount = countOnesInBoolArray( SkeletonGrid.keepCellMarkerArray, SkeletonInfo.cellCount );
		Info.cellCountFull = 8 * countOnesInBoolArray( SkeletonGrid.keepCellMarkerArray, SkeletonInfo.cellCount );
	}
	else
	{
		markRefinementCells( GridCoarse, Voxelizer, GridCoarse.Info.cellCount );
		InfoCoarse.deepRefinementCount = countOnesInBoolArray( GridCoarse.deepRefinementMarkerArray, InfoCoarse.cellCount );
		InfoCoarse.refinementCount = countOnesInBoolArray( GridCoarse.refinementMarkerArray, InfoCoarse.cellCount );
		InfoCoarse.fineToCoarseCount = countOnesInBoolArray( GridCoarse.fineToCoarseMarkerArray, InfoCoarse.cellCount );
		InfoCoarse.coarseToFineCount = InfoCoarse.refinementCount - InfoCoarse.deepRefinementCount - InfoCoarse.fineToCoarseCount;
		Info.cellCountFull = 8 * InfoCoarse.refinementCount;
	}
	
	if ( initPass )
	{
		Info.memoryCountFull = Info.cellCountFull + ( ( Info.cellCountFull * MEMORY_RESERVE_PERCENTAGE ) / 100 );
		Grid.IJK.iArray.setSize( Info.memoryCountFull );
		Grid.IJK.jArray.setSize( Info.memoryCountFull );
		Grid.IJK.kArray.setSize( Info.memoryCountFull );
		Grid.NBR.jPlusArray.setSize( Info.memoryCountFull );
		Grid.NBR.kPlusArray.setSize( Info.memoryCountFull );
		Grid.bitPackedMarkerArray.setSize( Info.memoryCountFull );
		Grid.NBR.jMinusArray.setSize( Info.memoryCountFull );
		Grid.NBR.kMinusArray.setSize( Info.memoryCountFull );
		Grid.NBR.isGeometricBitPackedMarkerArray.setSize( Info.memoryCountFull );
		Grid.parentMapArray.setSize( Info.memoryCountFull );
		Grid.keepCellMarkerArray.setSize( Info.memoryCountFull );	
		Grid.movingBouncebackMarkerArray.setSize( Info.memoryCountFull );
		Grid.forcedVelocityMarkerArray.setSize( Info.memoryCountFull );
		Grid.changedStateMarkerArray.setSize( Info.memoryCountFull ); Grid.changedStateMarkerArray.setValue( false );
		Grid.markerBuffer.setSize( Info.memoryCountFull );
		Info.mbbUpdateMemoryCount = ( ( Info.cellCountFull * MEMORY_MBB_UPDATE_PERCENTAGE ) / 100 );
		Grid.newlyFluidIndexArray.setSize( Info.mbbUpdateMemoryCount );
		Grid.newlyMBBIndexArray.setSize( Info.mbbUpdateMemoryCount );
		Grid.fBufferArray.setSizes( 27, Info.mbbUpdateMemoryCount );
		Info.gridMemoryBytes += (long long)(9 * 4 + 5 * 1 + 1 * 1) * (long long)(Info.memoryCountFull); // 9 int arrays, 5 bool arrays, 1 uint8_t
		Info.gridMemoryBytes += (long long)(2 * 4 + 27 * 4) * (long long)(Info.mbbUpdateMemoryCount); // 2 int arrays, 27 float arrays
		if ( iAmFinest )
		{
			Grid.bouncebackMarkerArray.setSize( Info.memoryCountFull );
			Info.gridMemoryBytes += (1 * 1) * (Info.memoryCountFull); // 1 bool array
		}
		else
		{
			Grid.childMapArray.setSize( Info.memoryCountFull );
			Grid.refinementMarkerArray.setSize( Info.memoryCountFull );
			Grid.deepRefinementMarkerArray.setSize( Info.memoryCountFull );
			Grid.fineToCoarseMarkerArray.setSize( Info.memoryCountFull );
			Grid.coarseToFineMarkerArray.setSize( Info.memoryCountFull );
			Info.gridMemoryBytes += (long long)(1 * 4 + 4 * 1) * (long long)(Info.memoryCountFull); // 1 int array, 4 bool arrays
		}
	}
	else if ( Info.cellCountFull > Info.memoryCountFull )
	{
		std::cout << "rebuildGrid failed on level " << level << ", memoryCountFull = " << Info.memoryCountFull << ", cellCountFull = " << Info.cellCountFull << std::endl;
		throw std::runtime_error("rebuildGrid failed, cellCountFull exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE in your main file.");
	}
	// 3) Build our grid (we are the "finer grid" with respect to the grid we are taking spatial information from)
	
	if ( iAmCoarsest ) buildFinerGrid( SkeletonGrid, Grid );
	else buildFinerGrid( GridCoarse, Grid );
	IntArrayType &oldToFullArray = Grid.intBuffer1; // We cannot touch intBuffer1 now!
	// 4) Get rid of the cells that are deep inside solid. Only keep the necessary ones, mark them in keepCellMarkerArray
	markKeepCells( Grid, Voxelizer, Info.cellCountFull );
	Info.cellCount = countOnesInBoolArray( Grid.keepCellMarkerArray, Info.cellCountFull );
	
	if ( initPass )
	{
		Info.memoryCount = Info.cellCount + ( ( Info.cellCount * MEMORY_RESERVE_PERCENTAGE ) / 100 );
		Grid.fArray.setSizes( 28, Info.memoryCount );
		Info.gridMemoryBytes += (long long)(4 * 28) * (long long)(Info.memoryCount); // 28 float arrays
	}
	else if ( Info.cellCount > Info.memoryCount )
	{
		std::cout << "rebuildGrid failed on level " << level << ", memoryCount = " << Info.memoryCount << ", cellCount = " << Info.cellCount << std::endl;
		throw std::runtime_error("rebuildGrid failed, cellCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE in your main file.");
	}
	
	// 5) Now we know which cells to keep, skip the unmarked ones in main NBR arrays
	int jPlus, kPlus;
	jPlus = 1; kPlus = 0;
	skipUnmarkedNBRArray( Grid.NBR.jPlusArray, Grid.keepCellMarkerArray, jPlus, kPlus, Grid, Info.cellCountFull ); 
	jPlus = 0; kPlus = 1;
	skipUnmarkedNBRArray( Grid.NBR.kPlusArray, Grid.keepCellMarkerArray, jPlus, kPlus, Grid, Info.cellCountFull ); 
	
	// 6) We already have oldToFullArray from step 3, now let's build fullToKeepArray map
	IntArrayType &fullToKeepArray = Grid.intBuffer2;
	intArrayFromBoolArray( fullToKeepArray, Grid.keepCellMarkerArray, Info.cellCountFull );
	TNL::Algorithms::inplaceExclusiveScan( fullToKeepArray, 0, Info.cellCountFull, TNL::Plus{} );

	// 7) IJK, NBR full to keep transformation
	fullToKeepTransform( Grid.IJK.iArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.IJK.jArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.IJK.kArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransformWithIndexRepair( Grid.NBR.jPlusArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransformWithIndexRepair( Grid.NBR.kPlusArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );
	fullToKeepTransform( Grid.parentMapArray, Grid.keepCellMarkerArray, fullToKeepArray, Grid.intBuffer3, Info.cellCountFull );

	// 8) transform childMapArray of the coarser grid
	if ( !iAmCoarsest )
	{
		auto childMapView = GridCoarse.childMapArray.getView();
		auto keepCellMarkerView = Grid.keepCellMarkerArray.getConstView();
		auto fullToKeepView = fullToKeepArray.getConstView();
		auto cellLambda = [=] __cuda_callable__ ( const int cellCoarse ) mutable
		{	
			const int cellFine = childMapView[ cellCoarse ];
			if ( cellFine < 0 ) return;
			if ( keepCellMarkerView[ cellFine ] )
			{
				const int newIndex = fullToKeepView[ cellFine ];
				childMapView[ cellCoarse ] = newIndex;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.Info.cellCount, cellLambda );
	}
	
	if ( !initPass )
	{
		// 9) build oldToKeepArray map
		IntArrayType &oldToKeepArray = Grid.intBuffer3;
		auto oldToFullView = oldToFullArray.getConstView(); // we have been holding this since step 3
		auto fullToKeepView = fullToKeepArray.getConstView();
		auto oldToKeepView = oldToKeepArray.getView();
		auto cellLambda = [=] __cuda_callable__ ( const int cellOld ) mutable
		{	
			const int oldToFullIndex = oldToFullView[ cellOld ];
			if ( oldToFullIndex > 0 ) oldToKeepView[ cellOld ] = fullToKeepView[ oldToFullIndex ];
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCountOld, cellLambda );

		// 10) fArray old to keep transformation
		oldToKeepTransform( Grid );
		
		// 11) we changed our indexes, so we want to repair parentMapArray of the next finer grid
		if ( !iAmFinest )
		{
			auto parentMapView = grids[level+1].parentMapArray.getView();
			auto oldToKeepView = oldToKeepArray.getView();
			auto cellLambda = [=] __cuda_callable__ ( const int cellFine ) mutable
			{	
				const int cellCoarseOld = parentMapView[ cellFine ];
				const int cellCoarseNew = oldToKeepView[ cellCoarseOld ];
				parentMapView[ cellFine ] = cellCoarseNew;
			};
			TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, grids[level+1].Info.cellCount, cellLambda );
		}
	}
	// 12) build interface lists of the coarse grid
	if ( !iAmCoarsest )
	{
		// fine to coarse
		GridCoarse.Info.fineToCoarseCount = countOnesInBoolArray( GridCoarse.fineToCoarseMarkerArray, GridCoarse.Info.cellCount );
		if ( initPass )
		{
			InfoCoarse.fineToCoarseMemoryCount = GridCoarse.Info.fineToCoarseCount + ( ( GridCoarse.Info.fineToCoarseCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			GridCoarse.fineToCoarseIndexArray.setSize( InfoCoarse.fineToCoarseMemoryCount );
			InfoCoarse.gridMemoryBytes += (long long)(4) * (long long)(InfoCoarse.fineToCoarseMemoryCount); // 1 int array
		}
		else if ( GridCoarse.Info.fineToCoarseCount > GridCoarse.Info.fineToCoarseMemoryCount )
		{
			std::cout 	<< "rebuildGrid failed on level " << level << ", fineToCoarseMemoryCount = " << GridCoarse.Info.fineToCoarseMemoryCount 
						<< ", fineToCoarseCount = " << GridCoarse.Info.fineToCoarseCount << std::endl;
			throw std::runtime_error("rebuildGrid failed, fineToCoarseCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE_INTERFACE in your main file.");
		}
		intArrayFromBoolArray( GridCoarse.intBuffer1, GridCoarse.fineToCoarseMarkerArray, GridCoarse.Info.cellCount );
		TNL::Algorithms::inplaceExclusiveScan( GridCoarse.intBuffer1, 0, GridCoarse.Info.cellCount, TNL::Plus{} );
		auto fineToCoarseMarkerView = GridCoarse.fineToCoarseMarkerArray.getConstView();
		auto intBuffer1View = GridCoarse.intBuffer1.getView();
		auto fineToCoarseIndexView = GridCoarse.fineToCoarseIndexArray.getView();
		auto cellLambdaFineToCoarse = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( fineToCoarseMarkerView[ cell ] )
			{
				const int index = intBuffer1View[ cell ];
				fineToCoarseIndexView[ index ] = cell;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.Info.cellCount, cellLambdaFineToCoarse );
		
		// coarse to fine
		GridCoarse.Info.coarseToFineCount = countOnesInBoolArray( GridCoarse.coarseToFineMarkerArray, GridCoarse.Info.cellCount );
		if ( initPass )
		{
			InfoCoarse.coarseToFineMemoryCount = GridCoarse.Info.coarseToFineCount + ( ( GridCoarse.Info.coarseToFineCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			GridCoarse.coarseToFineIndexArray.setSize( InfoCoarse.coarseToFineMemoryCount );
			InfoCoarse.gridMemoryBytes += (long long)(4) * (long long)(InfoCoarse.coarseToFineMemoryCount); // 1 int array
		}
		else if ( GridCoarse.Info.coarseToFineCount > GridCoarse.Info.coarseToFineMemoryCount )
		{
			std::cout 	<< "rebuildGrid failed on level " << level << ", coarseToFineMemoryCount = " << GridCoarse.Info.coarseToFineMemoryCount 
						<< ", coarseToFineCount = " << GridCoarse.Info.coarseToFineCount << std::endl;
			throw std::runtime_error("rebuildGrid failed, coarseToFineCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE_INTERFACE in your main file.");
		}
		intArrayFromBoolArray( GridCoarse.intBuffer1, GridCoarse.coarseToFineMarkerArray, GridCoarse.Info.cellCount );
		TNL::Algorithms::inplaceExclusiveScan( GridCoarse.intBuffer1, 0, GridCoarse.Info.cellCount, TNL::Plus{} );
		auto coarseToFineMarkerView = GridCoarse.coarseToFineMarkerArray.getConstView();
		// auto intBuffer1View = GridCoarse.intBuffer1.getView(); // already declared
		auto coarseToFineIndexView = GridCoarse.coarseToFineIndexArray.getView();
		auto cellLambdaCoarseToFine = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( coarseToFineMarkerView[ cell ] )
			{
				const int index = intBuffer1View[ cell ];
				coarseToFineIndexView[ index ] = cell;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.Info.cellCount, cellLambdaCoarseToFine );
	}
	
	// 13) build non reflective outlet cell list for the finest grid
	if ( iAmFinest )
	{
		applyNonReflectiveOutletMarker( Grid.markerBuffer, Grid, Info.cellCount );
		Info.nonReflectiveOutletCount = countOnesInBoolArray( Grid.markerBuffer, Info.cellCount );
		if ( initPass )
		{
			Info.nonReflectiveOutletMemoryCount = Info.nonReflectiveOutletCount + ( ( Info.nonReflectiveOutletCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			Grid.nonReflectiveOutletIndexArray.setSize( Info.nonReflectiveOutletMemoryCount );
			Info.gridMemoryBytes += (long long)(4) * (long long)(Info.nonReflectiveOutletMemoryCount); // 1 int array
		}
		else if ( Info.nonReflectiveOutletCount > Info.nonReflectiveOutletMemoryCount )
		{
			std::cout 	<< "rebuildGrid failed on level " << level << ", nonReflectiveOutletMemoryCount = " << Info.nonReflectiveOutletMemoryCount 
						<< ", nonReflectiveOutletCount = " << GridCoarse.Info.nonReflectiveOutletCount << std::endl;
			throw std::runtime_error("rebuildGrid failed, nonReflectiveOutletCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE_INTERFACE in your main file.");
		}
		intArrayFromBoolArray( Grid.intBuffer1, Grid.markerBuffer, Grid.Info.cellCount );
		TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer1, 0, Grid.Info.cellCount, TNL::Plus{} );
		auto nonReflectiveOutletMarkerView = Grid.markerBuffer.getConstView();
		auto intBuffer1View = Grid.intBuffer1.getConstView();
		auto nonReflectiveOutletIndexView = Grid.nonReflectiveOutletIndexArray.getView();
		auto cellLambdaNonReflectiveOutlet = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( nonReflectiveOutletMarkerView[ cell ] )
			{
				const int index = intBuffer1View[ cell ];
				nonReflectiveOutletIndexView[ index ] = cell;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Info.cellCount, cellLambdaNonReflectiveOutlet );
	}
	
	// 13) build non reflective inlet cell list for the finest grid
	if ( iAmFinest )
	{
		applyNonReflectiveInletMarker( Grid.markerBuffer, Grid, Info.cellCount );
		Info.nonReflectiveInletCount = countOnesInBoolArray( Grid.markerBuffer, Info.cellCount );
		if ( initPass )
		{
			Info.nonReflectiveInletMemoryCount = Info.nonReflectiveInletCount + ( ( Info.nonReflectiveInletCount * MEMORY_RESERVE_PERCENTAGE_INTERFACE ) / 100 );
			Grid.nonReflectiveInletIndexArray.setSize( Info.nonReflectiveInletMemoryCount );
			Info.gridMemoryBytes += (long long)(4) * (long long)(Info.nonReflectiveInletMemoryCount); // 1 int array
		}
		else if ( Info.nonReflectiveInletCount > Info.nonReflectiveInletMemoryCount )
		{
			std::cout 	<< "rebuildGrid failed on level " << level << ", nonReflectiveInletMemoryCount = " << Info.nonReflectiveInletMemoryCount 
						<< ", nonReflectiveInletCount = " << GridCoarse.Info.nonReflectiveInletCount << std::endl;
			throw std::runtime_error("rebuildGrid failed, nonReflectiveInletCount exceeded allocated memory. Try increasing MEMORY_RESERVE_PERCENTAGE_INTERFACE in your main file.");
		}
		intArrayFromBoolArray( Grid.intBuffer1, Grid.markerBuffer, Grid.Info.cellCount );
		TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer1, 0, Grid.Info.cellCount, TNL::Plus{} );
		auto nonReflectiveInletMarkerView = Grid.markerBuffer.getConstView();
		auto intBuffer1View = Grid.intBuffer1.getConstView();
		auto nonReflectiveInletIndexView = Grid.nonReflectiveInletIndexArray.getView();
		auto cellLambdaNonReflectiveInlet = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			if ( nonReflectiveInletMarkerView[ cell ] )
			{
				const int index = intBuffer1View[ cell ];
				nonReflectiveInletIndexView[ index ] = cell;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.Info.cellCount, cellLambdaNonReflectiveInlet );
	}
	
	// 13) fill jkPlus and NBR minus
	
	//auto jkPlusView = Grid.NBR.jkPlusArray.getView();
	//auto NBRPlusFinishLambda = [=] __cuda_callable__ ( const int cell ) mutable
	//{	
	//	jkPlusView[ cell ] = jPlusView[ kPlusView[ cell ] ];
	//};
	//TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, NBRPlusFinishLambda );
	
	// 14) mark NBR geometric validity
	markGeometricNBRPlus( Grid, Info.cellCount );
	
	// 15) if we are the finest grid, mark moving bounceback and bounceback
	if ( iAmFinest )
	{
		applyMarkersFromRayMap( Grid.bouncebackMarkerArray, Voxelizer.rayMapBounceback, Grid, Grid.Info.cellCount );
		applyMarkersFromRayMap( Grid.movingBouncebackMarkerArray, Voxelizer.rayMapMovingBounceback, Grid, Grid.Info.cellCount );
	}
	
	// 20) build list of interpolated BB cells
	int interpolatedBBCount = countOnesInBoolArray( Grid.bouncebackMarkerArray, Info.cellCount );
	interpolatedBBCount += countOnesInBoolArray( Grid.movingBouncebackMarkerArray, Info.cellCount );
	Grid.interpolatedBBCellList.setSize( interpolatedBBCount );
	Grid.interpolatedBBLinkLengths.setSizes( 27, interpolatedBBCount );
	Grid.interpolatedBBLinkLengths.setValue( 0.5f );
	Info.gridMemoryBytes += (long long)(4 * 1 + 26 * 4 * 1 ) * (long long)(interpolatedBBCount); // 1 int array, 26 float arrays
	
	Grid.markerBuffer.setValue( false );
	spreadMarkers( Grid.markerBuffer, Grid.bouncebackMarkerArray, Grid, Info.cellCount );
	Grid.markerBuffer = Grid.markerBuffer - Grid.bouncebackMarkerArray;
	intArrayFromBoolArray( Grid.intBuffer3, Grid.markerBuffer, Info.cellCount );
	TNL::Algorithms::inplaceExclusiveScan( Grid.intBuffer3, 0, Info.cellCount, TNL::Plus{} );
	auto someMarkerView = Grid.markerBuffer.getConstView();
	auto indexBBlistView = Grid.interpolatedBBCellList.getView();
	auto intBuffer3View = Grid.intBuffer3.getConstView();
	auto cellLambdaBleble = [=] __cuda_callable__ ( const int cell ) mutable
	{	
		if ( someMarkerView( cell ) )
		{
			indexBBlistView( intBuffer3View( cell ) ) = cell;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambdaBleble );
	
	
	// 15) we apply initial condition for fArray on the coarser grid, because only now we have marked it correctly
	// if we are finest we can apply initial condition on ourselves too
	// at the same time, report memory size
	if ( initPass ) 
	{
		if ( !iAmCoarsest ) 
		{
			applyInitialCondition( grids[ level-1 ] );
			std::cout << "Grid level " << level-1 << " allocated on GPU, it takes " << grids[level-1].Info.gridMemoryBytes / 1048576.0 << " MiB" << std::endl;
		}
		if ( iAmFinest ) 
		{
			applyInitialCondition( Grid );
			std::cout << "Grid level " << level << " allocated on GPU, it takes " << Grid.Info.gridMemoryBytes / 1048576.0 << " MiB" << std::endl;
		}
	}
	
	// 16) recursion
	if ( !iAmFinest ) rebuildGrids( grids, Voxelizer, level+1 );
		
	// 17) Repair NBR minus of the coarser grid and fill its bitPackedMarker
	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	if ( !iAmCoarsest ) // && grids[level - 1].Info.updatesSinceRebuild != 0 )
	{
		auto jPlusViewCoarse = grids[level - 1].NBR.jPlusArray.getConstView();
		auto kPlusViewCoarse = grids[level - 1].NBR.kPlusArray.getConstView();
		auto jMinusViewCoarse = grids[level - 1].NBR.jMinusArray.getView();
		auto kMinusViewCoarse = grids[level - 1].NBR.kMinusArray.getView();
		auto NBRCoarseRepairLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			jMinusViewCoarse[ jPlusViewCoarse[ cell ] ] = cell;
			kMinusViewCoarse[ kPlusViewCoarse[ cell ] ] = cell;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, grids[level - 1].Info.cellCount, NBRCoarseRepairLambda );
		updateForcedVelocity( grids[level - 1], Voxelizer );
		// fill bit packed marker
		fillBitPackedMarkerArray( grids[level - 1], grids[level - 1].Info.cellCount );
	}
	
	// 18) if we are the finest, we need to fill NBR minus ourselves and fill our bitPackedMarker
	if ( iAmFinest )
	{
		auto jMinusView = Grid.NBR.jMinusArray.getView();
		auto kMinusView = Grid.NBR.kMinusArray.getView();
		auto NBRMinusFinishLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{	
			jMinusView[ jPlusView[ cell ] ] = cell;
			kMinusView[ kPlusView[ cell ] ] = cell;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, NBRMinusFinishLambda );
		updateForcedVelocity( Grid, Voxelizer );
		// fill bit packed marker
		fillBitPackedMarkerArray( Grid, Grid.Info.cellCount );
	}
	
	// 19) update interface with the coarser grid
	if ( !iAmCoarsest && !initPass )
	{
		updateInterface( grids[level - 1], Grid );
	}
	
}
*/
