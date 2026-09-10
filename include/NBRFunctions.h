#pragma once

#include "./types.h"

__host__ __device__ void finishNBRPlus( NBRStruct &NBR, const InfoStruct &Info )
{
	NBR.iPlus = NBR.self + 1; if ( NBR.iPlus >= Info.cellCount ) NBR.iPlus = 0;		
	NBR.ijPlus = NBR.jPlus + 1; if ( NBR.ijPlus >= Info.cellCount ) NBR.ijPlus = 0;
	NBR.ikPlus = NBR.kPlus + 1; if ( NBR.ikPlus >= Info.cellCount ) NBR.ikPlus = 0;
	NBR.ijkPlus = NBR.jkPlus + 1; if ( NBR.ijkPlus >= Info.cellCount ) NBR.ijkPlus = 0;
}

__host__ __device__ void finishNBRAll( NBRStruct &NBR, const InfoStruct &Info )
{
	NBR.iPlus = NBR.self + 1; if ( NBR.iPlus >= Info.cellCount ) NBR.iPlus = 0;		
	NBR.ijPlus = NBR.jPlus + 1; if ( NBR.ijPlus >= Info.cellCount ) NBR.ijPlus = 0;
	NBR.ikPlus = NBR.kPlus + 1; if ( NBR.ikPlus >= Info.cellCount ) NBR.ikPlus = 0;
	NBR.ijkPlus = NBR.jkPlus + 1; if ( NBR.ijkPlus >= Info.cellCount ) NBR.ijkPlus = 0;
	NBR.iMinus = NBR.self - 1; if ( NBR.iMinus < 0 ) NBR.iMinus = Info.cellCount-1;		
}

// this is used to bit unpack the information from GridBuilder.NBR.isGeometricBitPackedMarkerArray
__host__ __device__ inline void byteToBools( const uint8_t &value, bool (&bools)[8] )
{
    for (int i = 0; i < 8; ++i)
    {
        bools[i] = ((value >> i) & uint8_t{1}) != 0;
    }
}

// this is used to bit pack the information for GridBuilder.NBR.isGeometricBitPackedMarkerArray
__host__ __device__ inline void boolsToByte( uint8_t& value, const bool (&bools)[8] )
{
    value = 0;
    for (int i = 0; i < 8; i++ )
    {
        if (bools[i])
        {
            value |= static_cast<uint8_t>(uint8_t{1} << i);
        }
    }
}

void connectNBRInRows( IntArrayType &NBRArray, IntArrayType &rowMapArray, const IntArrayType &rowCounterScanArray )
{
	const int totalRows = rowCounterScanArray.getSize() - 1;
	auto NBRView = NBRArray.getView();
	auto rowMapView = rowMapArray.getView();
	auto rowCounterScanView = rowCounterScanArray.getConstView();
	auto rowLambda = [=] __cuda_callable__ ( const int rowIndex ) mutable
	{
		const int startingPoint = rowCounterScanView( rowIndex );
		const int rowCount = rowCounterScanView( rowIndex + 1 ) - startingPoint;
		if (rowCount == 0) return;
		// sort the row in ascending order
		for ( int layer = 1; layer < rowCount; layer++ ) 
		{
			int key = rowMapView[startingPoint + layer];
			int slider = layer - 1;
			while ( slider >= 0 && rowMapView[startingPoint + slider] > key ) 
			{
				rowMapView[startingPoint + slider + 1] = rowMapView[startingPoint + slider];
				slider = slider - 1;
			}
			rowMapView[startingPoint + slider + 1] = key;
		}
		// connect the neighbours
		for ( int layer = 0; layer < rowCount - 1; layer++ ) 
		{
			const int owner = rowMapView[startingPoint + layer];
			const int neighbour = rowMapView[startingPoint + layer + 1];
			NBRView( owner ) = neighbour;
		}
		// to the last owner, assign the first neighbour (periodic wrap)
		const int lastOwner = rowMapView[startingPoint + rowCount - 1];
		const int firstNeighbour = rowMapView[startingPoint];
		NBRView( lastOwner ) = firstNeighbour;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, totalRows, rowLambda );	
}

void buildNBRPlus( GridBuilderStruct &GridBuilder )
{
	InfoStruct &Info = GridBuilder.Info;
	NBRArrayStruct &NBR = GridBuilder.NBR;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	
	IntArrayType rowMapArray( Info.cellCount );
	auto rowMapView = rowMapArray.getView();
	IntArrayType rowCounterScanArray;
	IntArrayType rowCounterTempArray;
	
	// build map of IK rows
	rowCounterScanArray.setSize( Info.cellCountX * Info.cellCountZ + 1 );
	rowCounterTempArray.setSize( Info.cellCountX * Info.cellCountZ + 1 );
	rowCounterScanArray.setValue( 0 );
	rowCounterTempArray.setValue( 0 );
	auto ikRowCounterScanView = rowCounterScanArray.getView();
	auto ikRowCounterTempView = rowCounterTempArray.getView();
	auto ikCounterLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int kCell = kView( cell );
		const int rowIndex = kCell * Info.cellCountX + iCell;
		TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(ikRowCounterScanView( rowIndex ), 1);
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, ikCounterLambda );	
	TNL::Algorithms::inplaceExclusiveScan( rowCounterScanArray, 0, Info.cellCountX * Info.cellCountZ + 1, TNL::Plus{} );
	auto ikWriteLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int kCell = kView( cell );
		const int rowIndex = kCell * Info.cellCountX + iCell;
		int writeIndex = ikRowCounterScanView( rowIndex );
		writeIndex += TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(ikRowCounterTempView( rowIndex ), 1);
		rowMapView( writeIndex ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, ikWriteLambda );
	// build jPlus
	connectNBRInRows( NBR.jPlusArray, rowMapArray, rowCounterScanArray );
	
	// build map of IJ rows
	rowCounterScanArray.setSize( Info.cellCountX * Info.cellCountY + 1 );
	rowCounterTempArray.setSize( Info.cellCountX * Info.cellCountY + 1 );
	rowCounterScanArray.setValue( 0 );
	rowCounterTempArray.setValue( 0 );
	auto ijRowCounterScanView = rowCounterScanArray.getView();
	auto ijRowCounterTempView = rowCounterTempArray.getView();
	auto ijCounterLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int rowIndex = jCell * Info.cellCountX + iCell;
		TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(ijRowCounterScanView( rowIndex ), 1);
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, ijCounterLambda );	
	TNL::Algorithms::inplaceExclusiveScan( rowCounterScanArray, 0, Info.cellCountX * Info.cellCountY + 1, TNL::Plus{} );
	auto ijWriteLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int rowIndex = jCell * Info.cellCountX + iCell;
		int writeIndex = ijRowCounterScanView( rowIndex );
		writeIndex += TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(ijRowCounterTempView( rowIndex ), 1);
		rowMapView( writeIndex ) = cell;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, ijWriteLambda );
	// build kPlus
	connectNBRInRows( NBR.kPlusArray, rowMapArray, rowCounterScanArray );
}

void markGeometricNBRPlus( GridBuilderStruct &GridBuilder )
{
	const int &cellCount = GridBuilder.Info.cellCount;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	auto jPlusView = GridBuilder.NBR.jPlusArray.getConstView();
	auto kPlusView = GridBuilder.NBR.kPlusArray.getConstView();
	auto isGeometricBitPackedMarkerView = GridBuilder.NBR.isGeometricBitPackedMarkerArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		
		const int iPlus = (cell + 1 < cellCount) ? cell + 1 : 0;
		const int jPlus = jPlusView[ cell ];
		const int ijPlus = (jPlus + 1 < cellCount) ? jPlus + 1 : 0;
		const int kPlus = kPlusView[ cell ];
		const int ikPlus = (kPlus + 1 < cellCount) ? kPlus + 1 : 0;
		const int jkPlus = jPlusView[ kPlus ];		
		const int ijkPlus = (jkPlus + 1 < cellCount) ? jkPlus + 1 : 0;
		
		bool isGeometricMarker[8] = {false};
		
		isGeometricMarker[0] = ( iView[iPlus]==iCell+1 && jView[iPlus]==jCell && kView[iPlus]==kCell );
		isGeometricMarker[1] = ( iView[jPlus]==iCell && jView[jPlus]==jCell+1 && kView[jPlus]==kCell );
		isGeometricMarker[2] = ( iView[ijPlus]==iCell+1 && jView[ijPlus]==jCell+1 && kView[ijPlus]==kCell );
		isGeometricMarker[3] = ( iView[kPlus]==iCell && jView[kPlus]==jCell && kView[kPlus]==kCell+1 );
		isGeometricMarker[4] = ( iView[ikPlus]==iCell+1 && jView[ikPlus]==jCell && kView[ikPlus]==kCell+1 );
		isGeometricMarker[5] = ( iView[jkPlus]==iCell && jView[jkPlus]==jCell+1 && kView[jkPlus]==kCell+1 );
		isGeometricMarker[6] = ( iView[ijkPlus]==iCell+1 && jView[ijkPlus]==jCell+1 && kView[ijkPlus]==kCell+1 );
		
		uint8_t isGeometricBitPack;
		boolsToByte( isGeometricBitPack, isGeometricMarker );
		
		isGeometricBitPackedMarkerView( cell ) = isGeometricBitPack;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, cellCount, cellLambda );	
}
