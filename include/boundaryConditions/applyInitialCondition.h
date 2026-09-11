#pragma once

#include "../esotwistStreamingFunctions.h"
#include "../cellFunctions.h"
#include "../NBRFunctions.h"
#include "./interpolatedBouncebackFunctions.h"

void applyInitialCondition( GridStruct &Grid )
{
	const InfoStruct &Info = Grid.Info;
	
	auto fView  = Grid.fArray.getView();
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	auto shifterView = Grid.IJKNBR.shifterArray.getConstView();	
	auto iView = Grid.IJKNBR.iArray.getConstView();
	auto jView = Grid.IJKNBR.jArray.getConstView();
	auto kView = Grid.IJKNBR.kArray.getConstView();
	auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
	auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
	auto wallMapView = Grid.Wall.wallMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		int iCell, jCell, kCell;
		NBRStruct NBR;
		getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
							shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
							Info );		
		BCStruct BC;
		getInitialCondition( BC, iCell, jCell, kCell, Info ); 
		
		float f[27];
		getFeq( BC.rho, BC.ux, BC.uy, BC.uz, f );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	
	// Now fill the initial values as previous values for open BC
	for ( int openBCID = 0; openBCID < (int)Grid.openBCs.size(); openBCID++ )
	{
		OpenBCArrayStruct &OpenBC = Grid.openBCs[ openBCID ];
		auto indexView = OpenBC.indexArray.getConstView();
		auto rhoPrevView = OpenBC.rhoPrevArray.getView();
		auto uNormalPrevView = OpenBC.uNormalPrevArray.getView();
		// loop over open boundary cells
		auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
		{
			const int cell = indexView( index );
			
			// fill iCell, jCell, kCell and NBR
			int iCell, jCell, kCell;
			NBRStruct NBR;
			getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
								shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
								Info );
			
			// identify outer normal
			int outerNormalX, outerNormalY, outerNormalZ;
			getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
			
			// read all f
			float f[27];
			int cellIndex[27];
			int fIndex[27];
			getPreCollisionIndex( cellIndex, fIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fIndex[direction], cellIndex[direction]);
			
			// get rho, ux, uy, uz
			float rho, ux, uy, uz;
			getRhoUxUyUz( rho, ux, uy, uz, f );
			
			// fill those as previous values
			const float uNormal = (float)outerNormalX * ux + (float)outerNormalY * uy + (float)outerNormalZ * uz;
			rhoPrevView( index ) = rho;
			uNormalPrevView( index ) = uNormal;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, OpenBC.openBCCount, cellLambda );
	}
	
}
