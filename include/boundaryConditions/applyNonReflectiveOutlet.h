#pragma once

// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cx is negative for: { 2, 8, 10, 11, 15, 19, 21, 24, 25 };
// cx is positive for: { 1, 7, 9, 12, 16, 20, 22, 23, 26 };
// cy is negative for: { 5, 11, 14, 16, 18, 20, 21, 23, 25 };
// cy is positive for: { 6, 12, 13, 15, 17, 19, 22, 24, 26 };
// cz is negative for: { 3, 7, 10, 13, 18, 19, 22, 23, 25 };
// cz is positive for: { 4, 8, 9, 14, 17, 20, 21, 24, 26};

// Modified version of Geier's non reflective outlet
// There was an issue with the original outlet version that as soon as backflow happened anywhere in the area,
// it crashed. I tried to fix this by smoothly switching to hard pressure MBBC, but then
// I had to manually choose the velocity threshold and even then there was still a risk
// of creating artifacts in places where backflow happened.
// Now, what I can do is this:
// Run a reduction on the non reflective outlet cells and calculate their average pressure
// There is no need to write the resulting f, just save the pressure to Info.nonReflectiveOutletRho
// Then apply a completely normal and reliable MBBC but use some interpolated
// value between the non reflective average pressure and the target pressure

void applyNonReflectiveGeierOutlet( GridStruct &Grid )
{
	InfoStruct &Info = Grid.Info;
	if ( Info.nonReflectiveOutletCount == 0 ) return;
	
	auto nonReflectiveOutletIndexView = Grid.nonReflectiveOutletIndexArray.getConstView();
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	auto jMinusView = Grid.NBR.jMinusArray.getConstView();
	auto kMinusView = Grid.NBR.kMinusArray.getConstView();
	
	auto fetch = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = nonReflectiveOutletIndexView( index );
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		int outerNormalX, outerNormalY, outerNormalZ;
		getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		int upstreamCell;
		const int* directionList = nullptr;
		static const int dir_xp[] = { 2, 8, 10, 11, 15, 19, 21, 24, 25 };
		static const int dir_xm[] = { 1, 7, 9, 12, 16, 20, 22, 23, 26 };
		static const int dir_yp[] = { 5, 11, 14, 16, 18, 20, 21, 23, 25 };
		static const int dir_ym[] = { 6, 12, 13, 15, 17, 19, 22, 24, 26 };
		static const int dir_zp[] = { 3, 7, 10, 13, 18, 19, 22, 23, 25 };
		static const int dir_zm[] = { 4, 8, 9, 14, 17, 20, 21, 24, 26 };
		if ( outerNormalX > 0 ) { 
			upstreamCell = cell-1; if ( upstreamCell < 0 ) upstreamCell = Info.cellCount-1;
			directionList = dir_xp;
		}
		else if ( outerNormalX < 0 ) { 
			upstreamCell = cell+1; if ( upstreamCell >= Info.cellCount ) upstreamCell = 0;
			directionList = dir_xm;
		}
		else if ( outerNormalY > 0 ) { 
			upstreamCell = jMinusView( cell );
			directionList = dir_yp;
		}
		else if ( outerNormalY < 0 ) { 
			upstreamCell = jPlusView( cell );
			directionList = dir_ym;
		}
		else if ( outerNormalZ > 0 ) { 
			upstreamCell = kMinusView( cell );
			directionList = dir_zp;
		}
		else { 
			upstreamCell = kPlusView( cell );
			directionList = dir_zm;
		}
		
		NBRStruct upstreamCellNBR;
		upstreamCellNBR.self = upstreamCell;
		upstreamCellNBR.jPlus = jPlusView( upstreamCell );
		upstreamCellNBR.kPlus = kPlusView( upstreamCell );
		upstreamCellNBR.jkPlus = jPlusView( upstreamCellNBR.kPlus );
		finishNBRPlus( upstreamCellNBR, Info );
		
		int cellNewReadIndex[27];
		int fNewReadIndex[27];
		getPreCollisionIndex( cellNewReadIndex, fNewReadIndex, NBR, esotwistFlipper, Info );
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		int upstreamCellCellReadIndex[27];
		int upstreamCellfReadIndex[27];
		getPreviousPostCollisionIndex( upstreamCellCellReadIndex, upstreamCellfReadIndex, upstreamCellNBR, esotwistFlipper, Info );
		
		float f[27];
		for (int direction = 0; direction < 27; direction++)
		{
			f[direction] = fView(fNewReadIndex[direction], cellNewReadIndex[direction]);
		}
		for (int i = 0; i < 9; i++)
		{
			const int direction = directionList[i];
			f[direction] = 0.577350269f * fView(upstreamCellfReadIndex[direction], upstreamCellCellReadIndex[direction]) + (1.f - 0.577350269f) * fView(fReadIndex[direction], cellReadIndex[direction]);
		}
		
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, f );

		//int cellWriteIndex[27];
		//int fWriteIndex[27];
		//getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		//for (int i = 0; i < 9; i++)
		//{
		//	const int direction = directionList[i];
		//	fView(fWriteIndex[direction], cellWriteIndex[direction]) = f[direction];
		//}
		
		return ( rho - 1.f );
		
	};
	auto reduction = [] __cuda_callable__( const float& a, const float& b ) { return a + b; };
	
	float rhoAvg = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Info.nonReflectiveOutletCount, fetch, reduction, 0.f );
	rhoAvg /= (float)Info.nonReflectiveOutletCount;
	// Info.nonReflectiveOutletRho = 1.f + rhoAvg;
}


void applyNonReflectiveOutlet( GridStruct &Grid )
{
	applyNonReflectiveGeierOutlet( Grid );
	
	InfoStruct &Info = Grid.Info;
	if ( Info.nonReflectiveOutletCount == 0 ) return;
	
	auto nonReflectiveOutletIndexView = Grid.nonReflectiveOutletIndexArray.getConstView();
	
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	
	auto fetch = [=] __cuda_callable__ ( const int index )
	{
		const int cell = nonReflectiveOutletIndexView( index );
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		
		int outerNormalX, outerNormalY, outerNormalZ;
		getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		int cellPrevIndex[27];
		int fPrevIndex[27];
		getPreviousPostCollisionIndex( cellPrevIndex, fPrevIndex, NBR, esotwistFlipper, Info );
		float fPrev[27];
		for ( int direction = 0; direction < 27; direction++ )	fPrev[direction] = fView(fPrevIndex[direction], cellPrevIndex[direction]);
		float rhoPrev, uxPrev, uyPrev, uzPrev;
		getRhoUxUyUz( rhoPrev, uxPrev, uyPrev, uzPrev, fPrev );
		
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		
		const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
		const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
		const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
		
		MarkerStruct Marker;
		Marker.nonReflectiveInlet = 1;
		BCStruct BC;
		// pass the current state into the boundary condition function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, U
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		BC.ux = uxPrev; BC.uy = uyPrev; BC.uz = uzPrev;
		
		float rhoZ = 1.f;
		for (int direction = 0; direction < 27; direction++)
		{
			const int cx = cxArray[direction]; const int cy = cyArray[direction]; const int cz = czArray[direction];
			if ( outerNormalX != 0 )
			{
				if ( cx == 0 ) rhoZ += fView(fReadIndex[direction], cellReadIndex[direction]);
				else if ( cx * outerNormalX > 0 ) rhoZ += 2.f * fView(fReadIndex[direction], cellReadIndex[direction]);
			}
			else if ( outerNormalY != 0 )
			{
				if ( cy == 0 ) rhoZ += fView(fReadIndex[direction], cellReadIndex[direction]);
				else if ( cy * outerNormalY > 0 ) rhoZ += 2.f * fView(fReadIndex[direction], cellReadIndex[direction]);
			}
			else
			{
				if ( cz == 0 ) rhoZ += fView(fReadIndex[direction], cellReadIndex[direction]);
				else if ( cz * outerNormalZ > 0 ) rhoZ += 2.f * fView(fReadIndex[direction], cellReadIndex[direction]);
			}
		}
		
		const float rhoZInv = rhoPrev / rhoZ; 
		const float temp = 0.577350269f + (1.f/3.f) * rhoZInv;
		
		float uNormalPrev = (uxPrev * outerNormalX) + (uyPrev * outerNormalY) + (uzPrev * outerNormalZ);
		float uNormalImp = uNormalPrev + temp - TNL::sqrt( temp*temp + (2.f/3.f) * ( rhoZInv * ( uNormalPrev + 1.f ) - 1.f ) );
		float rhoImp = rhoZ / ( 1.f + uNormalImp );

		return ( rhoImp - 1.f );		
	};
	auto reduction = [] __cuda_callable__( const float& a, const float& b ) { return a + b; };
	
	float rhoAvg = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Info.nonReflectiveOutletCount, fetch, reduction, 0.f );
	rhoAvg /= (float)Info.nonReflectiveOutletCount;
	Info.nonReflectiveOutletRho = 1.f + rhoAvg;
}

