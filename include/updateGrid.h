#pragma once

#include "./applyCollision.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"

#include "./boundaryConditions/interpolatedBouncebackFunctions.h"
#include "./boundaryConditions/restoreRho.h"
#include "./boundaryConditions/restoreUxUyUz.h"
#include "./boundaryConditions/applyMBBC.h"
#include "./boundaryConditions/getNonReflectiveRho.h"

void updateGrid( GridStruct &Grid )
{	
	InfoStruct &Info = Grid.Info;
	
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
	auto wallDataView = Grid.Wall.wallDataArray.getConstView();
	auto gxWallView = Grid.Wall.gxArray.getView();
	auto gyWallView = Grid.Wall.gyArray.getView();
	auto gzWallView = Grid.Wall.gzArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		// read wallMap, early return if the cell itself is a wall
		const int wallMap = wallMapView( cell );
		if ( wallMap == -3 ) return; // this cell itself is a wall
		
		// decide if we track force for this cell. We dont track force if the cell is under a parent interface
		bool trackForce = true;
		if ( wallMap == -2 ) trackForce = false; // fluid cell under a parent interface -> dont track force
		
		// read wallData if this is a wall adjacent cell. So far only unpack wallID and parentInterfaceMarker
		uint32_t packed[4];
		int wallID = -1; 
		// here we dont want to allocate any more variables because it can still be a free fluid cell
		// so dont waste memory by allocating 26 link lengths
		if ( wallMap >= 0 )
		{
			const uint4 wallData = wallDataView( wallMap );
			packed[0] = wallData.x; packed[1] = wallData.y; packed[2] = wallData.z;	packed[3] = wallData.w;
			// so far unpack only wallID and parentInterfaceMarker
			bool parentInterfaceMarker;
			unpackWallID( packed, wallID, parentInterfaceMarker );
			if ( parentInterfaceMarker ) trackForce = false;
		}
		
		// fill iCell, jCell, kCell and NBR
		int iCell, jCell, kCell;
		NBRStruct NBR;
		getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
							shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
							Info );
							
		// read pre collision fPre
		float fPre[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ )	fPre[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
		
		// setup BC struct and load the current state into it
		// we will then pass the current state into the getLocalBC function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, ux, uy, uz
		BCStruct BC;
		BC.wallID = wallID;
		getRhoUxUyUz( BC.rho, BC.ux, BC.uy, BC.uz, fPre );
		getLocalBC( BC, iCell, jCell, kCell, Info );
		
		// add the rotor processing here. 
		// In case that gx, gy, gz is already non zero, for the rotor pretend that this forcing is already applied and results in shifted velocity
		// This way the rotor compensates for the global forcing by adding enough of its own force
		// the rotor only needs iCell, jCell, kCell, Info as input, we have that
		// as output it gives gx, gy, gz
		// in case of multiple rotors that could even overlap in the blurred area (gear pump!) gx, gy, gz should be averaged between all those
		// write rotor force for each rotor if trackForce is true
		// put the complete final gx, gy, gz ( combination of global forcing and all rotors ) back into the BC struct where collision will read it
		
		// now solve the shorter free fluid branch
		if ( wallMap < 0 )
		{
			applyCollision( fPre, BC, Info.nu );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
			// here fPre is just incorrectly named, we are writing fPost
			for ( int direction = 0; direction < 27; direction++ ) fView( fWriteIndex[direction], cellWriteIndex[direction] ) = fPre[direction]; 
			return;
		}
		
		// if we got here, we are dealing with a wall adjacent fluid cell
		bool linkExists[26];
		float linkLength[26];
		bool parentInterfaceMarker;
		unpackWallData( packed, linkExists, linkLength, wallID, parentInterfaceMarker );
		
		// to apply interpolated bounceback we want to remember both fPre and fPost, so
		float fPost[27];
		for ( int direction = 0; direction < 27; direction++ ) fPost[direction] = fPre[direction];
		applyCollision( fPost, BC, Info.nu );
		
		// now apply interpolated boundary condition, this will overwrite fPre
		// fPre[direction] will then contain the value that should get pulled from the wall next iteration
		applyIBB( fPre, fPost, linkExists, linkLength, BC );
		
		// write post collision distributions except for those that would run into a wall (we do not need those anymore)
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) 
		{
			if ( direction > 0 && linkExists[ direction-1 ] ) continue; // this one would hit a wall
			fView( fWriteIndex[direction], cellWriteIndex[direction] ) = fPost[direction]; 
		}
		
		// write distributions which will be pulled from walls the next iteration
		getNextPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 1; direction < 27; direction++ ) 
		{
			const int inverseDirection = INVERSE_DIRECTIONS[ direction ];
			if ( !linkExists[ inverseDirection-1 ] ) continue; // link does not exist
			fView( fWriteIndex[direction], cellWriteIndex[direction] ) = fPre[direction]; 
		}
		
		// last step: track force using momentum exchange method
		// Shuai Wang, Xinnan Wu, Cheng Peng, Songying Chen, Hao Liu
		// Analysis on the force evaluation by the momentum exchange 
		// method and a localized r­filling scheme for the lattice Boltzmann method, 2025
		// eq (15)
		if ( trackForce )
		{
			float gxWall = 0.f; float gyWall = 0.f; float gzWall = 0.f;
			for ( int direction = 1; direction < 27; direction++ ) 
			{
				const int inverseDirection = INVERSE_DIRECTIONS[ direction ];
				if ( !linkExists[ inverseDirection-1 ] ) continue; // link does not exist -> no force
				gxWall += fPost[ inverseDirection ] * ( CX_DIRECTIONS[ inverseDirection ] - BC.ux ) - fPre[ direction ] * ( CX_DIRECTIONS[ direction ] - BC.ux );
				gyWall += fPost[ inverseDirection ] * ( CY_DIRECTIONS[ inverseDirection ] - BC.uy ) - fPre[ direction ] * ( CY_DIRECTIONS[ direction ] - BC.uy );
				gzWall += fPost[ inverseDirection ] * ( CZ_DIRECTIONS[ inverseDirection ] - BC.uz ) - fPre[ direction ] * ( CZ_DIRECTIONS[ direction ] - BC.uz );
			}
			gxWallView( wallMap ) += gxWall;
			gyWallView( wallMap ) += gyWall;
			gzWallView( wallMap ) += gzWall;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	
	applyStreaming( Grid );
	
	// Now apply open boundary conditions
	for ( int openBCID = 0; openBCID < Grid.openBCs.size(); openBCID++ )
	{
		OpenBCArrayStruct &OpenBC = Grid.openBCs[ openBCID ];
		auto indexView = OpenBC.indexArray.getConstView();
		auto rhoPrevView = OpenBC.rhoPrevArray.getView();
		auto uNormalPrevView = OpenBC.uNormalPrevArray.getView();
		auto rhoCumulativeView = OpenBC.rhoCumulativeArray.getView();
		auto uNormalCumulativeView = OpenBC.uNormalCumulativeArray.getView();
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
			
			// identify known directions
							
			// read f
			float f[27];
			int cellIndex[27];
			int fIndex[27];
			getPreCollisionIndex( cellIndex, fIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fIndex[direction], cellIndex[direction]);
			// open boundary conditions are not well conditioned -> compensate
			for ( int direction = 0; direction < 27; direction++ ) f[direction] += DIRECTION_WEIGHTS[direction];
			
			// get BC
			BCStruct BC;
			getOpenBC( BC, iCell, jCell, kCell, Info );
			
			
			float rhoZ, rhoImp;
			
			getNonReflectiveInletValue( f, cxArray, cyArray, czArray, outerNormalX, outerNormalY, outerNormalZ, BC, rhoZ, rhoImp );
			
			// Boundary conditions are not well conditioned yet -> compensate
			const float weights[27] = { 8.f/27.f, 
				2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
				1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
				1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
			for ( int direction = 0; direction < 27; direction++ ) f[direction] += weights[direction];
			
			if ( Marker.nonReflectiveOutlet )
			{
				const float dRhoMax = 0.f; //0.0001f;
				const float rhoMin = Info.nonReflectiveOutletRho - dRhoMax;
				const float rhoMax = Info.nonReflectiveOutletRho + dRhoMax;
				//const float rhoMin = rhoImp - dRhoMax;
				//const float rhoMax = rhoImp + dRhoMax;
				BC.rho = std::clamp( BC.rho, rhoMin, rhoMax );
				BC.collisionLimiter = 0.f;
			}
			else if ( Marker.nonReflectiveInlet )
			{
				// Schlaffer 2013 eq (7.1) - (7.6)
				const float dRhoMax = 0.0001f;
				float uMin = 1.f - ( Info.nonReflectiveInletRhoZ / (Info.nonReflectiveInletRhoImp - dRhoMax) );
				float uMax = 1.f - ( Info.nonReflectiveInletRhoZ / (Info.nonReflectiveInletRhoImp + dRhoMax) );
				//float uMin = 1.f - ( rhoZ / (rhoImp - dRhoMax) );
				//float uMax = 1.f - ( rhoZ / (rhoImp + dRhoMax) );
				if ( outerNormalX + outerNormalY + outerNormalZ > 0 ) // right boundary -> inlet velocity is negative
				{
					float temp = uMax;
					uMax = - uMin;
					uMin = - temp;
				}
				if (outerNormalX != 0) BC.ux = std::clamp( BC.ux, uMin, uMax );
				else if (outerNormalY != 0) BC.uy = std::clamp( BC.uy, uMin, uMax );
				else if (outerNormalZ != 0) BC.uz = std::clamp( BC.uz, uMin, uMax );
			}
			if ( Marker.BCRho || Marker.nonReflectiveOutlet )
			{
				restoreUxUyUz( outerNormalX, outerNormalY, outerNormalZ, BC, f );				
			}
			else if ( Marker.BCU || Marker.nonReflectiveInlet )
			{
				restoreRho( outerNormalX, outerNormalY, outerNormalZ, BC, f );
			}
			applyMBBC( outerNormalX, outerNormalY, outerNormalZ, BC, f );
						
			// subtract the weights again for WC after BC is done
			for ( int direction = 0; direction < 27; direction++ ) f[direction] -= weights[direction];
			
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	}
	
	Info.updatesSinceTrackerReset++; 
	Info.iterationsFinished++;
}

/*

void updateGrid( GridStruct &Grid )
{	
	const InfoStruct &Info = Grid.Info;
	
	auto fArrayView  = Grid.fArray.getView();
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
		const int bitPackedMarkerInt = bitPackedMarkerView( cell );
		bool bitPackedMarkerBits[32];
		intToBools( bitPackedMarkerInt, bitPackedMarkerBits );
		
		MarkerStruct Marker;
		Marker.bounceback = bitPackedMarkerBits[27];
		Marker.movingBounceback = bitPackedMarkerBits[28];
		Marker.forcedVelocity = bitPackedMarkerBits[29] || bitPackedMarkerBits[31];
		Marker.deepRefinement = bitPackedMarkerBits[30];
		
		if ( Marker.deepRefinement ) return;
		
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		getMarkers( iCell, jCell, kCell, Marker, Info );
		
		if ( Marker.bounceback ) return; // bounceback gets implicitly applied by Esotwist
				
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		
		if ( Marker.movingBounceback )
		{
			const int cx[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
			const int cy[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
			const int cz[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
			const int inverseDirection[27] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25 };
			
			float fIn[27];
			for ( int direction = 1; direction < 27; direction++ )	
			{
				if ( bitPackedMarkerBits[direction] ) 
				{
					f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
					fIn[direction] = f[direction];
				}
			}
			BCStruct BC;
			getRhoUxUyUz( BC.rho, BC.ux, BC.uy, BC.uz, f );
			getBC( BC, iCell, jCell, kCell, Info, Marker ); 
			applyMovingBounceback( f, BC );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
			for ( int direction = 0; direction < 27; direction++ ) 
			{
				if ( bitPackedMarkerBits[inverseDirection[direction]] ) fView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
			}
			// track the torque
			float gx = 0.f;
			float gy = 0.f;
			float gz = 0.f;
			const float wallUx = BC.ux;
			const float wallUy = BC.uy;
			const float wallUz = BC.uz;
			
			for (int q = 1; q < 27; q++) {
				if ( !bitPackedMarkerBits[q] ) continue; // we are only interested if the neighbour is fluid
				gx += (cx[q] - wallUx) * fIn[q] - (cx[inverseDirection[q]] - wallUx) * f[inverseDirection[q]];
				gy += (cy[q] - wallUy) * fIn[q] - (cy[inverseDirection[q]] - wallUy) * f[inverseDirection[q]];
				gz += (cz[q] - wallUz) * fIn[q] - (cz[inverseDirection[q]] - wallUz) * f[inverseDirection[q]];
			}
			
			float x, y, z;
			getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );		
			convertToPhysicalForce( gx, gy, gz, Info );
			float T = - ( - gx * y + gy * x );
			
			fView( 27, cell ) += T;
			
			return;
		}
		
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
		
		BCStruct BC;
		// load the current state into the boundary condition struct
		getRhoUxUyUz( BC.rho, BC.ux, BC.uy, BC.uz, f );
		// pass the current state into the boundary condition function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, U
		getBC( BC, iCell, jCell, kCell, Info, Marker ); 
		
		if ( Marker.forcedVelocity )
		{
			const bool changedState = bitPackedMarkerBits[31];
			if ( changedState && bitPackedMarkerBits[29] ) // it has just become forced velocity
			{
				const float ratio = (float)Info.updatesSinceForcedVelocityUpdate / (float)FORCED_VELOCITY_UPDATE_PERIOD;
				BC.gx *= ratio; BC.gy *= ratio; BC.gz *= ratio;
			}
			else if ( changedState && !bitPackedMarkerBits[29] ) // it has just left forced velocity
			{
				const float ratio = (float)Info.updatesSinceForcedVelocityUpdate / (float)FORCED_VELOCITY_UPDATE_PERIOD;
				BC.gx *= (1.f-ratio); BC.gy *= (1.f-ratio); BC.gz *= (1.f-ratio);
			}
			float gx = BC.gx;
			float gy = BC.gy;
			float gz = BC.gz;
			float x, y, z;
			getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );		
			convertToPhysicalForce( gx, gy, gz, Info );
			float T = ( - gx * y + gy * x );
			fView( 27, cell ) += T;
		}
		else if ( Marker.fluid )
		{
			// do nothing, just skip the else block below
		}
		else
		{
			int outerNormalX, outerNormalY, outerNormalZ;
			getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
			const int cxArray[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
			const int cyArray[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
			const int czArray[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
			float rhoZ, rhoImp;
			
			getNonReflectiveInletValue( f, cxArray, cyArray, czArray, outerNormalX, outerNormalY, outerNormalZ, BC, rhoZ, rhoImp );
			
			// Boundary conditions are not well conditioned yet -> compensate
			const float weights[27] = { 8.f/27.f, 
				2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 2.f/27.f, 
				1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 1.f/54.f, 
				1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f, 1.f/216.f };
			for ( int direction = 0; direction < 27; direction++ ) f[direction] += weights[direction];
			
			if ( Marker.nonReflectiveOutlet )
			{
				const float dRhoMax = 0.f; //0.0001f;
				const float rhoMin = Info.nonReflectiveOutletRho - dRhoMax;
				const float rhoMax = Info.nonReflectiveOutletRho + dRhoMax;
				//const float rhoMin = rhoImp - dRhoMax;
				//const float rhoMax = rhoImp + dRhoMax;
				BC.rho = std::clamp( BC.rho, rhoMin, rhoMax );
				BC.collisionLimiter = 0.f;
			}
			else if ( Marker.nonReflectiveInlet )
			{
				// Schlaffer 2013 eq (7.1) - (7.6)
				const float dRhoMax = 0.0001f;
				float uMin = 1.f - ( Info.nonReflectiveInletRhoZ / (Info.nonReflectiveInletRhoImp - dRhoMax) );
				float uMax = 1.f - ( Info.nonReflectiveInletRhoZ / (Info.nonReflectiveInletRhoImp + dRhoMax) );
				//float uMin = 1.f - ( rhoZ / (rhoImp - dRhoMax) );
				//float uMax = 1.f - ( rhoZ / (rhoImp + dRhoMax) );
				if ( outerNormalX + outerNormalY + outerNormalZ > 0 ) // right boundary -> inlet velocity is negative
				{
					float temp = uMax;
					uMax = - uMin;
					uMin = - temp;
				}
				if (outerNormalX != 0) BC.ux = std::clamp( BC.ux, uMin, uMax );
				else if (outerNormalY != 0) BC.uy = std::clamp( BC.uy, uMin, uMax );
				else if (outerNormalZ != 0) BC.uz = std::clamp( BC.uz, uMin, uMax );
			}
			if ( Marker.BCRho || Marker.nonReflectiveOutlet )
			{
				restoreUxUyUz( outerNormalX, outerNormalY, outerNormalZ, BC, f );				
			}
			else if ( Marker.BCU || Marker.nonReflectiveInlet )
			{
				restoreRho( outerNormalX, outerNormalY, outerNormalZ, BC, f );
			}
			applyMBBC( outerNormalX, outerNormalY, outerNormalZ, BC, f );
						
			// subtract the weights again for WC after BC is done
			for ( int direction = 0; direction < 27; direction++ ) f[direction] -= weights[direction];
		}
		
		applyCollision( f, BC, Info.nu );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		
		for ( int direction = 0; direction < 27; direction++ ) fView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	
	applyStreaming( Grid );
	
	Info.updatesSinceRebuild++; 
	Info.updatesSinceMovingBouncebackUpdate++;
	Info.updatesSinceForcedVelocityUpdate++;
	Info.iterationsFinished++;
}

*/
