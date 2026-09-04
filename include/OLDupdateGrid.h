#pragma once

#include "./applyCollision.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"
#include "./markerFunctions.h"
#include "./boundaryConditions/applyBounceback.h"
#include "./boundaryConditions/applyMovingBounceback.h"
#include "./boundaryConditions/restoreRho.h"
#include "./boundaryConditions/restoreUxUyUz.h"
#include "./boundaryConditions/applyMBBC.h"
#include "./boundaryConditions/applyNonReflectiveOutlet.h"
#include "./boundaryConditions/applyNonReflectiveInlet.h"

void solveInterpolatedBB( GridStruct &Grid )
{
	InfoStruct &Info = Grid.Info;
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	
	auto bitPackedMarkerView = Grid.bitPackedMarkerArray.getConstView();
	auto indexList = Grid.interpolatedBBCellList.getConstView();
	auto interpolatedBBLinkLengthsView = Grid.interpolatedBBLinkLengths.getConstView();
	
	const float omega1 = 1.f / (3.f * (Info.nu) + 0.5f);
	const int inverseDirection[27] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25 };
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = indexList( index );
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( NBR.kPlus );
		finishNBRPlus( NBR, Info );
		
		float fStar[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
		for ( int direction = 0; direction < 27; direction++ ) fStar[direction] =  fView( fReadIndex[direction], cellReadIndex[direction] );
		
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, fStar );
		float fEq[27];
		getFeq( rho, ux, uy, uz, fEq );
		
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, Info );
		
		const int bitPackedMarkerInt = bitPackedMarkerView( cell );
		bool bitPackedMarkerBits[32];
		intToBools( bitPackedMarkerInt, bitPackedMarkerBits );
		for ( int direction = 1; direction < 27; direction++ )
		{
			if ( !bitPackedMarkerBits[direction] ) // this means there is no fluid coming from this direction
			{
				float q = interpolatedBBLinkLengthsView( direction, index );
				float fStarMinus = fStar[direction];
				float fStarPlus = fStar[inverseDirection[direction]];
				float fEqMinus = fEq[direction];
				float fEqPlus = fEq[inverseDirection[direction]];
				// Geier 2015 (E.4)
				float fPlus = 0.5f * ( fStarPlus - fStarMinus ) + ( fStarPlus + fStarMinus - omega1 * ( fEqPlus + fEqMinus ) ) / ( 2.f - 2.f * omega1 );
				// Geier 2015 (E.3)
				float fWallPlus = ( 1.f - q ) * fPlus + q * fStarPlus;
				// Geier 2015 (E.2)
				float fWallMinus = fWallPlus; // we dont use MBB here
				// Geier 2015 (E.1)
				float fResult = (1.f / ( q + 1.f )) * fWallMinus + (q / ( q + 1.f )) * fStarMinus;				
				//float fResult = fStar[inverseDirection[direction]];
				fView( fWriteIndex[direction], cellWriteIndex[direction] ) = fResult;
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.interpolatedBBCellList.getSize(), cellLambda );
}

void updateGrid( GridStruct &Grid )
{	
	InfoStruct &Info = Grid.Info;
	const bool &esotwistFlipper = Grid.esotwistFlipper;
	
	auto fView  = Grid.fArray.getView();
	
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();

	auto jPlusView = Grid.NBR.jPlusArray.getConstView();
	auto kPlusView = Grid.NBR.kPlusArray.getConstView();
	
	auto bitPackedMarkerView = Grid.bitPackedMarkerArray.getConstView();
	
	applyNonReflectiveInlet(Grid);
	applyNonReflectiveOutlet(Grid);
	solveInterpolatedBB(Grid);
	
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
