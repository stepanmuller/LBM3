#pragma once

#include "./updateInterface.h"

#include "./applyCollision.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"
#include "./rotorFunctions.h"

#include "./boundaryConditions/interpolatedBouncebackFunctions.h"
#include "./boundaryConditions/restoreRho.h"
#include "./boundaryConditions/restoreUxUyUz.h"
#include "./boundaryConditions/applyMBBC.h"
#include "./boundaryConditions/getNonReflectiveRho.h"

void updateSingleGrid( GridStruct &Grid )
{	
	InfoStruct &Info = Grid.Info;
	
	auto fView  = Grid.fArray.getView();
	const bool &esotwistFlipper = Grid.Info.esotwistFlipper;
	auto shifterView = Grid.IJKNBR.shifterArray.getConstView();	
	auto iView = Grid.IJKNBR.iArray.getConstView();
	auto jView = Grid.IJKNBR.jArray.getConstView();
	auto kView = Grid.IJKNBR.kArray.getConstView();
	auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
	auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
	auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
	auto wallMapView = Grid.Wall.wallMapArray.getConstView();
	auto wallDataView = Grid.Wall.wallDataArray.getConstView();
	auto linkLengthView = Grid.Wall.linkLengthArray.getConstView();
	auto gxWallView = Grid.Wall.gxArray.getView();
	auto gyWallView = Grid.Wall.gyArray.getView();
	auto gzWallView = Grid.Wall.gzArray.getView();
	
	const int rotorCount = Grid.rotorViews.size();
	auto* rotorViews = Grid.rotorViews.data();
	// Advance rotors and synchronize rotor Info
	constexpr double twoPi = 6.283185307179586476925286766559;
	for (int rotorID = 0; rotorID < rotorCount; rotorID++)
	{
		auto& RotorInfo = Grid.rotors[rotorID].Info;
		RotorInfo.radiansElapsed = std::fmod(	RotorInfo.radiansElapsed + static_cast<double>(Info.dtPhys) * RotorInfo.radiansPerSecond,	twoPi);
		// fmod can return a negative remainder.
		if (RotorInfo.radiansElapsed < 0.0) RotorInfo.radiansElapsed += twoPi;
		// Adding twoPi to a tiny negative remainder can round to twoPi.
		if (RotorInfo.radiansElapsed >= twoPi) RotorInfo.radiansElapsed = 0.0;
		Grid.rotorViews[rotorID].Info = RotorInfo;
	}
	
	auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
	{
		// read wallMap, early return if the cell itself is a wall
		const int wallMap = wallMapView( cell );
		if ( wallMap == -3 ) return; // this cell itself is a wall
		
		// decide if we track force for this cell. We dont track force if the cell is under a an interface overlap
		bool trackForce = true;
		if ( wallMap == -2 ) trackForce = false; // fluid cell under an interface overlap -> dont track force
		
		// read wallData if this is a wall adjacent cell. So far only unpack wallID and interfaceOverlapMarker
		uint32_t wallData = 0u;
		int wallID = -1; 
		if ( wallMap >= 0 )
		{
			wallData = wallDataView( wallMap );
			bool interfaceOverlapMarker;
			unpackWallID( wallData, wallID, interfaceOverlapMarker );
			if ( interfaceOverlapMarker ) trackForce = false;
		}
		
		// fill iCell, jCell, kCell and NBR
		int iCell, jCell, kCell;
		NBRStruct NBR;
		getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
							shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
							Info );
							
		// read pre collision f
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
		
		// calculate current state
		float rho, ux, uy, uz;
		getRhoUxUyUz( rho, ux, uy, uz, f );
		// setup BC struct and load the current state into it
		// we will then pass the current state into the getLocalBC function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, ux, uy, uz
		BCStruct BC;
		BC.wallID = wallID;
		BC.rho = rho; BC.ux = ux; BC.uy = uy; BC.uz = uz;
		getLocalBC( BC, iCell, jCell, kCell, Info );
		
		// process the rotors
		if ( rotorCount > 0 )
		{
			// the rotor needs x, y, z, Info as input, we have that
			float x, y, z;
			getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
			// In case that gx, gy, gz is already non zero, for the rotor pretend that this forcing is already applied and results in shifted velocity
			// This is because we want to ensure that after forcing, the target velocity is achieved, so the rotor must only supply
			// the part of the force that is missing
			const float rhoInv = 1.f / rho;
			const float uxPreRotor = ( ux * rho + BC.gx) * rhoInv;
			const float uyPreRotor = ( uy * rho + BC.gy) * rhoInv;
			const float uzPreRotor = ( uz * rho + BC.gz) * rhoInv;
			// it can also happen that there are more slightly overlapping rotors ( gear pump! )
			// because of this, we will be tracking the cumulative rotor fraction
			float rotorFractionCumulative = 0.f;
			// when browsing a rotor:
			// 1) find its fraction
			// 2) if rotorFractionCumulative + fraction > 1, 
			//			fraction = 1 - rotorFractionCumulative 
			//			set rotorFractionCumulative to 1
			// 	  else, rotorFractionCumulative += fraction
			// 3) proceed by doing BC.gx += gxRotor * fraction, etc -> this eventually results in BC having the complete forcing
			// 4) after the rotor processing, check if rotorFractionCumulative >= 1, break if so
			for (int rotorID = 0; rotorID < rotorCount; rotorID++)
			{
				processRotor( BC, rho, uxPreRotor, uyPreRotor, uzPreRotor, x, y, z, rotorFractionCumulative, trackForce, Info, rotorViews[rotorID] );
				// this adds rotor forcing to the BC forcing
				if ( rotorFractionCumulative >= 1.f ) break;
			}
		}
		
		applyCollision( f, BC, Info.nu );
		
		// do writes for the fluid branch and exit
		if ( wallMap < 0 )
		{
			int cellWriteIndex[27]; 
			int fWriteIndex[27];
			getPostCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper );
			for ( int direction = 0; direction < 27; direction++ ) fView( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction]; 
			return;
		}
		
		// if we got here, we are dealing with a wall adjacent fluid cell
		// apply IBB. f[direction] which would get streamed into a wall get overwritten.
		// In their place, we will find f[inverseDirection] that we need to receive from the wall next round.
		// use bit packed wallLinkMarker to remember which directions the walls are, also track the forces
		float gxWall = 0.f; float gyWall = 0.f; float gzWall = 0.f;
		applyIBB( f, BC, Info.nu, gxWall, gyWall, gzWall, wallData, wallMap, linkLengthView );
		
		// write all directions. If there is a wall, switch the writing index to next pre-collision and inverse direction
		int cellWriteIndex = 0; 
		int fWriteIndex = 0;
		for ( int direction = 0; direction < 27; direction++ )
		{
			const bool linkExists = direction != 0 && (wallData & (1u << direction)) != 0u;
			if ( linkExists ) 
			{
				// f[direction] would point into a wall, so instead we write the incoming distribution from the wall
				const int inverseDirection = INVERSE_DIRECTIONS[ direction ];
				getNextPreCollisionIndexSingle( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, inverseDirection );
				fView( fWriteIndex, cellWriteIndex ) = f[direction]; 
			}
			else
			{
				// regular post collision write of f[direction]
				getPostCollisionIndexSingle( cellWriteIndex, fWriteIndex, NBR, esotwistFlipper, direction );
				fView( fWriteIndex, cellWriteIndex ) = f[direction]; 
			}
		}
	
		// last step: write force
		if ( trackForce )
		{
			gxWallView( wallMap ) += gxWall;
			gyWallView( wallMap ) += gyWall;
			gzWallView( wallMap ) += gzWall;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	
	applyStreaming( Grid );
	
	// Now apply open boundary conditions
	for ( int openBCID = 0; openBCID < (int)Grid.openBCs.size(); openBCID++ )
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
			
			// read wallMap, this is to find if we should track flow through this cell
			// we dont track flow if the cell is under an interface overlap
			const int wallMap = wallMapView( cell );
			bool trackFlow = true;
			if ( wallMap == -2 ) trackFlow = false; // fluid cell under an interface overlap -> dont track flow
			
			// read wallData if this is a wall adjacent cell. So far only unpack wallID and interfaceOverlapMarker
			// read wallData if this is a wall adjacent cell. So far only unpack wallID and interfaceOverlapMarker
			uint32_t wallData = 0u;
			int wallID = -1; 
			if ( wallMap >= 0 )
			{
				wallData = wallDataView( wallMap );
				bool interfaceOverlapMarker;
				unpackWallID( wallData, wallID, interfaceOverlapMarker );
				if ( interfaceOverlapMarker ) trackFlow = false;
			}
			
			// fill iCell, jCell, kCell and NBR
			int iCell, jCell, kCell;
			NBRStruct NBR;
			getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
								shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
								Info );
			
			// identify outer normal
			int outerNormalX, outerNormalY, outerNormalZ;
			getOuterNormal( iCell, jCell, kCell, outerNormalX, outerNormalY, outerNormalZ, Info ); 
			
			// read known f only, calculate rhoZ along the way
			float f[27];
			int cellIndex[27];
			int fIndex[27];
			getPreCollisionIndex( cellIndex, fIndex, NBR, esotwistFlipper );
			float rhoZ = 0.f;
			for ( int direction = 0; direction < 27; direction++ )
			{
				const bool unknown = (	outerNormalX * CX_DIRECTIONS[direction] < 0 ||
									 	outerNormalY * CY_DIRECTIONS[direction] < 0 ||
										outerNormalZ * CZ_DIRECTIONS[direction] < 0 );
				if ( unknown ) continue;
				
				f[direction] = fView(fIndex[direction], cellIndex[direction]);
				// open boundary conditions are not well conditioned -> compensate
				f[direction] += DIRECTION_WEIGHTS[direction];
				
				const int product =   outerNormalX * CX_DIRECTIONS[ direction ]
									+ outerNormalY * CY_DIRECTIONS[ direction ]
									+ outerNormalZ * CZ_DIRECTIONS[ direction ];
				if ( product == 0 )	rhoZ += f[direction];
				else rhoZ += 2.f * f[direction];
			}
			
			// read rhoPrev, uNormalPrev
			const float rhoPrev = rhoPrevView( index );
			const float uNormalPrev = uNormalPrevView( index );
			
			// get BC
			BCStruct BC;
			getOpenBC( BC, iCell, jCell, kCell, Info );
			
			// get non reflective rho value if the cell is a face cell
			// if it is an edge or corner cell, disable non reflectivity and switch to strict dirichlet,
			// because non reflectivity is only defined for face cells
			bool useNonReflective = true;
			float rhoNonReflective = 1.f;
			if ( TNL::abs( outerNormalX ) + TNL::abs( outerNormalY ) + TNL::abs( outerNormalZ ) == 1 )
			{
				rhoNonReflective = getNonReflectiveRho( rhoZ, rhoPrev, uNormalPrev );
			}
			else useNonReflective = false;
			
			// also disable non reflectivity if BC.rhoReflectionTolerance >= 1.f
			if ( BC.rhoReflectionTolerance >= 1.f ) useNonReflective = false;
			
			// adjust rho, ux, uy, uz by combining Dirichlet with non reflectivity
			const float &dRhoMax = BC.rhoReflectionTolerance;
			if ( BC.dirichletRho )
			{
				if ( useNonReflective )
				{
					const float rhoMin = rhoNonReflective - dRhoMax;
					const float rhoMax = rhoNonReflective + dRhoMax;
					BC.rho = std::clamp( BC.rho, rhoMin, rhoMax ); 
				}
				restoreUxUyUz( outerNormalX, outerNormalY, outerNormalZ, BC, f );
			}
			else if ( BC.dirichletU )
			{
				if ( useNonReflective )
				{
					// Schlaffer disertation 2013 eq (7.1) - (7.6)
					float uNormalMin = ( rhoZ / ( rhoNonReflective + dRhoMax) ) - 1.f;
					float uNormalMax = ( rhoZ / ( rhoNonReflective - dRhoMax) ) - 1.f;
					if ( outerNormalX > 0 ) BC.ux = std::clamp( BC.ux, uNormalMin, uNormalMax );
					if ( outerNormalX < 0 ) BC.ux = std::clamp( BC.ux, -uNormalMax, -uNormalMin );
					if ( outerNormalY > 0 ) BC.uy = std::clamp( BC.uy, uNormalMin, uNormalMax );
					if ( outerNormalY < 0 ) BC.uy = std::clamp( BC.uy, -uNormalMax, -uNormalMin );
					if ( outerNormalZ > 0 ) BC.uz = std::clamp( BC.uz, uNormalMin, uNormalMax );
					if ( outerNormalZ < 0 ) BC.uz = std::clamp( BC.uz, -uNormalMax, -uNormalMin );
				}
				restoreRho( outerNormalX, outerNormalY, outerNormalZ, BC, f );
			}
			// now rho, ux, uy, uz is set as needed so go on calculate unknown f by using MBBC
			applyMBBC( outerNormalX, outerNormalY, outerNormalZ, BC, f );
						
			// write unknown f only
			for ( int direction = 0; direction < 27; direction++ )
			{
				const bool unknown = (	outerNormalX * CX_DIRECTIONS[direction] < 0 ||
									 	outerNormalY * CY_DIRECTIONS[direction] < 0 ||
										outerNormalZ * CZ_DIRECTIONS[direction] < 0 );
				if ( unknown ) 
				{
					// open boundary conditions are not well conditioned -> compensate
					f[direction] -= DIRECTION_WEIGHTS[direction];
					fView(fIndex[direction], cellIndex[direction]) = f[direction];
				}
			}
			
			// track previous and cumulative values
			const float uNormal = (float)outerNormalX * BC.ux + (float)outerNormalY * BC.uy + (float)outerNormalZ * BC.uz;
			rhoPrevView( index ) = BC.rho;
			uNormalPrevView( index ) = uNormal;
			if ( trackFlow )
			{
				rhoCumulativeView( index ) += BC.rho;
				uNormalCumulativeView( index ) += uNormal;
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, OpenBC.openBCCount, cellLambda );
	}
	
	Info.updatesSinceTrackerReset++; 
	Info.iterationsFinished++;
}

void updateAllGrids( std::vector<GridStruct>& grids, int level ) 
{
    updateSingleGrid( grids[ level ] );
    if (level < GRID_LEVEL_COUNT - 1 ) 
    {
        for ( int i = 0; i < 2; i++ ) updateAllGrids( grids, level + 1 );
        updateInterface( grids[level], grids[level + 1] );
    }
}
