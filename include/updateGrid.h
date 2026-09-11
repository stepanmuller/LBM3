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

void updateSingleGrid( GridStruct &Grid )
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
		if ( wallMap >= 0 )
		{
			const uint4 wallData = wallDataView( wallMap );
			packed[0] = wallData.x; packed[1] = wallData.y; packed[2] = wallData.z;	packed[3] = wallData.w;
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
							
		// read pre collision f
		float f[27];
		int cellReadIndex[27];
		int fReadIndex[27];
		getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper );
		for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
		
		// setup BC struct and load the current state into it
		// we will then pass the current state into the getLocalBC function so that BC can also be a function of the current state 
		// example: get forcing for rotating domain as a function of rho, ux, uy, uz
		BCStruct BC;
		BC.wallID = wallID;
		getRhoUxUyUz( BC.rho, BC.ux, BC.uy, BC.uz, f );
		getLocalBC( BC, iCell, jCell, kCell, Info );
		
		// add the rotor processing here. 
		// In case that gx, gy, gz is already non zero, for the rotor pretend that this forcing is already applied and results in shifted velocity
		// This way the rotor compensates for the global forcing by adding enough of its own force
		// the rotor only needs iCell, jCell, kCell, Info as input, we have that
		// as output it gives gx, gy, gz
		// in case of multiple rotors that could even overlap in the blurred area (gear pump!) gx, gy, gz should be averaged between all those
		// write rotor force for each rotor if trackForce is true
		// put the complete final gx, gy, gz ( combination of global forcing and all rotors ) back into the BC struct where collision will read it
		
		applyCollision( f, BC, Info.nu );
		
		// do writes for the fluid branch
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
		uint32_t wallLinkMarker = 0u;
		float gxWall = 0.f; float gyWall = 0.f; float gzWall = 0.f;
		if ( wallMap >= 0 ) applyIBB( f, BC, Info.nu, packed, wallLinkMarker, gxWall, gyWall, gzWall );
		
		// write all directions. If there is a wall, switch the writing index to next pre-collision
		int cellWriteIndex = 0; 
		int fWriteIndex = 0;
		for ( int direction = 0; direction < 27; direction++ )
		{
			const bool linkExists = (wallLinkMarker & (1u << direction)) != 0u;
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
				const bool linkExists = (wallLinkMarker & (1u << direction)) != 0u;
				if ( !linkExists ) continue; // link does not exist -> no force
				// so we have a problem here that we no longer have fPre.. gonna solve this later by integrating this into the IBB
				gxWall += f[ inverseDirection ] * ( CX_DIRECTIONS[ inverseDirection ] - BC.ux ) - f[ direction ] * ( CX_DIRECTIONS[ direction ] - BC.ux );
				gyWall += f[ inverseDirection ] * ( CY_DIRECTIONS[ inverseDirection ] - BC.uy ) - f[ direction ] * ( CY_DIRECTIONS[ direction ] - BC.uy );
				gzWall += f[ inverseDirection ] * ( CZ_DIRECTIONS[ inverseDirection ] - BC.uz ) - f[ direction ] * ( CZ_DIRECTIONS[ direction ] - BC.uz );
			}
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
			rhoCumulativeView( index ) += BC.rho;
			uNormalPrevView( index ) = uNormal;
			uNormalCumulativeView( index ) += uNormal;
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
