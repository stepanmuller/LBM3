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
			getPreCollisionIndex( cellIndex, fIndex, NBR, esotwistFlipper, Info );
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
