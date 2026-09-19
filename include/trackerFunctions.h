#pragma once

#include "./types.h"
#include "./boundaryConditions/interpolatedBouncebackFunctions.h"

void initializeTracker( TrackerStruct &Tracker, std::vector<GridStruct>& grids )
{
	std::cout << "Initializing Tracker ... " << std::flush;
	// set openBCCount, rotorCount, wallCount
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
	{
		Tracker.openBCCount = TNL::max( Tracker.openBCCount, grids[level].openBCs.size() );
		Tracker.rotorCount = TNL::max( Tracker.rotorCount, grids[level].rotors.size() );
		// wallCount requires a reduction
		auto indexView = grids[level].Wall.indexArray.getConstView();
		auto wallMapView = grids[level].Wall.wallMapArray.getConstView();
		auto wallDataView = grids[level].Wall.wallDataArray.getConstView();
		auto fetch = [ = ] __cuda_callable__( const int index )
		{
			const int cell = indexView( index );
			const int wallMap = wallMapView( cell );
			int wallID = -1; 
			if ( wallMap < 0 ) return wallID; 
			// read wallData if this is a wall adjacent cell, unpack wallID
			uint32_t wallData = 0u;
			if ( wallMap >= 0 )
			{
				wallData = wallDataView( wallMap );
				bool interfaceOverlapMarker;
				unpackWallID( wallData, wallID, interfaceOverlapMarker );
			}
			return wallID;
		};
		auto reduction = [] __cuda_callable__( const int& a, const int& b )
		{
			return TNL::max( a, b );
		};
		const int wallIDMax = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, grids[level].Wall.wallAdjacentCount, fetch, reduction, -1 );
		Tracker.wallCount = TNL::max( Tracker.wallCount, wallIDMax + 1 );
	}
	// set sizes and values
	if constexpr (TRACK_OPEN_BOUNDARIES)
	{
		Tracker.normalVelocityArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
		Tracker.massFlowArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
		Tracker.momentumThrustArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
		Tracker.pressureArray.setSizes( Tracker.openBCCount, ITERATION_COUNT );
		Tracker.pressurePowerArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
		Tracker.normalKineticPowerArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
		Tracker.normalVelocityArray.setValue( 0.f ); 
		Tracker.massFlowArray.setValue( 0.f ); 
		Tracker.momentumThrustArray.setValue( 0.f ); 
		Tracker.pressureArray.setValue( 0.f );
		Tracker.pressurePowerArray.setValue( 0.f ); 
		Tracker.normalKineticPowerArray.setValue( 0.f ); 
		// now tiny arrays that only hold the value from the last iteration
		Tracker.normalVelocity.setSize( Tracker.openBCCount ); 
		Tracker.massFlow.setSize( Tracker.openBCCount ); 
		Tracker.momentumThrust.setSize( Tracker.openBCCount ); 
		Tracker.pressure.setSize( Tracker.openBCCount );
		Tracker.pressurePower.setSize( Tracker.openBCCount ); 
		Tracker.normalKineticPower.setSize( Tracker.openBCCount ); 
	}
	if constexpr (TRACK_WALL_FORCE)
	{
		Tracker.wallFxArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
		Tracker.wallFyArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
		Tracker.wallFzArray.setSizes( Tracker.wallCount, ITERATION_COUNT );
		Tracker.wallTxArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
		Tracker.wallTyArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
		Tracker.wallTzArray.setSizes( Tracker.wallCount, ITERATION_COUNT );
		Tracker.wallFxArray.setValue( 0.f ); 
		Tracker.wallFyArray.setValue( 0.f ); 
		Tracker.wallFzArray.setValue( 0.f );
		Tracker.wallTxArray.setValue( 0.f ); 
		Tracker.wallTyArray.setValue( 0.f ); 
		Tracker.wallTzArray.setValue( 0.f );
		// now tiny arrays that only hold the value from the last iteration
		Tracker.wallFx.setSize( Tracker.wallCount ); 
		Tracker.wallFy.setSize( Tracker.wallCount ); 
		Tracker.wallFz.setSize( Tracker.wallCount );
		Tracker.wallTx.setSize( Tracker.wallCount ); 
		Tracker.wallTy.setSize( Tracker.wallCount ); 
		Tracker.wallTz.setSize( Tracker.wallCount );
	}
	if constexpr (TRACK_ROTOR_FORCE)
	{
		Tracker.rotorFxArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
		Tracker.rotorFyArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
		Tracker.rotorFzArray.setSizes( Tracker.rotorCount, ITERATION_COUNT );
		Tracker.rotorTxArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
		Tracker.rotorTyArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
		Tracker.rotorTzArray.setSizes( Tracker.rotorCount, ITERATION_COUNT );
		Tracker.rotorFxArray.setValue( 0.f ); 
		Tracker.rotorFyArray.setValue( 0.f ); 
		Tracker.rotorFzArray.setValue( 0.f );
		Tracker.rotorTxArray.setValue( 0.f ); 
		Tracker.rotorTyArray.setValue( 0.f ); 
		Tracker.rotorTzArray.setValue( 0.f );
		// now tiny arrays that only hold the value from the last iteration
		Tracker.rotorFx.setSize( Tracker.rotorCount ); 
		Tracker.rotorFy.setSize( Tracker.rotorCount ); 
		Tracker.rotorFz.setSize( Tracker.rotorCount );
		Tracker.rotorTx.setSize( Tracker.rotorCount ); 
		Tracker.rotorTy.setSize( Tracker.rotorCount ); 
		Tracker.rotorTz.setSize( Tracker.rotorCount );
	}
	
	std::cout << "Done" << std::endl;
	std::cout << std::endl;
}

void updateTracker( TrackerStruct &Tracker, std::vector<GridStruct>& grids )
{
	// update iterationsFinished
	Tracker.iterationsFinished = grids[0].Info.iterationsFinished;
	// trackerIndex is the write position
	const int trackerIndex = grids[0].Info.iterationsFinished - 1; 
	
	// 1) Open BC
	if constexpr (TRACK_OPEN_BOUNDARIES)
	{
		for ( int openBCID = 0; openBCID < Tracker.openBCCount; openBCID++ )
		{
			float aream2 = 0.f;
			float normalVelocity = 0.f; 
			float massFlow = 0.f; 
			float momentumThrust = 0.f;
			float pressure = 0.f;
			float pressurePower = 0.f;
			float normalKineticPower = 0.f;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
			{
				GridStruct &Grid = grids[level];
				const InfoStruct &Info = Grid.Info;
				if ( (int)Grid.openBCs.size() > openBCID )
				{
					const float invIterationSpan = 1.f / (float)Grid.Info.iterationsSinceTrackerReset;
					
					auto dRhoView = Grid.openBCs[openBCID].dRhoCumulativeArray.getConstView();
					auto uNormalView = Grid.openBCs[openBCID].uNormalCumulativeArray.getConstView();
					
					auto fetch = [=] __cuda_callable__ (const int index) -> SixFloatArrayType 
					{
						const float dRho = dRhoView(index) * invIterationSpan;
						const float uNormal = uNormalView(index) * invIterationSpan;
						const float rho = 1.0f + dRho;

						SixFloatArrayType resultArray;
						resultArray[0] = uNormal;                    				// volumetricFlow
						resultArray[1] = rho * uNormal;                    			// massFlow
						resultArray[2] = rho * uNormal * TNL::abs(uNormal);      	// momentumThrust
						resultArray[3] = dRho;                          			// pressure integral
						resultArray[4] = dRho * uNormal;                     		// pressurePower
						resultArray[5] = 0.5f * rho * uNormal * uNormal * uNormal; 	// normalKineticPower
						return resultArray;
					};
					auto reduction = [] __cuda_callable__ (const SixFloatArrayType& a, const SixFloatArrayType& b) 
					{
						SixFloatArrayType resultArray;
						for (int resultIndex = 0; resultIndex < 6; resultIndex++ )
							resultArray[resultIndex] = a[resultIndex] + b[resultIndex];
						return resultArray;
					};
					SixFloatArrayType resultArray = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Grid.openBCs[openBCID].openBCCount, 
																									fetch, reduction, SixFloatArrayType(0.0f) );
					// convert to physical units
					const float cellAream2 = (Info.res/1000.f) * (Info.res/1000.f);
					const float velocityMultiplier = (Info.res/1000.f) / Info.dtPhys;
					resultArray[0] *= cellAream2 * velocityMultiplier;
					resultArray[1] *= cellAream2 * velocityMultiplier * RHO_PHYS;
					resultArray[2] *= cellAream2 * velocityMultiplier * velocityMultiplier * RHO_PHYS;
					convertToPhysicalPressure( resultArray[3], Info );
					resultArray[3] *= cellAream2;
					convertToPhysicalPressure( resultArray[4], Info );
					resultArray[4] *= cellAream2 * velocityMultiplier;
					resultArray[5] *= cellAream2 * velocityMultiplier * velocityMultiplier * velocityMultiplier * RHO_PHYS;
					
					// accumulate
					aream2 += Grid.openBCs[openBCID].trackFlowCount * cellAream2;
					normalVelocity     += resultArray[0];
					massFlow           += resultArray[1];
					momentumThrust     += resultArray[2];
					pressure           += resultArray[3];
					pressurePower      += resultArray[4];
					normalKineticPower += resultArray[5];
				}
			}
			// we have a pressure and velocity integral now, divide to get area averaged pressure and velocity
			normalVelocity /= aream2;
			pressure /= aream2;
			// write the results
			// if tracker period is > 1, also write (period-1) values backward and forward
			// backward, because in those steps we did not launch updateTracker
			// forward, because if on the very last iteration the tracker does not launch,
			// we still want to have some values there (prevent zeros at the very end of the history plot)
			for ( int shift = 1 - TRACKER_PERIOD; shift < TRACKER_PERIOD; shift++ )
			{
				const int sampleIndex = trackerIndex + shift;
				if (sampleIndex < 0 || sampleIndex >= ITERATION_COUNT) continue;
				Tracker.normalVelocityArray( openBCID, sampleIndex ) = normalVelocity; 
				Tracker.massFlowArray( openBCID, sampleIndex ) = massFlow;
				Tracker.momentumThrustArray( openBCID, sampleIndex ) = momentumThrust;
				Tracker.pressureArray( openBCID, sampleIndex ) = pressure;
				Tracker.pressurePowerArray( openBCID, sampleIndex ) = pressurePower;
				Tracker.normalKineticPowerArray( openBCID, sampleIndex ) = normalKineticPower;
			}
			// update the small arrays
			Tracker.normalVelocity( openBCID ) = normalVelocity; 
			Tracker.massFlow( openBCID ) = massFlow;
			Tracker.momentumThrust( openBCID ) = momentumThrust;
			Tracker.pressure( openBCID ) = pressure;
			Tracker.pressurePower( openBCID ) = pressurePower;
			Tracker.normalKineticPower( openBCID ) = normalKineticPower;
		}
	}
	
	// 2) Rotors
	if constexpr (TRACK_ROTOR_FORCE)
	{
		for ( int rotorID = 0; rotorID < Tracker.rotorCount; rotorID++ )
		{
			float fx = 0.f;
			float fy = 0.f; 
			float fz = 0.f; 
			float tx = 0.f;
			float ty = 0.f;
			float tz = 0.f;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
			{
				GridStruct &Grid = grids[level];
				const InfoStruct &Info = Grid.Info;
				if ( (int)Grid.rotors.size() > rotorID )
				{
					const float invIterationSpan = 1.f / (float)Grid.Info.iterationsSinceTrackerReset;
					
					const RotorInfoStruct &InfoRotor = Grid.rotors[rotorID].Info;
					const BoundsStruct &Bounds = InfoRotor.Bounds;
					const float resBlock = InfoRotor.res * 4.f;
					const float resForce = resBlock / 3.f;
					const int blockCountX = InfoRotor.cellCountX / 4;
					const int blockCountY = InfoRotor.cellCountY / 4;
					const int blockCountXY = blockCountX * blockCountY;
					auto indexView = Grid.rotors[rotorID].indexArray.getConstView();
					auto rotorMapView = Grid.rotors[rotorID].rotorMapArray.getConstView();
					auto gxView = Grid.rotors[rotorID].gxArray.getConstView();
					auto gyView = Grid.rotors[rotorID].gyArray.getConstView();
					auto gzView = Grid.rotors[rotorID].gzArray.getConstView();
					
					auto fetch = [=] __cuda_callable__ (const int index) -> SixFloatArrayType 
					{
						const int blockIndex = indexView( index );
						const int kBlock = blockIndex / blockCountXY;
						const int remainder = blockIndex % blockCountXY;
						const int jBlock = remainder / blockCountX;
						const int iBlock = remainder % blockCountX;
						// x0, y0, z0 = position of block bottom left force cell relative to the point of rotation
						const float x0 = iBlock * resBlock + Bounds.xMin - InfoRotor.ox + 0.5f * resForce; 
						const float y0 = jBlock * resBlock + Bounds.yMin - InfoRotor.oy + 0.5f * resForce;
						const float z0 = kBlock * resBlock + Bounds.zMin - InfoRotor.oz + 0.5f * resForce;
						float fxBlock = 0.f; float fyBlock = 0.f; float fzBlock = 0.f;
						float txBlock = 0.f; float tyBlock = 0.f; float tzBlock = 0.f;
						for ( int kForce = 0; kForce < 3; kForce++ )
						{
							const float z = z0 + kForce * resForce;
							for ( int jForce = 0; jForce < 3; jForce++ )
							{
								const float y = y0 + jForce * resForce; 
								for ( int iForce = 0; iForce < 3; iForce++ )
								{
									const float x = x0 + iForce * resForce; 
									const int forceIndex = index * 27 + kForce * 9 + jForce * 3 + iForce;
									const float gx = gxView( forceIndex );
									const float gy = gyView( forceIndex );
									const float gz = gzView( forceIndex );
									fxBlock += gx; fyBlock += gy; fzBlock += gz;
									txBlock += + y * gz - z * gy;
									tyBlock += + z * gx - x * gz;
									tzBlock += + x * gy - y * gx;
								}
							}
						}

						SixFloatArrayType resultArray;
						resultArray[0] = fxBlock;                    				
						resultArray[1] = fyBlock;	
						resultArray[2] = fzBlock;	
						resultArray[3] = txBlock;	
						resultArray[4] = tyBlock;	
						resultArray[5] = tzBlock; 	
						return resultArray;
					};
					auto reduction = [] __cuda_callable__ (const SixFloatArrayType& a, const SixFloatArrayType& b) 
					{
						SixFloatArrayType resultArray;
						for (int resultIndex = 0; resultIndex < 6; resultIndex++ )
							resultArray[resultIndex] = a[resultIndex] + b[resultIndex];
						return resultArray;
					};
					SixFloatArrayType resultArray = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, (int)Grid.rotors[rotorID].indexArray.getSize(), 
																									fetch, reduction, SixFloatArrayType(0.0f) );
					// normalize via invIterationSpan
					for ( int i = 0; i < 6; ++i ) resultArray[i] *= invIterationSpan;
					
					// convert to physical units
					convertToPhysicalForce( resultArray[0], resultArray[1], resultArray[2], Info );
					convertToPhysicalForce( resultArray[3], resultArray[4], resultArray[5], Info );
					// at this point torque is in Nmm, convert to Nm
					resultArray[3] *= 0.001f;
					resultArray[4] *= 0.001f;
					resultArray[5] *= 0.001f;
					
					// accumulate
					fx += resultArray[0];
					fy += resultArray[1];
					fz += resultArray[2];
					tx += resultArray[3];
					ty += resultArray[4];
					tz += resultArray[5];
				}
			}
			
			// write the results
			// if tracker period is > 1, also write (period-1) values backward and forward
			// backward, because in those steps we did not launch updateTracker
			// forward, because if on the very last iteration the tracker does not launch,
			// we still want to have some values there (prevent zeros at the very end of the history plot)
			for ( int shift = 1 - TRACKER_PERIOD; shift < TRACKER_PERIOD; shift++ )
			{
				const int sampleIndex = trackerIndex + shift;
				if (sampleIndex < 0 || sampleIndex >= ITERATION_COUNT) continue;
				Tracker.rotorFxArray( rotorID, sampleIndex ) = fx; 
				Tracker.rotorFyArray( rotorID, sampleIndex ) = fy;
				Tracker.rotorFzArray( rotorID, sampleIndex ) = fz;
				Tracker.rotorTxArray( rotorID, sampleIndex ) = tx;
				Tracker.rotorTyArray( rotorID, sampleIndex ) = ty;
				Tracker.rotorTzArray( rotorID, sampleIndex ) = tz;
			}
			// update the small arrays
			Tracker.rotorFx( rotorID ) = fx; 
			Tracker.rotorFy( rotorID ) = fy;
			Tracker.rotorFz( rotorID ) = fz;
			Tracker.rotorTx( rotorID ) = tx;
			Tracker.rotorTy( rotorID ) = ty;
			Tracker.rotorTz( rotorID ) = tz;
		}
	}
	
	// 3) Walls
	if constexpr (TRACK_WALL_FORCE)
	{
		for ( int wallID = 0; wallID < Tracker.wallCount; wallID++ )
		{
			float fx = 0.f;
			float fy = 0.f; 
			float fz = 0.f; 
			float tx = 0.f;
			float ty = 0.f;
			float tz = 0.f;
			for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
			{
				GridStruct &Grid = grids[level];
				const InfoStruct &Info = Grid.Info;
				if ( (int)Grid.Wall.wallAdjacentCount > 0 )
				{
					const float invIterationSpan = 1.f / (float)Grid.Info.iterationsSinceTrackerReset;
					
					auto indexView = Grid.Wall.indexArray.getConstView();
					auto wallMapView = Grid.Wall.wallMapArray.getConstView();
					auto wallDataView = Grid.Wall.wallDataArray.getConstView();
					auto gxView = Grid.Wall.gxArray.getConstView();
					auto gyView = Grid.Wall.gyArray.getConstView();
					auto gzView = Grid.Wall.gzArray.getConstView();
					auto shifterView = Grid.IJKNBR.shifterArray.getConstView();	
					auto iView = Grid.IJKNBR.iArray.getConstView();
					auto jView = Grid.IJKNBR.jArray.getConstView();
					auto kView = Grid.IJKNBR.kArray.getConstView();
					
					auto fetch = [=] __cuda_callable__ (const int index) -> SixFloatArrayType 
					{
						SixFloatArrayType resultArray;
						resultArray.setValue( 0.f );
						
						const int cell = indexView( index );
						const int wallMap = wallMapView( cell );
						if ( wallMap < 0 ) return resultArray; // this should never happen tho
						
						// read wallData, unpack wallID
						uint32_t wallData = wallDataView( wallMap );
						int wallIDCell; 
						bool interfaceOverlapMarker;
						unpackWallID( wallData, wallIDCell, interfaceOverlapMarker );
						if ( wallIDCell != wallID ) return resultArray; // return zeros if wallID is not correct
						
						// fill iCell, jCell, kCell, this is needed for torque
						int iCell, jCell, kCell;
						getCompressedIJK( cell, iCell, jCell, kCell, shifterView, iView, jView, kView, Info );
						float x, y, z;
						getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );

						const float gx = gxView( wallMap );
						const float gy = gyView( wallMap );
						const float gz = gzView( wallMap );
						
						resultArray[0] = gx;                    				
						resultArray[1] = gy;	
						resultArray[2] = gz;	
						resultArray[3] = + y * gz - z * gy;
						resultArray[4] = + z * gx - x * gz;
						resultArray[5] = + x * gy - y * gx;	
						return resultArray;
					};
					auto reduction = [] __cuda_callable__ (const SixFloatArrayType& a, const SixFloatArrayType& b) 
					{
						SixFloatArrayType resultArray;
						for (int resultIndex = 0; resultIndex < 6; resultIndex++ )
							resultArray[resultIndex] = a[resultIndex] + b[resultIndex];
						return resultArray;
					};
					SixFloatArrayType resultArray = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, Grid.Wall.wallAdjacentCount, 
																									fetch, reduction, SixFloatArrayType(0.0f) );
					// normalize via invIterationSpan
					for ( int i = 0; i < 6; ++i ) resultArray[i] *= invIterationSpan;
					
					// convert to physical units
					convertToPhysicalForce( resultArray[0], resultArray[1], resultArray[2], Info );
					convertToPhysicalForce( resultArray[3], resultArray[4], resultArray[5], Info );
					// at this point torque is in Nmm, convert to Nm
					resultArray[3] *= 0.001f;
					resultArray[4] *= 0.001f;
					resultArray[5] *= 0.001f;
					
					// accumulate
					fx += resultArray[0];
					fy += resultArray[1];
					fz += resultArray[2];
					tx += resultArray[3];
					ty += resultArray[4];
					tz += resultArray[5];
				}
			}
			
			// write the results
			// if tracker period is > 1, also write (period-1) values backward and forward
			// backward, because in those steps we did not launch updateTracker
			// forward, because if on the very last iteration the tracker does not launch,
			// we still want to have some values there (prevent zeros at the very end of the history plot)
			for ( int shift = 1 - TRACKER_PERIOD; shift < TRACKER_PERIOD; shift++ )
			{
				const int sampleIndex = trackerIndex + shift;
				if (sampleIndex < 0 || sampleIndex >= ITERATION_COUNT) continue;
				Tracker.wallFxArray( wallID, sampleIndex ) = fx; 
				Tracker.wallFyArray( wallID, sampleIndex ) = fy;
				Tracker.wallFzArray( wallID, sampleIndex ) = fz;
				Tracker.wallTxArray( wallID, sampleIndex ) = tx;
				Tracker.wallTyArray( wallID, sampleIndex ) = ty;
				Tracker.wallTzArray( wallID, sampleIndex ) = tz;
			}
			// update the small arrays
			Tracker.wallFx( wallID ) = fx; 
			Tracker.wallFy( wallID ) = fy;
			Tracker.wallFz( wallID ) = fz;
			Tracker.wallTx( wallID ) = tx;
			Tracker.wallTy( wallID ) = ty;
			Tracker.wallTz( wallID ) = tz;
		}
	}
	
	// reset all tracker arrays and set iterationsSinceTrackerReset to zero
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
	{
		GridStruct &Grid = grids[level];
		InfoStruct &Info = Grid.Info;
		Info.iterationsSinceTrackerReset = 0;
		for ( int openBCID = 0; openBCID < (int)Grid.openBCs.size(); openBCID++ )
		{
			Grid.openBCs[openBCID].dRhoCumulativeArray.setValue( 0.f );
			Grid.openBCs[openBCID].uNormalCumulativeArray.setValue( 0.f );
		}
		for ( int rotorID = 0; rotorID < (int)Grid.rotors.size(); rotorID++ )
		{
			Grid.rotors[rotorID].gxArray.setValue( 0.f );
			Grid.rotors[rotorID].gyArray.setValue( 0.f );
			Grid.rotors[rotorID].gzArray.setValue( 0.f );
		}
		Grid.Wall.gxArray.setValue( 0.f );
		Grid.Wall.gyArray.setValue( 0.f );
		Grid.Wall.gzArray.setValue( 0.f );
	}
}
