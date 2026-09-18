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
	// set sizes
	Tracker.volumetricFlowArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
	Tracker.massFlowArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
	Tracker.pressureArray.setSizes( Tracker.openBCCount, ITERATION_COUNT );
	Tracker.pressurePowerArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
	Tracker.momentumThrustArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
	Tracker.normalKineticPowerArray.setSizes( Tracker.openBCCount, ITERATION_COUNT ); 
	
	Tracker.wallFxArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
	Tracker.wallFyArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
	Tracker.wallFzArray.setSizes( Tracker.wallCount, ITERATION_COUNT );
	Tracker.wallTxArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
	Tracker.wallTyArray.setSizes( Tracker.wallCount, ITERATION_COUNT ); 
	Tracker.wallTzArray.setSizes( Tracker.wallCount, ITERATION_COUNT );
	
	Tracker.rotorFxArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
	Tracker.rotorFyArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
	Tracker.rotorFzArray.setSizes( Tracker.rotorCount, ITERATION_COUNT );
	Tracker.rotorTxArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
	Tracker.rotorTyArray.setSizes( Tracker.rotorCount, ITERATION_COUNT ); 
	Tracker.rotorTzArray.setSizes( Tracker.rotorCount, ITERATION_COUNT );
	
	std::cout << "Done" << std::endl;
	std::cout << std::endl;
}

void updateTracker( const int &trackerIndex, TrackerStruct &Tracker, std::vector<GridStruct>& grids )
{
	// 1) Open BC
	for ( int openBCID = 0; openBCID < Tracker.openBCCount; openBCID++ )
	{
		float aream2 = 0.f;
		float volumetricFlow = 0.f; 
		float massFlow = 0.f; 
		float pressure = 0.f;
		float pressurePower = 0.f;
		float momentumThrust = 0.f;
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
					resultArray[2] = dRho;                          			// pressure integral
					resultArray[3] = dRho * uNormal;                     		// pressurePower
					resultArray[4] = rho * uNormal * TNL::abs(uNormal);      	// momentumThrust
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
				convertToPhysicalPressure( resultArray[2], Info );
				resultArray[2] *= cellAream2;
				convertToPhysicalPressure( resultArray[3], Info );
				resultArray[3] *= cellAream2 * velocityMultiplier;
				resultArray[4] *= cellAream2 * velocityMultiplier * velocityMultiplier * RHO_PHYS;
				resultArray[5] *= cellAream2 * velocityMultiplier * velocityMultiplier * velocityMultiplier * RHO_PHYS;
				
				// accumulate
				aream2 += Grid.openBCs[openBCID].trackFlowCount * cellAream2;
				volumetricFlow     += resultArray[0];
				massFlow           += resultArray[1];
				pressure           += resultArray[2];
				pressurePower      += resultArray[3];
				momentumThrust     += resultArray[4];
				normalKineticPower += resultArray[5];
			}
		}
		// we have a pressure integral now, divide to get area averaged pressure
		pressure /= aream2;
	}
}
