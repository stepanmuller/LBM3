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
