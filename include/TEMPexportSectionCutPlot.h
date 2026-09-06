//#include "./OLDcellFunctions.h"
#include "./NBRFunctions.h"
#include "./esotwistStreamingFunctions.h"

enum PlaneEnum { XY, ZY, ZX };

void exportSectionCutPlotGeneral( std::vector<GridStruct> &grids, const int &cutIndex, const int &plotNumber, PlaneEnum plane )
{
	const int levelCount = grids.size();
	
	// 1. Find the finest level that fits within the memory limit
	int targetLevelCount = levelCount;
	int targetCellCountHorizontal = 0, targetCellCountVertical = 0;
	
	while ( targetLevelCount > 0 ) // Evaluate down to index 0
	{
		InfoStruct Info = grids[targetLevelCount - 1].Info;
		if ( plane == XY ) 		{ targetCellCountHorizontal = Info.cellCountX; targetCellCountVertical = Info.cellCountY; }
		else if ( plane == ZY ) { targetCellCountHorizontal = Info.cellCountZ; targetCellCountVertical = Info.cellCountY; }
		else 					{ targetCellCountHorizontal = Info.cellCountZ; targetCellCountVertical = Info.cellCountX; }
		
		// Use long long to prevent integer overflow on massive grids
		long long dataSize = (long long)targetCellCountHorizontal * targetCellCountVertical;
		
		// Break if it fits in memory OR if we are forced to use the absolute coarsest grid
		if ( dataSize < 20000000 || targetLevelCount == 1 ) break; 
		
		targetLevelCount--;
	}
	
	// How much smaller the output array is compared to the absolute finest grid
	// Using bitshift (1 << X) which is mathematically identical to std::pow(2, X) but faster/safer
	const int targetScale = 1 << (levelCount - targetLevelCount); 

	SectionCutStruct SectionCut;
	SectionCut.rhoArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uxArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uyArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.uzArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.markerArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	SectionCut.gridIDArray.setSizes( targetCellCountVertical, targetCellCountHorizontal );
	
	SectionCut.rhoArray.setValue( 1.f );
	SectionCut.uxArray.setValue( 0.f );
	SectionCut.uyArray.setValue( 0.f );
	SectionCut.uzArray.setValue( 0.f );
	SectionCut.markerArray.setValue( 1 );
	SectionCut.gridIDArray.setValue( 0 );
		
	auto rhoArrayView = SectionCut.rhoArray.getView();
	auto uxArrayView = SectionCut.uxArray.getView();
	auto uyArrayView = SectionCut.uyArray.getView();
	auto uzArrayView = SectionCut.uzArray.getView();
	auto markerArrayView = SectionCut.markerArray.getView();
	auto gridIDArrayView = SectionCut.gridIDArray.getView();
	
	// 2. Loop through ALL grids (none are dropped)
	for ( int level = 0; level < levelCount; level++ )
	{
		GridStruct &Grid = grids[level];
		const InfoStruct &Info = Grid.Info;
		
		// cellScale is relative to the absolute finest grid (levelCount)
		const int cellScale = std::pow(2, (levelCount - Info.gridID - 1) );
		
		auto iView = Grid.IJK.iArray.getConstView();
		auto jView = Grid.IJK.jArray.getConstView();
		auto kView = Grid.IJK.kArray.getConstView();
		
		auto fineToCoarseMarkerView = Grid.fineToCoarseMarkerArray.getConstView();
		auto coarseToFineMarkerView = Grid.coarseToFineMarkerArray.getConstView();
		auto parentMapView = Grid.parentMapArray.getConstView();
		auto wallMarkerView = Grid.wallMarkerArray.getConstView();
		
		auto jPlusView = Grid.NBR.jPlusArray.getConstView();
		auto kPlusView = Grid.NBR.kPlusArray.getConstView();

		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			// Coordinates on the absolute finest grid
			int iCell = iView[ cell ] * cellScale; 
			int jCell = jView[ cell ] * cellScale;
			int kCell = kView[ cell ] * cellScale;
			
			NBRStruct NBR;
			NBR.self = cell;
			NBR.jPlus = jPlusView( cell );
			NBR.kPlus = kPlusView( cell );
			NBR.jkPlus = jPlusView( NBR.kPlus );
			finishNBRPlus( NBR, Info );
			
			int indexHorizontal = 0;
			int indexVertical = 0;
			
			// cutIndex remains fully accurate because it checks against the absolute finest coordinates
			if ( plane == XY ) 
			{
				if ( cutIndex < kCell || cutIndex >= kCell + cellScale ) return;
				indexHorizontal = iCell; 
				indexVertical = jCell; 
			}
			else if ( plane == ZY ) 
			{ 
				if ( cutIndex < iCell || cutIndex >= iCell + cellScale ) return; 
				indexVertical = jCell; 
				indexHorizontal = kCell; 
			}
			else // ZX plane
			{ 
				if ( cutIndex < jCell || cutIndex >= jCell + cellScale ) return; 
				indexVertical = iCell; 
				indexHorizontal = kCell; 
			}
			
			//float f[27];
			//int cellReadIndex[27];
			//int fReadIndex[27];
			//getPreviousPostCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper, Info );
			//for ( int direction = 0; direction < 27; direction++ )	f[direction] = 0.f;
			
			float rho, ux, uy, uz;
			//getRhoUxUyUz(rho, ux, uy, uz, f);
			rho = 1.f; ux = 0.f; uy = 0.f; uz = 0.f;
			//if ( Info.gridID < GRID_LEVEL_COUNT - 1)
			//{
			//	ux = (float)( coarseToFineMarkerView(cell) || fineToCoarseMarkerView(cell));
			//}

			if ( Info.gridID > 0 )
			{
				if ( parentMapView(cell) >= 0 ) ux = (float)Info.gridID;
			}
		
			const float marker = (float)wallMarkerView(cell);
			
			// 3. Mapping coordinates to the scaled-down output array
			int outYStart = indexVertical / targetScale;
			int outXStart = indexHorizontal / targetScale;
			
			// How many pixels this cell spans on the output array (minimum 1 for downsampled fine grids)
			int spanY = max(1, cellScale / targetScale);
			int spanX = max(1, cellScale / targetScale);
			
			for ( int shiftY = 0; shiftY < spanY; shiftY++ )
			{
				int y = outYStart + shiftY;
				if (y >= targetCellCountVertical) continue; // Memory safety bound
				
				for ( int shiftX = 0; shiftX < spanX; shiftX++ )
				{
					int x = outXStart + shiftX;
					if (x >= targetCellCountHorizontal) continue; // Memory safety bound
					
					rhoArrayView( y, x ) = rho;
					uxArrayView( y, x ) = ux;
					uyArrayView( y, x ) = uy;
					uzArrayView( y, x ) = uz;
					markerArrayView( y, x ) = marker;
					gridIDArrayView( y, x ) = Info.gridID;
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	}
	
	SectionCutStructCPU SectionCutCPU;
	SectionCutCPU.rhoArray = SectionCut.rhoArray;
	SectionCutCPU.uxArray = SectionCut.uxArray;
	SectionCutCPU.uyArray = SectionCut.uyArray;
	SectionCutCPU.uzArray = SectionCut.uzArray;
	SectionCutCPU.markerArray = SectionCut.markerArray;
	SectionCutCPU.gridIDArray = SectionCut.gridIDArray;
	
	FILE* fp = fopen("/dev/shm/sim_data.bin", "wb");
	int header[4] = {plotNumber, (int)targetCellCountVertical, (int)targetCellCountHorizontal, 6};
	fwrite(header, sizeof(int), 4, fp);
	
	for (int indexVertical = 0; indexVertical < targetCellCountVertical; indexVertical++)
	{
		for (int indexHorizontal = 0; indexHorizontal < targetCellCountHorizontal; indexHorizontal++)
		{
			float rho = SectionCutCPU.rhoArray.getElement(indexVertical, indexHorizontal);
			float ux = SectionCutCPU.uxArray.getElement(indexVertical, indexHorizontal);
			float uy = SectionCutCPU.uyArray.getElement(indexVertical, indexHorizontal);
			float uz = SectionCutCPU.uzArray.getElement(indexVertical, indexHorizontal);
			float marker = SectionCutCPU.markerArray.getElement(indexVertical, indexHorizontal);
			int gridID = SectionCutCPU.gridIDArray.getElement(indexVertical, indexHorizontal);
			float p = rho;
			
			// Use the actual gridID to scale physical parameters properly
			//convertToPhysicalVelocity( ux, uy, uz, grids[gridID].Info );
			//convertToPhysicalPressure( p, grids[gridID].Info );
			
			float uHorizontal, uVertical, uNormal;
			if ( plane == XY ) 		{ uHorizontal = ux; uVertical = uy; uNormal = uz; }
			else if ( plane == ZY ) { uHorizontal = uz; uVertical = uy; uNormal = ux; }
			else 					{ uHorizontal = uz; uVertical = ux; uNormal = uy; }
			
			float data[6] = {p, uHorizontal, uVertical, uNormal, marker, (float)gridID};
			fwrite(data, sizeof(float), 6, fp);
		}
	}
	fclose(fp);
}

void exportSectionCutPlotXY( std::vector<GridStruct> &grids, const int &kCell, const int &plotNumber )
{
	std::cout << "Exporting XY section cut plot " << plotNumber << std::endl;
	exportSectionCutPlotGeneral( grids, kCell, plotNumber, XY );
}
void exportSectionCutPlotZY( std::vector<GridStruct> &grids, const int &iCell, const int &plotNumber )
{
	std::cout << "Exporting ZY section cut plot " << plotNumber << std::endl;
	exportSectionCutPlotGeneral( grids, iCell, plotNumber, ZY );
}
void exportSectionCutPlotZX( std::vector<GridStruct> &grids, const int &jCell, const int &plotNumber )
{
	std::cout << "Exporting ZX section cut plot " << plotNumber << std::endl;
	exportSectionCutPlotGeneral( grids, jCell, plotNumber, ZX );
}
