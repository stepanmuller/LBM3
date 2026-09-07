#pragma once

constexpr long long EXPORT_RESOLUTION_PIXEL_LIMIT = 16000000;

//#include "./OLDcellFunctions.h"
#include "./NBRFunctions.h"
#include "./esotwistStreamingFunctions.h"

enum PlaneEnum { XY, ZY, ZX };

void exportSectionCutPlotGeneral( std::vector<GridStruct> &grids, const int &cutIndex, const int &plotNumber, PlaneEnum plane )
{
	// 1) find the finest level that fits the resolution limit
	int imageLevel = GRID_LEVEL_COUNT-1; // start with the finest level
	int pixelsHorizontal = 0, pixelsVertical = 0;
	if ( plane == XY ) 		{ pixelsHorizontal = grids[imageLevel].Info.cellCountX; pixelsVertical = grids[imageLevel].Info.cellCountY; }
	else if ( plane == ZY ) { pixelsHorizontal = grids[imageLevel].Info.cellCountZ; pixelsVertical = grids[imageLevel].Info.cellCountY; }
	else 					{ pixelsHorizontal = grids[imageLevel].Info.cellCountZ; pixelsVertical = grids[imageLevel].Info.cellCountX; }
	long long pixelCount = (long long)pixelsHorizontal * (long long)pixelsVertical;
	
	while ( pixelCount > EXPORT_RESOLUTION_PIXEL_LIMIT )
	{
		imageLevel--; 
		pixelsHorizontal /= 2; 
		pixelsVertical /= 2; 
		pixelCount /= 4LL;
	}
	
	// how much coarser the output is compared to the finest grid
	const int downsampleFromFinestLevel = std::pow( 2, ( GRID_LEVEL_COUNT-1 - imageLevel ) ); 
	const int cutIndexImage = cutIndex / downsampleFromFinestLevel;
	
	// 2) Initialize the sectionCut
	SectionCutStruct SectionCut;
	SectionCut.rhoArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.rhoArray.setValue( 1.f );
	SectionCut.uxArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.uxArray.setValue( 0.f );
	SectionCut.uyArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.uyArray.setValue( 0.f );
	SectionCut.uzArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.uzArray.setValue( 0.f );
	SectionCut.markerArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.markerArray.setValue( 1.f );
	SectionCut.gridIDArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.gridIDArray.setValue( 0 );
		
	auto rhoArrayView = SectionCut.rhoArray.getView();
	auto uxArrayView = SectionCut.uxArray.getView();
	auto uyArrayView = SectionCut.uyArray.getView();
	auto uzArrayView = SectionCut.uzArray.getView();
	auto markerArrayView = SectionCut.markerArray.getView();
	auto gridIDArrayView = SectionCut.gridIDArray.getView();
	
	// 3) Loop through all grid levels
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
	{
		GridStruct &Grid = grids[level];
		const InfoStruct &Info = Grid.Info;
		
		int downsample = 1;
		int upsample = 1;
		if ( level > imageLevel ) downsample = std::pow( 2, ( level - imageLevel ) );
		else if ( level < imageLevel ) upsample = std::pow( 2, ( imageLevel - level ) );
				
		auto iView = Grid.IJK.iArray.getConstView();
		auto jView = Grid.IJK.jArray.getConstView();
		auto kView = Grid.IJK.kArray.getConstView();
		
		auto wallMarkerView = Grid.wallMarkerArray.getConstView();
		auto wallIDView = Grid.wallIDArray.getConstView();
		
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			const int iCell = iView[ cell ]; 
			const int jCell = jView[ cell ];
			const int kCell = kView[ cell ];
			
			int iImage = iCell;
			int jImage = jCell;
			int kImage = kCell;
			
			// for levels that are finer than imageLevel, we only accept the "left bottom" cell of the downsample^3 block
			if ( level > imageLevel )
			{
				if ( (iCell % downsample != 0) || (jCell % downsample != 0) || (kCell % downsample != 0) ) return;
				iImage /= downsample;
				jImage /= downsample;
				kImage /= downsample;
			}
			else if ( level < imageLevel )
			{
				iImage *= upsample;
				jImage *= upsample;
				kImage *= upsample;
			}
			
			int indexHorizontal = 0;
			int indexVertical = 0;
			
			if ( plane == XY ) 
			{
				if ( cutIndexImage < kImage || cutIndexImage > kImage + (upsample-1) ) return;
				indexHorizontal = iImage; 
				indexVertical = jImage; 
			}
			else if ( plane == ZY ) 
			{ 
				if ( cutIndexImage < iImage || cutIndexImage > iImage + (upsample-1) ) return; 
				indexVertical = jImage; 
				indexHorizontal = kImage; 
			}
			else // ZX plane
			{ 
				if ( cutIndexImage < jImage || cutIndexImage > jImage + (upsample-1) ) return; 
				indexVertical = iImage; 
				indexHorizontal = kImage; 
			}
			
			// PLACEHOLDER SECTION START
			float rho, ux, uy, uz;
			rho = 1.f; ux = 0.f; uy = 0.f; uz = 0.f; // placeholder values
			const float marker = (float)wallMarkerView(cell);
			const int wallID  = wallIDView( cell );
			if ( wallID >= 0 ) ux = 1.f + (float)wallID;
			// PLACEHOLDER SECTION END
			
			for ( int shiftVertical = 0; shiftVertical < upsample; shiftVertical++ )
			{
				const int y = indexVertical + shiftVertical;
				for ( int shiftHorizontal = 0; shiftHorizontal < upsample; shiftHorizontal++ )
				{
					const int x = indexHorizontal + shiftHorizontal;
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
	int header[4] = {plotNumber, (int)pixelsVertical, (int)pixelsHorizontal, 6};
	fwrite(header, sizeof(int), 4, fp);
	
	for (int indexVertical = 0; indexVertical < pixelsVertical; indexVertical++)
	{
		for (int indexHorizontal = 0; indexHorizontal < pixelsHorizontal; indexHorizontal++)
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
