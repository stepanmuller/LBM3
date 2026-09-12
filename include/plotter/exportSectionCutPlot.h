#pragma once

constexpr long long EXPORT_RESOLUTION_PIXEL_LIMIT = 16000000;

//#include "./OLDcellFunctions.h"
#include "../NBRFunctions.h"
#include "../esotwistStreamingFunctions.h"

enum PlaneEnum { XY, ZY, ZX };

void exportSectionCutPlotGeneral( std::vector<GridStruct> &grids, BoundsStruct &Bounds, const int &cutIndex, const int &plotNumber, PlaneEnum plane )
{
	const InfoStruct InfoFinest = grids[GRID_LEVEL_COUNT-1].Info;
	
	// 1) use bounds
	int iStartFinest = 0; int iEndFinest = InfoFinest.cellCountX;
	int jStartFinest = 0; int jEndFinest = InfoFinest.cellCountY;
	int kStartFinest = 0; int kEndFinest = InfoFinest.cellCountZ;
	
	BoundsStruct TotalGridBounds;
	TotalGridBounds.xMin = InfoFinest.ox - 0.5f * InfoFinest.res;
	TotalGridBounds.xMax = InfoFinest.ox + InfoFinest.res * InfoFinest.cellCountX + 0.5f * InfoFinest.res;
	TotalGridBounds.yMin = InfoFinest.oy - 0.5f * InfoFinest.res;
	TotalGridBounds.yMax = InfoFinest.oy + InfoFinest.res * InfoFinest.cellCountY + 0.5f * InfoFinest.res;
	TotalGridBounds.zMin = InfoFinest.oz - 0.5f * InfoFinest.res;
	TotalGridBounds.zMax = InfoFinest.oz + InfoFinest.res * InfoFinest.cellCountZ + 0.5f * InfoFinest.res;
	
	if ( Bounds.xMin != 0.f || Bounds.xMax != 0.f || Bounds.yMin != 0.f || Bounds.yMax != 0.f || Bounds.zMin != 0.f ||Bounds.zMax != 0.f )
	{
		iStartFinest = TNL::max(0, (int)ceilf(((Bounds.xMin - TotalGridBounds.xMin) / InfoFinest.res)));
		iEndFinest = TNL::min(InfoFinest.cellCountX, InfoFinest.cellCountX - (int)ceilf(((TotalGridBounds.xMax - Bounds.xMax) / InfoFinest.res)));
		jStartFinest = TNL::max(0, (int)ceilf(((Bounds.yMin - TotalGridBounds.yMin) / InfoFinest.res)));
		jEndFinest = TNL::min(InfoFinest.cellCountY, InfoFinest.cellCountY - (int)ceilf(((TotalGridBounds.yMax - Bounds.yMax) / InfoFinest.res)));
		kStartFinest = TNL::max(0, (int)ceilf(((Bounds.zMin - TotalGridBounds.zMin) / InfoFinest.res)));
		kEndFinest = TNL::min(InfoFinest.cellCountZ, InfoFinest.cellCountZ - (int)ceilf(((TotalGridBounds.zMax - Bounds.zMax) / InfoFinest.res)));
	}
	// 2) find the finest level that fits the resolution limit
	int imageLevel = GRID_LEVEL_COUNT-1; // start with the finest level
	int pixelsHorizontal = 0, pixelsVertical = 0, startHorizontal = 0, startVertical = 0;
	if ( plane == XY ) 		{ pixelsHorizontal = (iEndFinest - iStartFinest); pixelsVertical = (jEndFinest - jStartFinest); 
								startHorizontal = iStartFinest; startVertical = jStartFinest; }
	else if ( plane == ZY ) { pixelsHorizontal = (kEndFinest - kStartFinest); pixelsVertical = (jEndFinest - jStartFinest); 
								startHorizontal = kStartFinest; startVertical = jStartFinest; }
	else 					{ pixelsHorizontal = (kEndFinest - kStartFinest); pixelsVertical = (iEndFinest - iStartFinest); 
								startHorizontal = kStartFinest; startVertical = iStartFinest; }
	long long pixelCount = (long long)pixelsHorizontal * (long long)pixelsVertical;
	
	while ( pixelCount > EXPORT_RESOLUTION_PIXEL_LIMIT )
	{
		imageLevel--; 
		pixelsHorizontal /= 2; startHorizontal /= 2;
		pixelsVertical /= 2; startVertical /= 2;
		pixelCount /= 4LL;
	}
	
	// how much coarser the output is compared to the finest grid
	const int downsampleFromFinestLevel = std::pow( 2, ( GRID_LEVEL_COUNT-1 - imageLevel ) ); 
	const int cutIndexImage = cutIndex / downsampleFromFinestLevel;
	
	// 3) Initialize the sectionCut
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
	
	// 4) Loop through all grid levels
	for ( int level = 0; level < GRID_LEVEL_COUNT; level++ )
	{
		GridStruct &Grid = grids[level];
		const InfoStruct &Info = Grid.Info;
		
		int downsample = 1;
		int upsample = 1;
		if ( level > imageLevel ) downsample = std::pow( 2, ( level - imageLevel ) );
		else if ( level < imageLevel ) upsample = std::pow( 2, ( imageLevel - level ) );
			
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
		
		auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
		{
			int iCell, jCell, kCell;
			NBRStruct NBR;
			getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
								shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
								Info );
			
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
			// refuse out of bounds indexes
			if (indexHorizontal + upsample <= startHorizontal || indexHorizontal >= startHorizontal + pixelsHorizontal)	return;

			if (indexVertical + upsample <= startVertical || indexVertical >= startVertical + pixelsVertical) return;
			
			float marker = 0.f; 
			const int wallMap = wallMapView( cell );
			if ( wallMap == -3 ) marker = 1.f;
			
			// here we also need to browse through rotors and find rotor fraction,
			// if marker was zero till here set it to rotor fraction
			
			// read f
			float f[27];
			int cellReadIndex[27];
			int fReadIndex[27];
			getPreCollisionIndex( cellReadIndex, fReadIndex, NBR, esotwistFlipper );
			for ( int direction = 0; direction < 27; direction++ )	f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
			float rho, ux, uy, uz;
			getRhoUxUyUz( rho, ux, uy, uz, f );
			
			for ( int shiftVertical = 0; shiftVertical < upsample; shiftVertical++ )
			{
				const int y = indexVertical + shiftVertical - startVertical;
				if (y < 0 || y >= pixelsVertical) continue;
				for ( int shiftHorizontal = 0; shiftHorizontal < upsample; shiftHorizontal++ )
				{
					const int x = indexHorizontal + shiftHorizontal - startHorizontal;
					if (x < 0 || x >= pixelsHorizontal) continue;
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
			convertToPhysicalVelocity( ux, uy, uz, grids[gridID].Info );
			convertToPhysicalPressure( p, grids[gridID].Info );
			
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
	std::cout << "Exporting XY section cut plot " << plotNumber << " ... " << std::flush;
	BoundsStruct Bounds; exportSectionCutPlotGeneral( grids, Bounds, kCell, plotNumber, XY );
}
void exportSectionCutPlotZY( std::vector<GridStruct> &grids, const int &iCell, const int &plotNumber )
{
	std::cout << "Exporting ZY section cut plot " << plotNumber << " ... " << std::flush;
	BoundsStruct Bounds; exportSectionCutPlotGeneral( grids, Bounds, iCell, plotNumber, ZY );
}
void exportSectionCutPlotZX( std::vector<GridStruct> &grids, const int &jCell, const int &plotNumber )
{
	std::cout << "Exporting ZX section cut plot " << plotNumber << " ... " << std::flush;
	BoundsStruct Bounds; exportSectionCutPlotGeneral( grids, Bounds, jCell, plotNumber, ZX );
}
void exportSectionCutPlotXY( std::vector<GridStruct> &grids, BoundsStruct &Bounds, const int &kCell, const int &plotNumber )
{
	std::cout << "Exporting XY section cut plot " << plotNumber << " ... " << std::flush;
	exportSectionCutPlotGeneral( grids, Bounds, kCell, plotNumber, XY );
}
void exportSectionCutPlotZY( std::vector<GridStruct> &grids, BoundsStruct &Bounds, const int &iCell, const int &plotNumber )
{
	std::cout << "Exporting ZY section cut plot " << plotNumber << " ... " << std::flush;
	exportSectionCutPlotGeneral( grids, Bounds, iCell, plotNumber, ZY );
}
void exportSectionCutPlotZX( std::vector<GridStruct> &grids, BoundsStruct &Bounds, const int &jCell, const int &plotNumber )
{
	std::cout << "Exporting ZX section cut plot " << plotNumber << " ... " << std::flush;
	exportSectionCutPlotGeneral( grids, Bounds, jCell, plotNumber, ZX );
}
