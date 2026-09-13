#pragma once

constexpr long long EXPORT_RESOLUTION_PIXEL_LIMIT = 16000000;

//#include "./OLDcellFunctions.h"
#include "../NBRFunctions.h"
#include "../esotwistStreamingFunctions.h"

enum PlaneEnum { XY, ZY, ZX };

// One selected grid on the finest grid's full XY pixel canvas.
// State values are replicated, never interpolated.
// kCut is a finest-grid index; InfoFinest supplies its physical coordinates.
inline void exportSectionCutPlotXY(
    GridStruct &Grid, const InfoStruct &InfoFinest,
    const int &kCut, const int &plotNumber )
{
    const InfoStruct Info = Grid.Info;
    if (kCut < 0 || kCut >= InfoFinest.cellCountZ ||
        InfoFinest.res <= 0.f || Info.res <= 0.f ||
        InfoFinest.cellCountX <= 0 || InfoFinest.cellCountY <= 0 ||
        Info.cellCountX <= 0 || Info.cellCountY <= 0 || Info.cellCountZ <= 0)
    {
        std::cerr << "Cell-state plot: invalid cut or grid dimensions.\n";
        return;
    }

    const double zCut = double(InfoFinest.oz) + double(kCut) * InfoFinest.res;
    const double kReal = (zCut - double(Info.oz)) / double(Info.res);
    if (kReal < -0.5 || kReal > double(Info.cellCountZ) - 0.5)
    {
        std::cerr << "Cell-state plot: cut is outside the selected grid.\n";
        return;
    }
    // Nearest cell center, ties to the lower layer. Never interpolate states.
    const int kLower = static_cast<int>(std::floor(kReal));
    const int kSelected = std::clamp(
        kLower + (kReal - kLower > 0.5 ? 1 : 0), 0, Info.cellCountZ - 1);
    const int width = Info.cellCountX;
    const int height = Info.cellCountY;

    FloatArray2DType overlapArray, interfaceArray, solidArray;
    overlapArray.setSizes(height, width); overlapArray.setValue(0.f);
    interfaceArray.setSizes(height, width); interfaceArray.setValue(0.f);
    // Unallocated sparse cells and solid cells keep the existing black mask.
    solidArray.setSizes(height, width); solidArray.setValue(1.f);
    auto overlapView = overlapArray.getView();
    auto interfaceView = interfaceArray.getView();
    auto solidView = solidArray.getView();

    auto shifterView = Grid.IJKNBR.shifterArray.getConstView();
    auto iView = Grid.IJKNBR.iArray.getConstView();
    auto jView = Grid.IJKNBR.jArray.getConstView();
    auto kView = Grid.IJKNBR.kArray.getConstView();
    auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
    auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
    auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
    auto wallMapView = Grid.Wall.wallMapArray.getConstView();
    auto wallDataView = Grid.Wall.wallDataArray.getConstView();

    // Pass 0: overlap/solid state. Pass 1: FTC=-1. Pass 2: CTF=+1.
    // If a cell appears in both lists, the CTF pass takes precedence.
    for (int pass = 0; pass < 3; ++pass)
    {
        const auto indexView = pass == 1
            ? Grid.FineToCoarseInterface.indexArray.getConstView()
            : Grid.CoarseToFineInterface.indexArray.getConstView();
        const int count = pass == 0 ? Info.cellCount
                        : pass == 1 ? Grid.FineToCoarseInterface.interfaceCount
                                    : Grid.CoarseToFineInterface.interfaceCount;
        if (count <= 0) continue;
        auto paint = [=] __cuda_callable__ (const int index) mutable
        {
            const int cell = pass == 0 ? index : indexView(index);
            if (cell < 0 || cell >= Info.cellCount) return;
            int i, j, k;
            NBRStruct NBR;
            getCompressedIJKNBR(cell, i, j, k, NBR,
                shifterView, iView, jView, kView,
                jPlusView, kPlusView, jkPlusView, Info);
            if (k != kSelected || i < 0 || i >= width || j < 0 || j >= height)
                return;
            if (pass != 0)
            {
                interfaceView(j, i) = pass == 1 ? 2.f : 1.f;
                return;
            }

            const int wallMap = wallMapView(cell);
            bool overlap = wallMap == -2;
            if (wallMap >= 0)
            {
                const uint4 data = wallDataView(wallMap);
                const uint32_t packed[4] = {data.x, data.y, data.z, data.w};
                int wallID;
                unpackWallID(packed, wallID, overlap);
            }
            overlapView(j, i) = overlap ? 1.f : 0.f;
            solidView(j, i) = wallMap == -3 ? 1.f : 0.f;
        };
        TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, count, paint);
    }

    FloatArray2DTypeCPU overlapCPU, interfaceCPU, solidCPU;
    overlapCPU = overlapArray;
    interfaceCPU = interfaceArray;
    solidCPU = solidArray;

    FILE *fp = std::fopen("/dev/shm/sim_data.bin", "wb");
    if (!fp)
    {
        std::perror("Cell-state plot: cannot open /dev/shm/sim_data.bin");
        return;
    }
    // Keep GPU state arrays at the selected level's native size. Expand only
    // the exported pixels onto the same physical canvas for every level.
    const int outputWidth = InfoFinest.cellCountX;
    const int outputHeight = InfoFinest.cellCountY;
    const int header[4] = {plotNumber, outputHeight, outputWidth, 6};
    bool ok = std::fwrite(header, sizeof(int), 4, fp) == 4;
    for (int y = 0; y < outputHeight && ok; ++y)
    {
        const double yPhysical = double(InfoFinest.oy) + double(y) * InfoFinest.res;
        const int j = static_cast<int>(std::floor(
            (yPhysical - double(Info.oy)) / double(Info.res) + 0.5));
        for (int x = 0; x < outputWidth && ok; ++x)
        {
            const double xPhysical = double(InfoFinest.ox) + double(x) * InfoFinest.res;
            const int i = static_cast<int>(std::floor(
                (xPhysical - double(Info.ox)) / double(Info.res) + 0.5));
            // Outside this level's extent: retain an empty masked pixel.
            float data[6] = {0.f, 0.f, 0.f, 0.f, 1.f, float(Info.gridID)};
            if (i >= 0 && i < width && j >= 0 && j < height)
            {
                const float overlap = overlapCPU.getElement(j, i);
                data[0] = interfaceCPU.getElement(j, i);
                data[1] = overlap; // planar magnitude: hypot(overlap, 0)
                data[3] = overlap; // normal panel
                data[4] = solidCPU.getElement(j, i);
            }
            ok = std::fwrite(data, sizeof(float), 6, fp) == 6;
        }
    }
    const int closeResult = std::fclose(fp);
    if (!ok || closeResult != 0)
        std::cerr << "Cell-state plot: failed to write complete output.\n";
    else
        std::cout << "Exported cell-state plot " << plotNumber
                  << ": grid " << Info.gridID << ", k=" << kSelected
                  << ", " << outputWidth << " x " << outputHeight << " pixels.\n";
}

/*
void exportSectionCutPlotGeneral( std::vector<GridStruct> &grids, BoundsStruct &Bounds, const int &cutIndex, const int &plotNumber, PlaneEnum plane )
{
	if (grids.size() < static_cast<size_t>(GRID_LEVEL_COUNT))
    {
        std::cerr << "Section cut: not enough grid levels.\n";
        return;
    }
    const InfoStruct InfoFinest = grids[GRID_LEVEL_COUNT-1].Info;
    const int normalCount = plane == XY ? InfoFinest.cellCountZ
                         : plane == ZY ? InfoFinest.cellCountX : InfoFinest.cellCountY;
    if (cutIndex < 0 || cutIndex >= normalCount)
    {
        std::cerr << "Section cut: cut index outside the finest grid.\n";
        return;
    }
    // Same physical cut on every level, irrespective of image downsampling.
    const double normalOriginFinest = plane == XY ? InfoFinest.oz
                                   : plane == ZY ? InfoFinest.ox : InfoFinest.oy;
    const double cutPosition = normalOriginFinest + double(cutIndex) * InfoFinest.res;
	
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
	
	while ( pixelCount > EXPORT_RESOLUTION_PIXEL_LIMIT && imageLevel > 0 )
	{
		imageLevel--; 
		pixelsHorizontal /= 2; startHorizontal /= 2;
		pixelsVertical /= 2; startVertical /= 2;
		pixelCount = (long long)pixelsHorizontal * pixelsVertical;
	}
	
    if (pixelsHorizontal <= 0 || pixelsVertical <= 0 ||
        pixelCount > EXPORT_RESOLUTION_PIXEL_LIMIT)
    {
        std::cerr << "Section cut: empty crop or image exceeds the resolution limit.\n";
        return;
    }

	// 3) Initialize the sectionCut
	SectionCutStruct SectionCut;
	SectionCut.rhoArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.rhoArray.setValue( 0.f );
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
		const InfoStruct Info = Grid.Info;
        if (Info.cellCount <= 0) continue;
        const int levelNormalCount = plane == XY ? Info.cellCountZ
                                   : plane == ZY ? Info.cellCountX : Info.cellCountY;
        if (levelNormalCount <= 0 || Info.res <= 0.f) continue;
        const double normalOrigin = plane == XY ? Info.oz
                                  : plane == ZY ? Info.ox : Info.oy;
        double normalIndex = (cutPosition - normalOrigin) / double(Info.res);
        // A cut between a domain face and the first/last coarse center has
        // no bracketing pair: use that boundary center, without extrapolation.
        normalIndex = std::clamp(normalIndex, 0.0, double(levelNormalCount - 1));
        // Cell states are categorical: choose the nearest center, with ties
        // going to the lower layer, as for the previous solid-mask selection.
        const int lowerNormal = static_cast<int>(floor(normalIndex));
        const int selectedNormal = lowerNormal + (normalIndex - lowerNormal > 0.5 ? 1 : 0);

		int downsample = 1;
		int upsample = 1;
		if ( level > imageLevel ) downsample = std::pow( 2, ( level - imageLevel ) );
		else if ( level < imageLevel ) upsample = std::pow( 2, ( imageLevel - level ) );
			
		auto shifterView = Grid.IJKNBR.shifterArray.getConstView();	
		auto iView = Grid.IJKNBR.iArray.getConstView();
		auto jView = Grid.IJKNBR.jArray.getConstView();
		auto kView = Grid.IJKNBR.kArray.getConstView();
		auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
		auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
		auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
        auto wallMapView = Grid.Wall.wallMapArray.getConstView();
        auto wallDataView = Grid.Wall.wallDataArray.getConstView();

        // Coarse-to-fine order preserves the existing compositing. Within each
        // level paint ordinary states first, then FTC=-1, then CTF=+1.
        // An overlap cell does not erase rho written by a coarser level.
        for (int pass = 0; pass < 3; ++pass)
        {
            const auto indexView = pass == 1
                ? Grid.FineToCoarseInterface.indexArray.getConstView()
                : Grid.CoarseToFineInterface.indexArray.getConstView();
            const int count = pass == 0 ? Info.cellCount
                            : pass == 1 ? Grid.FineToCoarseInterface.interfaceCount
                                        : Grid.CoarseToFineInterface.interfaceCount;
            if (count <= 0) continue;
		
		auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
        {
            const int cell = pass == 0 ? index : indexView(index);
            if (cell < 0 || cell >= Info.cellCount) return;
			int iCell, jCell, kCell;
			NBRStruct NBR;
			getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR, 
								shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView,
								Info );
			
            const int cellNormal = plane == XY ? kCell : plane == ZY ? iCell : jCell;
            if (cellNormal != selectedNormal) return;

            // Preserve existing sampling in the two in-plane directions only.
            // Never downsample or round the normal-axis cut to imageLevel.
            const int cellHorizontal = plane == XY ? iCell : kCell;
            const int cellVertical = plane == ZX ? iCell : jCell;
            if (cellHorizontal % downsample != 0 || cellVertical % downsample != 0)
                return;
            const int indexHorizontal = (cellHorizontal / downsample) * upsample;
            const int indexVertical = (cellVertical / downsample) * upsample;
            if (indexHorizontal + upsample <= startHorizontal ||
                indexHorizontal >= startHorizontal + pixelsHorizontal) return;
            if (indexVertical + upsample <= startVertical ||
                indexVertical >= startVertical + pixelsVertical) return;

            const int wallMap = wallMapView(cell);
            bool overlap = wallMap == -2;
            if (wallMap >= 0)
            {
                const uint4 data = wallDataView(wallMap);
                const uint32_t packed[4] = {data.x, data.y, data.z, data.w};
                int wallID;
                unpackWallID(packed, wallID, overlap);
            }
            const float state = overlap ? 1.f : 0.f;
            const float marker = wallMap == -3 ? 1.f : 0.f;

			for ( int shiftVertical = 0; shiftVertical < upsample; shiftVertical++ )
			{
				const int y = indexVertical + shiftVertical - startVertical;
				if (y < 0 || y >= pixelsVertical) continue;
				for ( int shiftHorizontal = 0; shiftHorizontal < upsample; shiftHorizontal++ )
				{
					const int x = indexHorizontal + shiftHorizontal - startHorizontal;
					if (x < 0 || x >= pixelsHorizontal) continue;
                    if (pass == 0)
                    {
                        if (!overlap) rhoArrayView(y, x) = 0.f;
                        // ux/uy/uz slots hold horizontal/vertical/normal data
                        // directly. Planar magnitude and normal value are 0/1.
                        uxArrayView(y, x) = state;
                        uyArrayView(y, x) = 0.f;
                        uzArrayView(y, x) = state;
                        markerArrayView(y, x) = marker;
                        gridIDArrayView(y, x) = Info.gridID;
                    }
                    else
                    {
                        rhoArrayView(y, x) = pass == 1 ? -1.f : 1.f;
                    }
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, count, cellLambda );
        }
	}
	
	SectionCutStructCPU SectionCutCPU;
	SectionCutCPU.rhoArray = SectionCut.rhoArray;
	SectionCutCPU.uxArray = SectionCut.uxArray;
	SectionCutCPU.uyArray = SectionCut.uyArray;
	SectionCutCPU.uzArray = SectionCut.uzArray;
	SectionCutCPU.markerArray = SectionCut.markerArray;
	SectionCutCPU.gridIDArray = SectionCut.gridIDArray;
	
	FILE* fp = fopen("/dev/shm/sim_data.bin", "wb");
    if (!fp)
    {
        perror("Section cut: cannot open /dev/shm/sim_data.bin");
        return;
    }
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
            // Categorical diagnostic values must not undergo physical-unit
            // conversion. These slots already use plot-plane coordinates.
            float data[6] = {rho, ux, uy, uz, marker, (float)gridID};
			fwrite(data, sizeof(float), 6, fp);
		}
	}
	fclose(fp);
}
*/

/*
void exportSectionCutPlotGeneral( std::vector<GridStruct> &grids, BoundsStruct &Bounds, const int &cutIndex, const int &plotNumber, PlaneEnum plane )
{
	if (grids.size() < static_cast<size_t>(GRID_LEVEL_COUNT))
    {
        std::cerr << "Section cut: not enough grid levels.\n";
        return;
    }
    const InfoStruct InfoFinest = grids[GRID_LEVEL_COUNT-1].Info;
    const int normalCount = plane == XY ? InfoFinest.cellCountZ
                         : plane == ZY ? InfoFinest.cellCountX : InfoFinest.cellCountY;
    if (cutIndex < 0 || cutIndex >= normalCount)
    {
        std::cerr << "Section cut: cut index outside the finest grid.\n";
        return;
    }
    // Same physical cut on every level, irrespective of image downsampling.
    const double normalOriginFinest = plane == XY ? InfoFinest.oz
                                   : plane == ZY ? InfoFinest.ox : InfoFinest.oy;
    const double cutPosition = normalOriginFinest + double(cutIndex) * InfoFinest.res;
	
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
	
	while ( pixelCount > EXPORT_RESOLUTION_PIXEL_LIMIT && imageLevel > 0 )
	{
		imageLevel--; 
		pixelsHorizontal /= 2; startHorizontal /= 2;
		pixelsVertical /= 2; startVertical /= 2;
		pixelCount = (long long)pixelsHorizontal * pixelsVertical;
	}
	
    if (pixelsHorizontal <= 0 || pixelsVertical <= 0 ||
        pixelCount > EXPORT_RESOLUTION_PIXEL_LIMIT)
    {
        std::cerr << "Section cut: empty crop or image exceeds the resolution limit.\n";
        return;
    }

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
		const InfoStruct Info = Grid.Info;
        if (Info.cellCount <= 0) continue;
        const int levelNormalCount = plane == XY ? Info.cellCountZ
                                   : plane == ZY ? Info.cellCountX : Info.cellCountY;
        if (levelNormalCount <= 0 || Info.res <= 0.f) continue;
        const double normalOrigin = plane == XY ? Info.oz
                                  : plane == ZY ? Info.ox : Info.oy;
        double normalIndex = (cutPosition - normalOrigin) / double(Info.res);
        // A cut between a domain face and the first/last coarse center has
        // no bracketing pair: use that boundary center, without extrapolation.
        normalIndex = std::clamp(normalIndex, 0.0, double(levelNormalCount - 1));
        const int lowerNormal = static_cast<int>(floor(normalIndex));
        const float alpha = static_cast<float>(normalIndex - lowerNormal);
        const bool interpolate = alpha > 0.f;
		
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
			
            const int cellNormal = plane == XY ? kCell : plane == ZY ? iCell : jCell;
            if (cellNormal != lowerNormal) return;

            // Preserve existing sampling in the two in-plane directions only.
            // Never downsample or round the normal-axis cut to imageLevel.
            const int cellHorizontal = plane == XY ? iCell : kCell;
            const int cellVertical = plane == ZX ? iCell : jCell;
            if (cellHorizontal % downsample != 0 || cellVertical % downsample != 0)
                return;
            const int indexHorizontal = (cellHorizontal / downsample) * upsample;
            const int indexVertical = (cellVertical / downsample) * upsample;
            if (indexHorizontal + upsample <= startHorizontal ||
                indexHorizontal >= startHorizontal + pixelsHorizontal) return;
            if (indexVertical + upsample <= startVertical ||
                indexVertical >= startVertical + pixelsVertical) return;

            int upperCell = cell;
            NBRStruct upperNBR = NBR;
            if (interpolate)
            {
                upperCell = plane == XY ? NBR.kPlus : plane == ZY ? NBR.iPlus : NBR.jPlus;
                if (upperCell < 0 || upperCell >= Info.cellCount) return;
                int iu, ju, ku;
                getCompressedIJKNBR(upperCell, iu, ju, ku, upperNBR,
                                    shifterView, iView, jView, kView,
                                    jPlusView, kPlusView, jkPlusView, Info);
                // Sparse neighbors may skip cells or wrap. Only accept an
                // actual adjacent center; otherwise retain the coarser result.
                const int expectedI = iCell + (plane == ZY ? 1 : 0);
                const int expectedJ = jCell + (plane == ZX ? 1 : 0);
                const int expectedK = kCell + (plane == XY ? 1 : 0);
                if (iu != expectedI || ju != expectedJ || ku != expectedK) return;
            }

            const bool lowerSolid = wallMapView(cell) == -3;
            const bool upperSolid = wallMapView(upperCell) == -3;
            // Keep the solid mask categorical, using the nearest normal sample.
            const bool selectedSolid = alpha <= 0.5f ? lowerSolid : upperSolid;
            const float marker = selectedSolid ? 1.f : 0.f;
            float rho = 1.f, ux = 0.f, uy = 0.f, uz = 0.f;
            if (!selectedSolid)
            {
                // At a fluid/solid bracket use its fluid sample. Never blend
                // solid populations into fluid. In open fluid use both samples.
                const bool blend = interpolate && !lowerSolid && !upperSolid;
                NBRStruct readNBR = lowerSolid ? upperNBR : NBR;
                float f[27];
                int cellReadIndex[27], fReadIndex[27];
                getPreCollisionIndex(cellReadIndex, fReadIndex, readNBR, esotwistFlipper);
                for (int direction = 0; direction < 27; ++direction)
                    f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
                getRhoUxUyUz(rho, ux, uy, uz, f);
                if (blend)
                {
                    getPreCollisionIndex(cellReadIndex, fReadIndex, upperNBR, esotwistFlipper);
                    for (int direction = 0; direction < 27; ++direction)
                        f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
                    float rhoUpper, uxUpper, uyUpper, uzUpper;
                    getRhoUxUyUz(rhoUpper, uxUpper, uyUpper, uzUpper, f);
                    rho += alpha * (rhoUpper - rho);
                    ux += alpha * (uxUpper - ux);
                    uy += alpha * (uyUpper - uy);
                    uz += alpha * (uzUpper - uz);
                }
            }

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
    if (!fp)
    {
        perror("Section cut: cannot open /dev/shm/sim_data.bin");
        return;
    }
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
*/

/*
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
*/
/*
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
*/
