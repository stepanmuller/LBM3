#pragma once

constexpr long long FLOW_REPORT_PIXEL_LIMIT = 16000000;

#include "./NBRFunctions.h"
#include "./esotwistStreamingFunctions.h"
#include "./rotorFunctions.h"
#include "./cellFunctions.h"

// Version with linear interpolation in normal direction for cells that are coarser than image resolution
void getFlowReportGeneral( 	FlowReportStruct &FlowReport, std::vector<GridStruct> &grids, BoundsStruct &Bounds,
							const int &cutIndex, PlaneEnum plane )
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
	TotalGridBounds.xMax = InfoFinest.ox + InfoFinest.res * (InfoFinest.cellCountX - 0.5f);
	TotalGridBounds.yMin = InfoFinest.oy - 0.5f * InfoFinest.res;
	TotalGridBounds.yMax = InfoFinest.oy + InfoFinest.res * (InfoFinest.cellCountY - 0.5f);
	TotalGridBounds.zMin = InfoFinest.oz - 0.5f * InfoFinest.res;
	TotalGridBounds.zMax = InfoFinest.oz + InfoFinest.res * (InfoFinest.cellCountZ - 0.5f);
	
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
	
	while ( pixelCount > FLOW_REPORT_PIXEL_LIMIT && imageLevel > 0 )
	{
		imageLevel--; 
		pixelsHorizontal /= 2; startHorizontal /= 2;
		pixelsVertical /= 2; startVertical /= 2;
		pixelCount = (long long)pixelsHorizontal * pixelsVertical;
	}
	
    if (pixelsHorizontal <= 0 || pixelsVertical <= 0 ||
        pixelCount > FLOW_REPORT_PIXEL_LIMIT)
    {
        std::cerr << "Section cut: empty crop or image exceeds the resolution limit.\n";
        return;
    }

	// 3) Initialize the sectionCut
	SectionCutStruct SectionCut;
	SectionCut.dRhoArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.dRhoArray.setValue( 0.f );
	SectionCut.uxArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.uxArray.setValue( 0.f );
	SectionCut.uyArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.uyArray.setValue( 0.f );
	SectionCut.uzArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.uzArray.setValue( 0.f );
	SectionCut.markerArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.markerArray.setValue( 1.f );
	SectionCut.gridIDArray.setSizes( pixelsVertical, pixelsHorizontal ); SectionCut.gridIDArray.setValue( 0 );
		
	auto dRhoView = SectionCut.dRhoArray.getView();
	auto uxView = SectionCut.uxArray.getView();
	auto uyView = SectionCut.uyArray.getView();
	auto uzView = SectionCut.uzArray.getView();
	auto markerView = SectionCut.markerArray.getView();
	auto gridIDView = SectionCut.gridIDArray.getView();
	
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
		const bool &esotwistFlipper = Grid.Info.esotwistFlipper;
		auto shifterView = Grid.IJKNBR.shifterArray.getConstView();	
		auto iView = Grid.IJKNBR.iArray.getConstView();
		auto jView = Grid.IJKNBR.jArray.getConstView();
		auto kView = Grid.IJKNBR.kArray.getConstView();
		auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
		auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
		auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
		auto wallMapView = Grid.Wall.wallMapArray.getConstView();
		const int rotorCount = Grid.rotorViews.size();
		auto* rotorViews = Grid.rotorViews.data();
		
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
            float marker = selectedSolid ? 1.f : 0.f;
            float dRho = 0.f, ux = 0.f, uy = 0.f, uz = 0.f;
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
                getDRhoUxUyUz(dRho, ux, uy, uz, f);
                
                // get position, we need it for the rotors
				float x, y, z;
				getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
				// here we also need to browse through rotors and find rotor fraction,
				// if marker was zero till here set it to rotor fraction
				// process the rotors
				if ( rotorCount > 0 )
				{
					for (int rotorID = 0; rotorID < rotorCount; rotorID++)
					{
						float rotorFraction;
						getRotorFraction( rotorFraction, x, y, z, Info, rotorViews[rotorID] );
						marker += rotorFraction;
					}
				}
				marker = std::clamp( marker, 0.f, 1.f );
  
                if (blend)
                {
                    getPreCollisionIndex(cellReadIndex, fReadIndex, upperNBR, esotwistFlipper);
                    for (int direction = 0; direction < 27; ++direction)
                        f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
                    float dRhoUpper, uxUpper, uyUpper, uzUpper;
                    getDRhoUxUyUz(dRhoUpper, uxUpper, uyUpper, uzUpper, f);
                    dRho += alpha * (dRhoUpper - dRho);
                    ux += alpha * (uxUpper - ux);
                    uy += alpha * (uyUpper - uy);
                    uz += alpha * (uzUpper - uz);
                    
                    // get position, we need it for the rotors
					float x, y, z;
					getXYZFromIJKCellIndex( iCell + (plane == ZY ? 1 : 0),
											jCell + (plane == ZX ? 1 : 0),
											kCell + (plane == XY ? 1 : 0),
											x, y, z, Info);
					// here we also need to browse through rotors and find rotor fraction,
					// if marker was zero till here set it to rotor fraction
					// process the rotors
					float markerUpper = 0.f;
					if ( rotorCount > 0 )
					{
						for (int rotorID = 0; rotorID < rotorCount; rotorID++)
						{
							float rotorFraction;
							getRotorFraction( rotorFraction, x, y, z, Info, rotorViews[rotorID] );
							markerUpper += rotorFraction;
						}
					}
					markerUpper = std::clamp( markerUpper, 0.f, 1.f );
					marker += alpha * (markerUpper - marker);
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
					dRhoView( y, x ) = dRho;
					uxView( y, x ) = ux;
					uyView( y, x ) = uy;
					uzView( y, x ) = uz;
					markerView( y, x ) = marker;
					gridIDView( y, x ) = Info.gridID;
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda );
	}
	
	// Now, do reduction on the section cut similar to how the tracker does it on open boundaries
	// Use this to fill the FlowReport struct
	// The normal direction here is not outer normal from a boundary, but 
	// it is the direction thats normal AND points in positive direction with respect to the coord system
	// so if the cut plane is submerged inside the grid, take the positive normal
	// apply unit conversion just like in the tracker
	auto fetch = [=] __cuda_callable__ ( const int index ) -> FlowReductionResult
	{
		FlowReductionResult result{};

		const int row = index / pixelsHorizontal;
		const int column = index % pixelsHorizontal;

		// Marker is the solid fraction. Count only fluid pixels.
		if ( !(markerView(row, column) < 0.5f) ) return result;

		const float dRho = dRhoView(row, column);
		const float u = plane == XY ? uzView(row, column)
					  : plane == ZY ? uxView(row, column)
									: uyView(row, column);

		// (1 + dRho) * u, with one final rounding.
		const float rhoU = fmaf(dRho, u, u);

		result.velocity        = u;
		result.densityVelocity = dRho * u;
		result.momentum        = rhoU * TNL::abs(u); // Signed, as in Tracker.
		result.dRho            = dRho;
		result.kinetic         = 0.5f * rhoU * u * u;
		result.cellCount       = 1;

		return result;
	};

	auto reduction = [] __cuda_callable__ ( const FlowReductionResult &a, const FlowReductionResult &b ) -> FlowReductionResult
	{
		FlowReductionResult result;

		result.velocity        = a.velocity        + b.velocity;
		result.densityVelocity = a.densityVelocity + b.densityVelocity;
		result.momentum        = a.momentum        + b.momentum;
		result.dRho            = a.dRho            + b.dRho;
		result.kinetic         = a.kinetic         + b.kinetic;
		result.cellCount       = a.cellCount       + b.cellCount;

		return result;
	};

	const FlowReductionResult totals = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, static_cast<int>(pixelCount), fetch, reduction, FlowReductionResult{} );

	// Also clear any previous report when the section contains no fluid.
	FlowReport = FlowReportStruct{};
	if ( totals.cellCount == 0 ) return;

	// Your grid builder halves res and dtPhys together, so velocity and
	// pressure conversion factors are identical across grid levels.
	// Each section pixel has the area of imageLevel.
	const InfoStruct &reportInfo = grids[imageLevel].Info;

	const float pixelSizeM = reportInfo.res / 1000.f;
	const float pixelAreaM2 = pixelSizeM * pixelSizeM;
	const float velocityScale = pixelSizeM / reportInfo.dtPhys;
	const float inverseCount = 1.f / static_cast<float>(totals.cellCount);

	float pressureScale = 1.f;
	convertToPhysicalPressure( pressureScale, reportInfo );

	const float massFluxScale = RHO_PHYS * pixelAreaM2 * velocityScale;

	// Area averages: equal-area pixels allow division by integer count.
	FlowReport.normalVelocity =	(totals.velocity * inverseCount) * velocityScale;

	FlowReport.pressure = (totals.dRho * inverseCount) * pressureScale;

	// Integrated fluxes. The density correction was accumulated separately.
	FlowReport.massFlow = (totals.velocity + totals.densityVelocity) * massFluxScale;

	FlowReport.momentumThrust =	totals.momentum * (massFluxScale * velocityScale);

	// The same sum(dRho * u) supplies pressure power.
	FlowReport.pressurePower = totals.densityVelocity * (pixelAreaM2 * pressureScale * velocityScale);

	FlowReport.normalKineticPower =	totals.kinetic * (massFluxScale * velocityScale * velocityScale);
}

// Yes bounds, no rotor frame
void getFlowReportXY( FlowReportStruct &FlowReport, std::vector<GridStruct> &grids, BoundsStruct &Bounds, const float &zCut )
{
	float xTemp = 0.f; float yTemp = 0.f; int iCell, jCell, kCell;
	getIJKCellIndexFromXYZ( iCell, jCell, kCell, xTemp, yTemp, zCut, grids[GRID_LEVEL_COUNT-1].Info );
	getFlowReportGeneral( FlowReport, grids, Bounds, kCell, XY );
}
void getFlowReportZY( FlowReportStruct &FlowReport, std::vector<GridStruct> &grids, BoundsStruct &Bounds, const float &xCut )
{
	float zTemp = 0.f; float yTemp = 0.f; int iCell, jCell, kCell;
	getIJKCellIndexFromXYZ( iCell, jCell, kCell, xCut, yTemp, zTemp, grids[GRID_LEVEL_COUNT-1].Info );
	getFlowReportGeneral( FlowReport, grids, Bounds, iCell, ZY );
}
void getFlowReportZX( FlowReportStruct &FlowReport, std::vector<GridStruct> &grids, BoundsStruct &Bounds, const float &yCut )
{
	float xTemp = 0.f; float zTemp = 0.f; int iCell, jCell, kCell;
	getIJKCellIndexFromXYZ( iCell, jCell, kCell, xTemp, yCut, zTemp, grids[GRID_LEVEL_COUNT-1].Info );
	getFlowReportGeneral( FlowReport, grids, Bounds, jCell, ZX );
}
