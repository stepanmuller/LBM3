#pragma once

// Place beside exportSectionCutPlot.h. Uses the same six-field binary format
// and the same Python plotter. Include after the case's physical constants.
#include "exportSectionCutPlot.h"

namespace ToiletPaperPlotDetail
{

enum class Axis { X, Y, Z };
constexpr float twoPi = 6.2831853071795864769f;

inline void exportPlot(
    std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds,
    const RotorInfoStruct &rotorFrameInfo,
    const float radius, const int plotNumber, const Axis axis )
{
    if ( grids.size() < static_cast<size_t>(GRID_LEVEL_COUNT) ||
         GRID_LEVEL_COUNT < 1 || !std::isfinite(radius) || radius <= 0.f )
    {
        std::cerr << "Toilet paper plot: invalid radius or missing grid levels.\n";
        return;
    }

    const InfoStruct &finest = grids[GRID_LEVEL_COUNT - 1].Info;
    if ( !std::isfinite(finest.res) || finest.res <= 0.f ||
         finest.cellCountX <= 0 || finest.cellCountY <= 0 || finest.cellCountZ <= 0 )
    {
        std::cerr << "Toilet paper plot: invalid grid dimensions.\n";
        return;
    }

    // Physical cell faces, not cell centres. The cylinder axis passes through
    // the global origin; the rotor argument changes only the velocity frame.
    BoundsStruct crop;
    crop.xMin = finest.ox - 0.5f * finest.res;
    crop.xMax = finest.ox + (finest.cellCountX - 0.5f) * finest.res;
    crop.yMin = finest.oy - 0.5f * finest.res;
    crop.yMax = finest.oy + (finest.cellCountY - 0.5f) * finest.res;
    crop.zMin = finest.oz - 0.5f * finest.res;
    crop.zMax = finest.oz + (finest.cellCountZ - 0.5f) * finest.res;

    // Bounds affect only the axial interval. All transverse bounds are ignored.
    // Like the section-cut default, two zero axial limits mean the full axis.
    const float requestedMin = axis == Axis::X ? Bounds.xMin : axis == Axis::Y ? Bounds.yMin : Bounds.zMin;
    const float requestedMax = axis == Axis::X ? Bounds.xMax : axis == Axis::Y ? Bounds.yMax : Bounds.zMax;
    const bool useBounds = requestedMin != 0.f || requestedMax != 0.f;
    if ( useBounds )
    {
        if ( !std::isfinite(requestedMin) || !std::isfinite(requestedMax) )
        {
            std::cerr << "Toilet paper plot: non-finite axial bounds.\n";
            return;
        }
        if ( axis == Axis::X )
        {
            crop.xMin = std::max(crop.xMin, requestedMin);
            crop.xMax = std::min(crop.xMax, requestedMax);
        }
        else if ( axis == Axis::Y )
        {
            crop.yMin = std::max(crop.yMin, requestedMin);
            crop.yMax = std::min(crop.yMax, requestedMax);
        }
        else
        {
            crop.zMin = std::max(crop.zMin, requestedMin);
            crop.zMax = std::min(crop.zMax, requestedMax);
        }
    }
    if ( !(crop.xMin < crop.xMax && crop.yMin < crop.yMax && crop.zMin < crop.zMax) )
    {
        std::cerr << "Toilet paper plot: empty crop.\n";
        return;
    }

    const float axisMin = axis == Axis::X ? crop.xMin : axis == Axis::Y ? crop.yMin : crop.zMin;
    const float axisMax = axis == Axis::X ? crop.xMax : axis == Axis::Y ? crop.yMax : crop.zMax;
    const float axisLength = axisMax - axisMin;
    const float circumference = twoPi * radius;

    // Lower only the image resolution if necessary. All grid levels remain
    // eligible as data sources. No averaging is introduced by downsampling.
    int pixelsHorizontal = 0;
    int pixelsVertical = 0;
    for ( int imageLevel = GRID_LEVEL_COUNT - 1; imageLevel >= 0; --imageLevel )
    {
        const float spacing = grids[imageLevel].Info.res;
        if ( !std::isfinite(spacing) || spacing <= 0.f ) continue;
        const float horizontal = std::max(2.f, ceilf(axisLength / spacing));
        const float vertical = std::max(4.f, ceilf(circumference / spacing));
        if ( !std::isfinite(horizontal) || !std::isfinite(vertical) ||
             horizontal > EXPORT_RESOLUTION_PIXEL_LIMIT ||
             vertical > EXPORT_RESOLUTION_PIXEL_LIMIT ) continue;

        const int width = static_cast<int>(horizontal);
        const int height = static_cast<int>(vertical);
        if ( static_cast<long long>(width) * height > EXPORT_RESOLUTION_PIXEL_LIMIT ) continue;
        pixelsHorizontal = width;
        pixelsVertical = height;
        break;
    }
    if ( pixelsHorizontal == 0 )
    {
        std::cerr << "Toilet paper plot: image exceeds the resolution limit.\n";
        return;
    }

    const float axialStep = axisLength / static_cast<float>(pixelsHorizontal);
    const float angleStep = twoPi / static_cast<float>(pixelsVertical);
    const char axisName = axis == Axis::X ? 'X' : axis == Axis::Y ? 'Y' : 'Z';
    std::cout << "Exporting " << axisName << " toilet paper plot " << plotNumber
             << " ... " << std::flush;

    SectionCutStruct SectionCut;
    SectionCut.dRhoArray.setSizes(pixelsVertical, pixelsHorizontal);
    SectionCut.uxArray.setSizes(pixelsVertical, pixelsHorizontal);
    SectionCut.uyArray.setSizes(pixelsVertical, pixelsHorizontal);
    SectionCut.uzArray.setSizes(pixelsVertical, pixelsHorizontal);
    SectionCut.markerArray.setSizes(pixelsVertical, pixelsHorizontal);
    SectionCut.gridIDArray.setSizes(pixelsVertical, pixelsHorizontal);
    SectionCut.dRhoArray.setValue(0.f);
    SectionCut.uxArray.setValue(0.f);
    SectionCut.uyArray.setValue(0.f);
    SectionCut.uzArray.setValue(0.f);
    SectionCut.markerArray.setValue(1.f);
    SectionCut.gridIDArray.setValue(-1);

    auto dRhoView = SectionCut.dRhoArray.getView();
    // These output arrays hold axial, tangential and radial components.
    auto axialView = SectionCut.uxArray.getView();
    auto tangentialView = SectionCut.uyArray.getView();
    auto radialView = SectionCut.uzArray.getView();
    auto markerView = SectionCut.markerArray.getView();
    auto gridIDView = SectionCut.gridIDArray.getView();

    // Keep the painter algorithm: coarse cells paint first, fine cells last.
    // Candidate footprints may overlap, but each pixel is written only by the
    // cell containing its cylinder point. There is no cell lookup or search.
    for ( int level = 0; level < GRID_LEVEL_COUNT; ++level )
    {
        GridStruct &Grid = grids[level];
        const InfoStruct Info = Grid.Info;
        if ( Info.cellCount <= 0 || !std::isfinite(Info.res) || Info.res <= 0.f ) continue;

        auto fView = Grid.fArray.getConstView();
        auto shifterView = Grid.IJKNBR.shifterArray.getConstView();
        auto iView = Grid.IJKNBR.iArray.getConstView();
        auto jView = Grid.IJKNBR.jArray.getConstView();
        auto kView = Grid.IJKNBR.kArray.getConstView();
        auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
        auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
        auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
        auto wallMapView = Grid.Wall.wallMapArray.getConstView();
        const bool esotwistFlipper = Info.esotwistFlipper;
        const int rotorCount = static_cast<int>(Grid.rotorViews.size());
        const auto *rotorViews = Grid.rotorViews.data();

        auto cellLambda = [=] __cuda_callable__ ( const int cell ) mutable
        {
            int iCell, jCell, kCell;
            NBRStruct NBR;
            getCompressedIJKNBR(cell, iCell, jCell, kCell, NBR,
                shifterView, iView, jView, kView, jPlusView, kPlusView, jkPlusView, Info);

            float xCell, yCell, zCell;
            getXYZFromIJKCellIndex(iCell, jCell, kCell, xCell, yCell, zCell, Info);
            const float axial = axis == Axis::X ? xCell : axis == Axis::Y ? yCell : zCell;
            // Right-handed transverse coordinates: (y,z), (z,x), or (x,y).
            const float transverseA = axis == Axis::X ? yCell : axis == Axis::Y ? zCell : xCell;
            const float transverseB = axis == Axis::X ? zCell : axis == Axis::Y ? xCell : yCell;
            const float halfWidth = 0.5f * Info.res;
            if ( axial + halfWidth < axisMin || axial - halfWidth >= axisMax ) return;

            const float distance = sqrtf(transverseA * transverseA + transverseB * transverseB);
            // 1.5*halfWidth encloses the transverse square's half-diagonal.
            const float footprintRadius = 1.5f * halfWidth;
            if ( distance < radius - footprintRadius || distance > radius + footprintRadius ) return;

            // Axial footprint, padded by a pixel to avoid rounding away an edge.
            const float columnFirst = (axial - halfWidth - axisMin) / axialStep - 1.5f;
            const float columnEnd = (axial + halfWidth - axisMin) / axialStep + 1.5f;
            const int firstColumn = static_cast<int>(floorf(TNL::max(0.f,
                TNL::min(static_cast<float>(pixelsHorizontal), columnFirst))));
            const int endColumn = static_cast<int>(ceilf(TNL::max(0.f,
                TNL::min(static_cast<float>(pixelsHorizontal), columnEnd))));
            if ( firstColumn >= endColumn ) return;

            // Angular footprint. Near the axis, a cell may cover the full wrap.
            const float centreAngle = atan2f(transverseB, transverseA);
            const float halfAngle = distance > footprintRadius
                ? asinf(TNL::min(1.f, footprintRadius / distance)) : 0.5f * twoPi;
            int firstRow = static_cast<int>(floorf((centreAngle - halfAngle) / angleStep - 0.5f)) - 1;
            int endRow = static_cast<int>(ceilf((centreAngle + halfAngle) / angleStep - 0.5f)) + 2;
            if ( endRow - firstRow >= pixelsVertical )
            {
                firstRow = 0;
                endRow = pixelsVertical;
            }

            const bool solid = wallMapView(cell) == -3;
            float dRho = 0.f, ux = 0.f, uy = 0.f, uz = 0.f;
            float marker = solid ? 1.f : 0.f;
            if ( !solid )
            {
                float f[27];
                int cellReadIndex[27], fReadIndex[27];
                getPreCollisionIndex(cellReadIndex, fReadIndex, NBR, esotwistFlipper);
                for ( int direction = 0; direction < 27; ++direction )
                    f[direction] = fView(fReadIndex[direction], cellReadIndex[direction]);
                getDRhoUxUyUz(dRho, ux, uy, uz, f);

                // Use the selected cell's centre, not an interpolated sample.
                for ( int rotorID = 0; rotorID < rotorCount; ++rotorID )
                {
                    float fraction;
                    getRotorFraction(fraction, xCell, yCell, zCell, Info, rotorViews[rotorID]);
                    marker += fraction;
                }
                marker = TNL::max(0.f, TNL::min(1.f, marker));

                if ( rotorFrameInfo.rotateAlongX || rotorFrameInfo.rotateAlongY || rotorFrameInfo.rotateAlongZ )
                {
                    float uxRotor, uyRotor, uzRotor;
                    getRotorVelocity(uxRotor, uyRotor, uzRotor, xCell, yCell, zCell, rotorFrameInfo, Info);
                    ux -= uxRotor;
                    uy -= uyRotor;
                    uz -= uzRotor;
                }
            }

            for ( int unwrappedRow = firstRow; unwrappedRow < endRow; ++unwrappedRow )
            {
                int row = unwrappedRow % pixelsVertical;
                if ( row < 0 ) row += pixelsVertical;
                const float angle = (static_cast<float>(row) + 0.5f) * angleStep;
                const float c = cosf(angle);
                const float s = sinf(angle);

                float uAxial, uTangential, uRadial;
                if ( axis == Axis::X )
                {
                    uAxial = ux; uTangential = -uy * s + uz * c; uRadial = uy * c + uz * s;
                }
                else if ( axis == Axis::Y )
                {
                    uAxial = uy; uTangential = -uz * s + ux * c; uRadial = uz * c + ux * s;
                }
                else
                {
                    uAxial = uz; uTangential = -ux * s + uy * c; uRadial = ux * c + uy * s;
                }

                for ( int column = firstColumn; column < endColumn; ++column )
                {
                    const float a = axisMin + (static_cast<float>(column) + 0.5f) * axialStep;
                    float x, y, z;
                    if ( axis == Axis::X )      { x = a;          y = radius * c; z = radius * s; }
                    else if ( axis == Axis::Y ) { x = radius * s; y = a;          z = radius * c; }
                    else                       { x = radius * c; y = radius * s; z = a;          }

                    if ( x < crop.xMin || x >= crop.xMax ||
                         y < crop.yMin || y >= crop.yMax ||
                         z < crop.zMin || z >= crop.zMax ) continue;

                    // The containing cell owns [centre-res/2, centre+res/2).
                    // This also gives shared faces a unique owner, so cells
                    // in this level cannot race to paint the same pixel.
                    const int iPoint = static_cast<int>(floorf((x - Info.ox) / Info.res + 0.5f));
                    const int jPoint = static_cast<int>(floorf((y - Info.oy) / Info.res + 0.5f));
                    const int kPoint = static_cast<int>(floorf((z - Info.oz) / Info.res + 0.5f));
                    if ( iPoint != iCell || jPoint != jCell || kPoint != kCell ) continue;

                    dRhoView(row, column) = dRho;
                    axialView(row, column) = uAxial;
                    tangentialView(row, column) = uTangential;
                    radialView(row, column) = uRadial;
                    markerView(row, column) = marker;
                    gridIDView(row, column) = level;
                }
            }
        };
        TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCount, cellLambda);
    }

    SectionCutStructCPU SectionCutCPU;
    SectionCutCPU.dRhoArray = SectionCut.dRhoArray;
    SectionCutCPU.uxArray = SectionCut.uxArray;
    SectionCutCPU.uyArray = SectionCut.uyArray;
    SectionCutCPU.uzArray = SectionCut.uzArray;
    SectionCutCPU.markerArray = SectionCut.markerArray;
    SectionCutCPU.gridIDArray = SectionCut.gridIDArray;

    FILE *fp = fopen("/dev/shm/sim_data.bin", "wb");
    if ( !fp )
    {
        perror("Toilet paper plot: cannot open /dev/shm/sim_data.bin");
        return;
    }
    const int header[4] = {plotNumber, pixelsVertical, pixelsHorizontal, 6};
    bool writeOK = fwrite(header, sizeof(int), 4, fp) == 4;
    for ( int row = 0; row < pixelsVertical && writeOK; ++row )
    {
        for ( int column = 0; column < pixelsHorizontal; ++column )
        {
            float p = SectionCutCPU.dRhoArray.getElement(row, column);
            float uAxial = SectionCutCPU.uxArray.getElement(row, column);
            float uTangential = SectionCutCPU.uyArray.getElement(row, column);
            float uRadial = SectionCutCPU.uzArray.getElement(row, column);
            const float marker = SectionCutCPU.markerArray.getElement(row, column);
            const int level = SectionCutCPU.gridIDArray.getElement(row, column);
            float resolution = 0.f;
            if ( level >= 0 )
            {
                const InfoStruct &sourceInfo = grids[level].Info;
                convertToPhysicalVelocity(uAxial, uTangential, uRadial, sourceInfo);
                convertToPhysicalPressure(p, sourceInfo);
                resolution = sourceInfo.res;
            }
            const float data[6] = {p, uAxial, uTangential, uRadial, marker, resolution};
            if ( fwrite(data, sizeof(float), 6, fp) != 6 )
            {
                writeOK = false;
                break;
            }
        }
    }
    const bool closeOK = fclose(fp) == 0;
    if ( !writeOK || !closeOK )
        std::cerr << "Toilet paper plot: could not write the complete binary file.\n";
}

} // namespace ToiletPaperPlotDetail

// Public overloads: same optional Bounds / RotorInfo pattern as section cuts.

inline void toiletPaperPlotX( std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds, const RotorInfoStruct &RotorInfo,
    const float &radius, const int &plotNumber )
{
    ToiletPaperPlotDetail::exportPlot(
        grids, Bounds, RotorInfo, radius, plotNumber, ToiletPaperPlotDetail::Axis::X );
}

inline void toiletPaperPlotX( std::vector<GridStruct> &grids,
    const float &radius, const int &plotNumber )
{
    toiletPaperPlotX(grids, BoundsStruct{}, RotorInfoStruct{}, radius, plotNumber);
}

inline void toiletPaperPlotX( std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds, const float &radius, const int &plotNumber )
{
    toiletPaperPlotX(grids, Bounds, RotorInfoStruct{}, radius, plotNumber);
}

inline void toiletPaperPlotX( std::vector<GridStruct> &grids,
    const RotorInfoStruct &RotorInfo, const float &radius, const int &plotNumber )
{
    toiletPaperPlotX(grids, BoundsStruct{}, RotorInfo, radius, plotNumber);
}

inline void toiletPaperPlotY( std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds, const RotorInfoStruct &RotorInfo,
    const float &radius, const int &plotNumber )
{
    ToiletPaperPlotDetail::exportPlot(
        grids, Bounds, RotorInfo, radius, plotNumber, ToiletPaperPlotDetail::Axis::Y );
}

inline void toiletPaperPlotY( std::vector<GridStruct> &grids,
    const float &radius, const int &plotNumber )
{
    toiletPaperPlotY(grids, BoundsStruct{}, RotorInfoStruct{}, radius, plotNumber);
}

inline void toiletPaperPlotY( std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds, const float &radius, const int &plotNumber )
{
    toiletPaperPlotY(grids, Bounds, RotorInfoStruct{}, radius, plotNumber);
}

inline void toiletPaperPlotY( std::vector<GridStruct> &grids,
    const RotorInfoStruct &RotorInfo, const float &radius, const int &plotNumber )
{
    toiletPaperPlotY(grids, BoundsStruct{}, RotorInfo, radius, plotNumber);
}

inline void toiletPaperPlotZ( std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds, const RotorInfoStruct &RotorInfo,
    const float &radius, const int &plotNumber )
{
    ToiletPaperPlotDetail::exportPlot(
        grids, Bounds, RotorInfo, radius, plotNumber, ToiletPaperPlotDetail::Axis::Z );
}

inline void toiletPaperPlotZ( std::vector<GridStruct> &grids,
    const float &radius, const int &plotNumber )
{
    toiletPaperPlotZ(grids, BoundsStruct{}, RotorInfoStruct{}, radius, plotNumber);
}

inline void toiletPaperPlotZ( std::vector<GridStruct> &grids,
    const BoundsStruct &Bounds, const float &radius, const int &plotNumber )
{
    toiletPaperPlotZ(grids, Bounds, RotorInfoStruct{}, radius, plotNumber);
}

inline void toiletPaperPlotZ( std::vector<GridStruct> &grids,
    const RotorInfoStruct &RotorInfo, const float &radius, const int &plotNumber )
{
    toiletPaperPlotZ(grids, BoundsStruct{}, RotorInfo, radius, plotNumber);
}
