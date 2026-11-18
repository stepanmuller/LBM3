#pragma once

#include "types.h"
#include "cellFunctions.h"
#include "esotwistStreamingFunctions.h"
#include <iomanip>
#include <locale>

// Include after the case constants (including RHO_PHYS).
// Uses existing wallMap/wallData flags; no changes to GridStruct are needed.
// interface_overlap is always zero on walls. On fluid cells it reports the
// stored overlap flag (also used for periodic overlap when periodic BCs exist).
// Writes every stored cell from every supplied level, including overlap cells,
// as one VTK point/vertex at its centre in mm.
// Point data: wall, velocity_m_s, pressure_Pa, grid_id,
//             grid_resolution_mm, interface_overlap (all markers are 0/1).
inline void exportVTK(const std::vector<GridStruct>& grids, const std::string& filename)
{
    std::size_t totalCells = 0;
    for (const auto& Grid : grids)
    {
        if (Grid.Info.cellCount < 0)
            throw std::runtime_error("exportVTK: negative cell count.");
        totalCells += static_cast<std::size_t>(Grid.Info.cellCount);
    }

    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Cannot open VTK file: " + filename);
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10);

    // Fields: x, y, z, wall, ux, uy, uz, pressure, interface overlap.
    // Host RAM: 36 bytes per exported cell. GPU scratch: at most 2.25 MiB.
    // Grid ID and resolution are constant per grid and written directly below.
    FloatArray2DTypeCPU data;
    data.setSizes(9, totalCells);
    constexpr int batchSize = 65536;
    FloatArray2DType batchGPU;
    FloatArray2DTypeCPU batchCPU;

    std::size_t offset = 0;
    for (const auto& Grid : grids)
    {
        const InfoStruct Info = Grid.Info;
        const int n = Info.cellCount;
        if (n == 0) continue;

        auto fView = Grid.fArray.getConstView();
        auto shifterView = Grid.IJKNBR.shifterArray.getConstView();
        auto iView = Grid.IJKNBR.iArray.getConstView();
        auto jView = Grid.IJKNBR.jArray.getConstView();
        auto kView = Grid.IJKNBR.kArray.getConstView();
        auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
        auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
        auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();
        auto wallMapView = Grid.Wall.wallMapArray.getConstView();
        auto wallDataView = Grid.Wall.wallDataArray.getConstView();

        for (int begin = 0; begin < n; )
        {
            const int count = std::min(batchSize, n - begin);
            batchGPU.setSizes(9, count);
            auto values = batchGPU.getView();

            auto sample = [=] __cuda_callable__ (const int local) mutable
            {
                const int cell = begin + local;
                int i, j, k;
                NBRStruct NBR;
                getCompressedIJKNBR(cell, i, j, k, NBR,
                                    shifterView, iView, jView, kView,
                                    jPlusView, kPlusView, jkPlusView, Info);

                float x, y, z;
                getXYZFromIJKCellIndex(i, j, k, x, y, z, Info);
                const int wallMap = wallMapView(cell);
                const bool wall = wallMap == -3;
                bool interfaceOverlap = false;
                float ux = 0.f, uy = 0.f, uz = 0.f, pressure = 0.f;
                if (!wall)
                {
                    // Free fluid stores overlap as -2; wall-adjacent fluid
                    // stores it in bit 0 of its packed wallData entry.
                    interfaceOverlap = wallMap == -2 ||
                        (wallMap >= 0 && (wallDataView(wallMap) & 1u) != 0u);
                    float f[27];
                    int cellIndex[27], fIndex[27];
                    // Each grid has its own final Esotwist parity and physical scales.
                    getPreCollisionIndex(cellIndex, fIndex, NBR, Info.esotwistFlipper);
                    for (int q = 0; q < 27; ++q)
                        f[q] = fView(fIndex[q], cellIndex[q]);
                    getDRhoUxUyUz(pressure, ux, uy, uz, f); // pressure is dRho here
                    convertToPhysicalVelocity(ux, uy, uz, Info);
                    convertToPhysicalPressure(pressure, Info);
                }

                values(0, local) = x;
                values(1, local) = y;
                values(2, local) = z;
                values(3, local) = wall ? 1.f : 0.f;
                values(4, local) = ux;
                values(5, local) = uy;
                values(6, local) = uz;
                values(7, local) = pressure;
                values(8, local) = interfaceOverlap ? 1.f : 0.f;
            };
            TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, count, sample);
            batchCPU = batchGPU; // Blocking device-to-host copy via TNL.
            for (int field = 0; field < 9; ++field)
                for (int local = 0; local < count; ++local)
                    data(field, offset + begin + local) = batchCPU(field, local);
            begin += count;
        }
        offset += static_cast<std::size_t>(n);
    }

    out << "# vtk DataFile Version 3.0\n"
        << "LBM snapshot; coordinates in mm; grid count " << grids.size() << '\n'
        << "ASCII\nDATASET POLYDATA\n"
        << "POINTS " << totalCells << " float\n";
    for (std::size_t cell = 0; cell < totalCells; ++cell)
        out << data(0, cell) << ' ' << data(1, cell) << ' ' << data(2, cell) << '\n';

    out << "VERTICES " << totalCells << ' ' << 2ULL * totalCells << '\n';
    for (std::size_t cell = 0; cell < totalCells; ++cell)
        out << "1 " << cell << '\n';

    out << "POINT_DATA " << totalCells << "\nSCALARS wall int 1\nLOOKUP_TABLE default\n";
    for (std::size_t cell = 0; cell < totalCells; ++cell)
        out << static_cast<int>(data(3, cell)) << '\n';

    out << "VECTORS velocity_m_s float\n";
    for (std::size_t cell = 0; cell < totalCells; ++cell)
        out << data(4, cell) << ' ' << data(5, cell) << ' ' << data(6, cell) << '\n';

    out << "SCALARS pressure_Pa float 1\nLOOKUP_TABLE default\n";
    for (std::size_t cell = 0; cell < totalCells; ++cell)
        out << data(7, cell) << '\n';

    out << "SCALARS grid_id int 1\nLOOKUP_TABLE default\n";
    for (const auto& Grid : grids)
        for (int cell = 0; cell < Grid.Info.cellCount; ++cell)
            out << static_cast<int>(Grid.Info.gridID) << '\n';

    out << "SCALARS grid_resolution_mm float 1\nLOOKUP_TABLE default\n";
    for (const auto& Grid : grids)
        for (int cell = 0; cell < Grid.Info.cellCount; ++cell)
            out << Grid.Info.res << '\n';

    out << "SCALARS interface_overlap int 1\nLOOKUP_TABLE default\n";
    for (std::size_t cell = 0; cell < totalCells; ++cell)
        out << static_cast<int>(data(8, cell)) << '\n';

    out.close();
    if (!out)
        throw std::runtime_error("Failed writing VTK file: " + filename);
}
