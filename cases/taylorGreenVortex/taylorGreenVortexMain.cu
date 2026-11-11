// Three-dimensional Taylor-Green vortex, periodic in all directions.
// NASA benchmark: https://ntrs.nasa.gov/citations/20130011044
// Match Re = 1600, Ma = 0.1, and 0 <= t* <= 20.
// This is the isothermal LBM counterpart of NASA's compressible test.

constexpr float PI = 3.14159265358979323846f;

// Choose the grid:
// Uniform coarse: cellsPerDomain = 128, GRID_LEVEL_COUNT = 1
// Uniform fine:   cellsPerDomain = 256, GRID_LEVEL_COUNT = 1
// Refined:        cellsPerDomain = 128, GRID_LEVEL_COUNT = 2
constexpr int cellsPerDomain = 128; // Active coarse cells per axis, excluding ghosts.
constexpr int GRID_LEVEL_COUNT = 2;

constexpr float reynoldsNumber = 1600.f;
constexpr float machNumber = 0.1f;
constexpr float endTimeStar = 20.f;

// Convenient physical units: L = 1 m and U0 = 1 m/s, so t* = time in seconds.
// The active periodic domain is [-pi L, pi L]^3.
constexpr float referenceLengthPhys = 1000.f; // mm
constexpr float uInitialPhys = 1.f;          // m/s
constexpr float RHO_PHYS = 1.f;              // kg/m3
constexpr float domainSizePhys = 2.f * PI * referenceLengthPhys;

// Lattice sound speed is 1/sqrt(3), so U0_lattice = Ma/sqrt(3).
constexpr float uInitial = machNumber * 0.5773502691896258f;
constexpr float RES_GLOBAL = domainSizePhys / cellsPerDomain; // mm
constexpr float NU_PHYS = uInitialPhys * (referenceLengthPhys / 1000.f)
                       / reynoldsNumber; // m2/s
constexpr float DT_PHYS_GLOBAL = (uInitial / uInitialPhys) * (RES_GLOBAL / 1000.f);
constexpr float referenceTimePhys = (referenceLengthPhys / 1000.f) / uInitialPhys;

constexpr int ITERATION_COUNT =
    static_cast<int>(endTimeStar * referenceTimePhys / DT_PHYS_GLOBAL + 0.5f);

// Same physical interval between section plots for the 128 and 256 cases.
constexpr int PLOTTER_PERIOD = 10000 * cellsPerDomain / 128; // only plot the last snapshot
constexpr int TRACKER_PERIOD = 10 * cellsPerDomain / 128;

constexpr int WALL_REFINEMENT_COUNT = 0;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = false;

#include "../../include/types.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier(
    const int& iCell, const int& jCell, const int& kCell,
    bool& refinementMarker, const InfoStruct& Info)
{
    refinementMarker = false;
    if (GRID_LEVEL_COUNT == 1 || Info.gridID != 0) return;

    float x, y, z;
    getXYZFromIJKCellIndex(iCell, jCell, kCell, x, y, z, Info);

    // Keep the existing central refinement cube: [-pi L/2, pi L/2]^3.
    const float refinementHalfWidth = 0.25f * domainSizePhys;
    refinementMarker =
        x > -refinementHalfWidth && x < refinementHalfWidth &&
        y > -refinementHalfWidth && y < refinementHalfWidth &&
        z > -refinementHalfWidth && z < refinementHalfWidth;
}

__cuda_callable__ void getInitialCondition(
    BCStruct& BC, const int& iCell, const int& jCell, const int& kCell,
    const InfoStruct& Info)
{
    float x, y, z;
    getXYZFromIJKCellIndex(iCell, jCell, kCell, x, y, z, Info);

    const float X = x / referenceLengthPhys;
    const float Y = y / referenceLengthPhys;
    const float Z = z / referenceLengthPhys;

    BC.ux =  uInitial * std::sin(X) * std::cos(Y) * std::cos(Z);
    BC.uy = -uInitial * std::cos(X) * std::sin(Y) * std::cos(Z);
    BC.uz = 0.f;

    // NASA pressure perturbation, represented through p' = dRho/3 in LBM.
    BC.dRho = (3.f * uInitial * uInitial / 16.f)
            * (std::cos(2.f * X) + std::cos(2.f * Y))
            * (2.f + std::cos(2.f * Z));
}

__cuda_callable__ void getOpenBC(
    BCStruct& BC, const int& iCell, const int& jCell, const int& kCell,
    const InfoStruct& Info)
{
    BC.periodic = true;
}

__cuda_callable__ void getLocalBC(
    BCStruct& BC, const int& iCell, const int& jCell, const int& kCell,
    const InfoStruct& Info)
{
    // Keep the same default collision parameters for all five runs.
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/plotter/exportSectionCutPlot.h"

void plotGrids(const int& iterationsFinished, std::vector<GridStruct>& grids)
{
    const float zCut = 0.f;
    exportSectionCutPlotXY(grids, zCut, iterationsFinished);
    if (system("python3 plotGridsFullCustom.py") != 0) {}
    std::cout << std::endl;
}

float getKineticEnergy(const std::vector<GridStruct>& grids)
{
    double energy = 0.0;

    for (const GridStruct& Grid : grids)
    {
        const InfoStruct Info = Grid.Info;
        auto fView = Grid.fArray.getConstView();
        auto wallMapView = Grid.Wall.wallMapArray.getConstView();
        auto wallDataView = Grid.Wall.wallDataArray.getConstView();

        auto shifterView = Grid.IJKNBR.shifterArray.getConstView();
        auto iView = Grid.IJKNBR.iArray.getConstView();
        auto jView = Grid.IJKNBR.jArray.getConstView();
        auto kView = Grid.IJKNBR.kArray.getConstView();
        auto jPlusView = Grid.IJKNBR.jPlusArray.getConstView();
        auto kPlusView = Grid.IJKNBR.kPlusArray.getConstView();
        auto jkPlusView = Grid.IJKNBR.jkPlusArray.getConstView();

        auto energyAtCell =
            [=] __cuda_callable__ (const int cell) mutable -> double
        {
            // Exclude walls, interface duplicates and periodic ghosts.
            const int wallMap = wallMapView(cell);
            if (wallMap == -3 || wallMap == -2) return 0.0;
            if (wallMap >= 0 && (wallDataView(wallMap) & 1u)) return 0.0;

            int iCell, jCell, kCell;
            NBRStruct NBR;
            getCompressedIJKNBR(cell, iCell, jCell, kCell, NBR,
                shifterView, iView, jView, kView,
                jPlusView, kPlusView, jkPlusView, Info);

            // Read populations using the current esotwist state.
            float f[27];
            int cellReadIndex[27], fReadIndex[27];
            getPreCollisionIndex(cellReadIndex, fReadIndex,
                                 NBR, Info.esotwistFlipper);

            for (int q = 0; q < 27; ++q)
                f[q] = fView(fReadIndex[q], cellReadIndex[q]);

            float dRho, ux, uy, uz;
            getDRhoUxUyUz(dRho, ux, uy, uz, f);

            // Here 1 + dRho is rho / rho0.
            return 0.5 * (1.0 + double(dRho))
                 * (double(ux)*ux + double(uy)*uy + double(uz)*uz);
        };

        const double sum = TNL::Algorithms::reduce<TNL::Devices::Cuda>(
            0, Info.cellCount, energyAtCell, TNL::Plus{}, 0.0);

        const double dx = double(Info.res) / 1000.0; // m

        // Converts lattice velocity into physical velocity divided by U0.
        const double velocityScale =
            dx / (double(Info.dtPhys) * uInitialPhys);

        energy += sum * dx*dx*dx * velocityScale*velocityScale;
    }

    const double domainLength = double(domainSizePhys) / 1000.0; // m
    return float(energy / (domainLength*domainLength*domainLength));
}

int main()
{
    std::vector<STLStruct> gridStaticSTLs(0);
    std::vector<STLStruct> rotorSTLs(0);
    std::vector<GridStruct> grids(GRID_LEVEL_COUNT);

    // One coarse ghost cell outside each face; retain the existing periodic setup.
    const float halfDomain = 0.5f * domainSizePhys;
    const float eps = 1e-6f * domainSizePhys;

    BoundsStruct DomainBounds;
    DomainBounds.xMin = -halfDomain - RES_GLOBAL - eps;
    DomainBounds.xMax =  halfDomain + RES_GLOBAL + eps;
    DomainBounds.yMin = -halfDomain - RES_GLOBAL - eps;
    DomainBounds.yMax =  halfDomain + RES_GLOBAL + eps;
    DomainBounds.zMin = -halfDomain - RES_GLOBAL - eps;
    DomainBounds.zMax =  halfDomain + RES_GLOBAL + eps;

    const long long fluidUpdatesPerIteration =
        buildGrids(grids, gridStaticSTLs, rotorSTLs, DomainBounds);

    std::cout << "Re = " << reynoldsNumber << ", Ma = " << machNumber
              << ", coarse cells per axis = " << cellsPerDomain
              << ", grid levels = " << GRID_LEVEL_COUNT << std::endl;
    std::cout << "Time step t* = " << DT_PHYS_GLOBAL / referenceTimePhys
              << ", total iterations = " << ITERATION_COUNT << std::endl;

    //plotGrids(0, grids);
	
	std::vector<std::pair<double, float>> kineticEnergyHistory;
	kineticEnergyHistory.emplace_back(0.0, getKineticEnergy(grids));
    
    TNL::Timer lapTimer;
    lapTimer.reset();
    lapTimer.start();
    int previousPlotIteration = 0;

    for (int iterationsFinished = 1; iterationsFinished <= ITERATION_COUNT; ++iterationsFinished)
    {
        updateAllGrids(grids, 0);
		
		if (iterationsFinished % TRACKER_PERIOD == 0 || iterationsFinished == ITERATION_COUNT)
		{
			const double timeStar = double(iterationsFinished) * DT_PHYS_GLOBAL / referenceTimePhys;
			const float kineticEnergy = getKineticEnergy(grids);
			kineticEnergyHistory.emplace_back(timeStar, kineticEnergy);
			//std::cout << timeStar << ";" << kineticEnergy << '\n';
		}
		
        if (iterationsFinished % PLOTTER_PERIOD == 0 || iterationsFinished == ITERATION_COUNT)
        {
            lapTimer.stop();

            const double timeStar = double(iterationsFinished) * DT_PHYS_GLOBAL / referenceTimePhys;
            const int iterationsThisLap = iterationsFinished - previousPlotIteration;
            const double updateCount = double(fluidUpdatesPerIteration) * iterationsThisLap;
            const double glups = updateCount / lapTimer.getRealTime() / 1e9;

            std::cout << "Iterations finished: " << iterationsFinished
                      << ", t* = " << timeStar << std::endl;
            std::cout << "GLUPS: " << glups << std::endl;

            plotGrids(iterationsFinished, grids);

            previousPlotIteration = iterationsFinished;
            lapTimer.reset();
            lapTimer.start();
        }
    }

	std::ofstream energyFile("results/kineticEnergy.txt");
	energyFile.precision(9);
	energyFile << "time star [1];Normalized kinetic energy [1]\n";
	for (const auto& [timeStar, kineticEnergy] : kineticEnergyHistory) energyFile << timeStar << ";" << kineticEnergy << '\n';
	
    return EXIT_SUCCESS;
}
