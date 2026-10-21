// Three-dimensional Taylor-Green vortex, periodic in all directions.

constexpr float PI = 3.14159265358979323846f;
constexpr float domainSizePhys = 2.f * PI * 1000.f; // mm, active domain excluding ghosts
constexpr int cellsPerDomain = 256;               	// active cells per axis, must be even

constexpr float reynoldsNumber = 1600.f;
constexpr float uInitial = 0.05f;                 	// lattice velocity amplitude
constexpr float uInitialPhys = 1.f;               	// m/s
constexpr int PLOTTER_PERIOD = 1000;

// set grid level count to 2 to test the interface
constexpr int GRID_LEVEL_COUNT = 2;

// End of case settings. Nothing below needs to be modified.

constexpr int WALL_REFINEMENT_COUNT = 0;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = false;

constexpr float RES_GLOBAL = domainSizePhys / cellsPerDomain; 		// mm
constexpr float referenceLengthPhys = domainSizePhys / (2.f * PI); 	// mm
constexpr float RHO_PHYS = 1.f;                               		// kg/m3
constexpr float NU_PHYS = uInitialPhys * (referenceLengthPhys / 1000.f) / reynoldsNumber; // m2/s
constexpr float DT_PHYS_GLOBAL = (uInitial / uInitialPhys) * (RES_GLOBAL / 1000.f); 	  // s

constexpr float referenceTimePhys = (referenceLengthPhys / 1000.f) / uInitialPhys; 		  // s
constexpr int ITERATION_COUNT = (int)(20.f * referenceTimePhys / DT_PHYS_GLOBAL + 0.5f);

#include "../../include/types.h"
#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( const int& iCell, const int& jCell, const int& kCell,
                                             bool &refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
    getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
    const float quaterDomain = 0.25f * domainSizePhys;
    
    refinementMarker = false;
   
    if ( Info.gridID == 0 )
    {
		if ( x > -quaterDomain && x < quaterDomain && y > -quaterDomain && y < quaterDomain && z > -quaterDomain && z < quaterDomain )
		{
			refinementMarker = true;
		}
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
                                          const InfoStruct& Info )
{
    float x, y, z;
    getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );

    const float xScaled = x / referenceLengthPhys;
    const float yScaled = y / referenceLengthPhys;
    const float zScaled = z / referenceLengthPhys;

    BC.ux =  uInitial * std::sin( xScaled ) * std::cos( yScaled ) * std::cos( zScaled );
    BC.uy = -uInitial * std::cos( xScaled ) * std::sin( yScaled ) * std::cos( zScaled );
    BC.uz = 0.f;

    // Initial pressure through density: p' = dRho / 3 in lattice units.
    BC.dRho = (3.f * uInitial * uInitial / 16.f)
            * (std::cos( 2.f * xScaled ) + std::cos( 2.f * yScaled ))
            * (2.f + std::cos( 2.f * zScaled ));
}

__cuda_callable__ void getOpenBC( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
                                const InfoStruct& Info )
{
    BC.periodic = true;
}

__cuda_callable__ void getLocalBC( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
                                 const InfoStruct& Info )
{
    // Keep the default collision parameters.
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/plotter/exportSectionCutPlot.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
    const float zCut = 0.f;
    exportSectionCutPlotXY( grids, zCut, iterationsFinished );
    if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    // No walls and no rotors.
    std::vector<STLStruct> gridStaticSTLs( 0 );
    std::vector<STLStruct> rotorSTLs( 0 );

    // One ghost cell outside each face of the active domain.
    std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
    const float halfDomain = 0.5f * domainSizePhys;
    const float eps = 1e-6f * domainSizePhys;

    BoundsStruct DomainBounds;
    DomainBounds.xMin = -halfDomain - RES_GLOBAL - eps;
    DomainBounds.xMax =  halfDomain + RES_GLOBAL + eps;
    DomainBounds.yMin = -halfDomain - RES_GLOBAL - eps;
    DomainBounds.yMax =  halfDomain + RES_GLOBAL + eps;
    DomainBounds.zMin = -halfDomain - RES_GLOBAL - eps;
    DomainBounds.zMax =  halfDomain + RES_GLOBAL + eps;

    long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );

    plotGrids( 0, grids );
    
    TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();

    for ( int iterationsFinished = 1; iterationsFinished <= ITERATION_COUNT; iterationsFinished++ )
    {
        updateAllGrids( grids, 0 );

        if ( iterationsFinished % PLOTTER_PERIOD == 0 || iterationsFinished == ITERATION_COUNT )
        {
			lapTimer.stop();
			std::cout << "Iterations finished: " << iterationsFinished << std::endl;
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)fluidUpdatesPerIteration * (float)PLOTTER_PERIOD;
			const float glups = updateCount / lapTime / 1000000000.f;
			std::cout << "GLUPS: " << glups << std::endl;
			
			plotGrids( iterationsFinished, grids );
			
			lapTimer.reset();
			lapTimer.start();
        }
    }

    return EXIT_SUCCESS;
}
