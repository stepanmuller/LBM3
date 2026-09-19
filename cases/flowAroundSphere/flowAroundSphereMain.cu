// This is a close copy of the drag crisis case shown in
// Martin Geier, Andrea Pasquali, Martin Schönherr:
// Parametrization of the cumulant lattice Boltzmann method for fourth order
// accurate diffusion Part II: application to flow around a sphere at drag crisis
// 2017

// Geier 2017 coarse settings
constexpr int cellsPerSphereDiameter = 410;
constexpr float uxInlet = 0.015625; 
constexpr int ITERATION_COUNT = 60000;

// Geier 2017 medium settings
//constexpr int cellsPerSphereDiameter = 512;
//constexpr float uxInlet = 0.0125f; 
//constexpr int ITERATION_COUNT = 60000;

// Geier 2017 fine settings
//constexpr int cellsPerSphereDiameter = 640;
//constexpr float uxInlet = 0.01f; 
//constexpr int ITERATION_COUNT = 90000;

constexpr float reynoldsNumber = 100000.f;
constexpr int PLOTTER_PERIOD = 500;

// End of case settings. Nothing below needs to be modified.

constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float sphereDiameterPhys = 1000.f;											// mm
constexpr float uxInletPhys = uxInlet; 													// m/s, physical velocity set to same as LBM velocity

constexpr int GRID_LEVEL_COUNT = 6;
constexpr int WALL_REFINEMENT_COUNT = 6;
constexpr float RES_GLOBAL = (sphereDiameterPhys / cellsPerSphereDiameter) * (1u << (GRID_LEVEL_COUNT - 1));

constexpr float NU_PHYS = uxInletPhys * (sphereDiameterPhys / 1000.f) / reynoldsNumber;	// m2/s
constexpr float RHO_PHYS = 1.225f;														// kg/m3 air
constexpr float DT_PHYS_GLOBAL = (uxInlet / uxInletPhys) * (RES_GLOBAL/1000); 			// s


#include "../../include/types.h"

std::string STLPathSphere = "sphere_D=1000mm.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r2 = y * y + z * z;
	const float xTransition = 1300.f;
	float xStart = 0.f; float xEnd = 0.f; float rLimit = 0.f;
	if ( Info.gridID == 0 )
	{
		xStart = -1300.f;
		xEnd = 3300.f;
		rLimit = 1500.f;
	}
	else if ( Info.gridID == 1 )
	{
		xStart = -950.f;
		xEnd = 3000.f;
		rLimit = 1100.f;
	}
	else if ( Info.gridID == 2 )
	{
		xStart = -750.f;
		xEnd = 2900.f;
		rLimit = 950.f;
	}
	else if ( Info.gridID == 3 )
	{
		xStart = -640.f;
		xEnd = 1300.f;
		rLimit = 850.f;
	}
	if ( Info.gridID < 4 )
	{
		refinementMarker = false;
		float r2Limit = rLimit * rLimit;
		if (x < xTransition)
		{
			const float s = (x - xTransition) / (xTransition - xStart);
			r2Limit *= 1.f - s * s;
		}
		if ( r2 < r2Limit && x > xStart && x < xEnd ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	//BC.ux = uxInlet; // Geier seems to use zero initial condition based on the Figure 6
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( iCell == Info.cellCountX-1 && jCell != 0 && jCell != Info.cellCountY-1 && kCell != 0 && kCell != Info.cellCountZ-1  ) // Outlet
	{
		BC.dirichletRho = true;
		BC.dRho = 0.f;
		BC.openBCID = 1;
		BC.rhoReflectionTolerance = 8e-4f * uxInlet * uxInlet; 
		// first scale because timestep gets smaller and so per one second we would get more reflection, 
		// second scale because as LBM Mach number gets smaller, values of dRho get smaller
	}
	else if ( iCell == 0 && jCell != 0 && jCell != Info.cellCountY-1 && kCell != 0 && kCell != Info.cellCountZ-1  ) // Inlet
	{
		BC.dirichletU = true;
		BC.ux = uxInlet;
		BC.uy = 0.f;
		BC.uz = 0.f;
		BC.rhoReflectionTolerance = 1.f;
		BC.openBCID = 0;
	}
	else // Every other boundary cell is strict dirichlet velocity
	{
		BC.dirichletU = true;
		BC.ux = uxInlet;
		BC.uy = 0.f;
		BC.uz = 0.f;
		BC.rhoReflectionTolerance = 1.f;
		BC.openBCID = 2;
	}
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	if ( BC.wallID == 0 ) 
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( Info.gridID == 0 ) BC.collisionLimiter = 0.f; // use more stable K15 collision for grid 0 where open BC happen
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	// XY section cut
	const int kCut = grids[ GRID_LEVEL_COUNT-1 ].Info.cellCountZ / 2;
	exportSectionCutPlotXY( grids, kCut, iterationsFinished );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	// Detail 1
	if ( GRID_LEVEL_COUNT > 1 )
	{
		exportSectionCutPlotXY( grids, grids[1].Info.Bounds, kCut, iterationsFinished + 1 );
		if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathSphere );
	
	std::vector<STLStruct> rotorSTLs( 0 ); // there are no rotors
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds.xMin = - 2000.f;
	DomainBounds.xMax =   9000.f;
	DomainBounds.yMin = - 5500.f;
	DomainBounds.yMax =   5500.f;
	DomainBounds.zMin = - 5500.f;
	DomainBounds.zMax =   5500.f;
	
	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );
	
	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = true;
	Tracker.customNames = { "Sphere Drag Coefficient" };
	Tracker.customUnits = { "[1]" };
	Tracker.averagePercent = 38.f; // Geier uses around 38% averaging interval
	initializeTracker( Tracker, grids );
	
	plotGrids( 0, grids );
	
	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();
	
	for ( int iterationsFinished = 1; iterationsFinished <= ITERATION_COUNT; iterationsFinished++ )
	{
		updateAllGrids( grids, 0 );
		
		if ( iterationsFinished % TRACKER_PERIOD == 0 )
		{
			updateTracker( Tracker, grids );
			const float drag = - Tracker.wallFx[0];
			const float dragCoefficient = (8.f * drag) / (RHO_PHYS * uxInletPhys * uxInletPhys * 3.14159f * (sphereDiameterPhys / 1000.f) * (sphereDiameterPhys / 1000.f));
			trackCustomVariables( Tracker, { dragCoefficient });
		}
		
		if ( iterationsFinished % PLOTTER_PERIOD == 0 )
		{
			lapTimer.stop();
			std::cout << "Iterations finished: " << iterationsFinished << std::endl;
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)fluidUpdatesPerIteration * (float)PLOTTER_PERIOD;
			const float glups = updateCount / lapTime / 1000000000.f;
			std::cout << "GLUPS: " << glups << std::endl;
			
			plotTracker( Tracker );
			
			plotGrids( iterationsFinished, grids );
			
			lapTimer.reset();
			lapTimer.start();
		}
	}
	
	return EXIT_SUCCESS;
}
