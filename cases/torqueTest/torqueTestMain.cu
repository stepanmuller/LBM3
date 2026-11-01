// Reynolds number based on inner-wall speed and annular gap:
// REYNOLDS = |Omega| * Ri * (Ro - Ri) / nu
constexpr float REYNOLDS = 10000.f;

// Geometry: millimetres, matching the STLs.
// AXIAL_LENGTH_PHYS sets the active periodic length, excluding ghost planes.
// The STLs must extend beyond both ends of this cropped domain.
constexpr float INNER_RADIUS_MM = 10.f;
constexpr float OUTER_RADIUS_MM = 30.f;
constexpr float REFINEMENT_RADIUS = 12.f;

constexpr float INNER_RADIUS_PHYS = INNER_RADIUS_MM / 1000.f;
constexpr float OUTER_RADIUS_PHYS = OUTER_RADIUS_MM / 1000.f;
constexpr float AXIAL_LENGTH_PHYS = 0.010f;
constexpr float GAP_PHYS = OUTER_RADIUS_PHYS - INNER_RADIUS_PHYS;

// Air
constexpr float RHO_PHYS = 1.2f;       // kg/m^3
constexpr float NU_PHYS = 1.5e-5f;       // m^2/s

// Rotation: use a positive speed magnitude for unit conversion.
// Change ROTATION_SIGN to +1.f to reverse rotation.
constexpr float ROTATION_SIGN = -1.f;
constexpr float OMEGA_MAG_PHYS =
    REYNOLDS * NU_PHYS / (INNER_RADIUS_PHYS * GAP_PHYS);

constexpr float radiansPerSecond = ROTATION_SIGN * OMEGA_MAG_PHYS;
constexpr float uNominalPhys = OMEGA_MAG_PHYS * INNER_RADIUS_PHYS;

// Coarse-grid spacing [mm] and inner-wall lattice speed.
constexpr float RES_GLOBAL = 0.2f;
constexpr float uNominal = 0.02f;

constexpr float DT_PHYS_GLOBAL =
    (uNominal / uNominalPhys) * (RES_GLOBAL / 1000.f);

// One level for the uniform-grid reference.
// Keep one level: the current periodic builder gives different axial periods
// on refined levels. The length check below rejects that inconsistent setup.
constexpr int GRID_LEVEL_COUNT = 1;

constexpr int ITERATION_COUNT = 10000;
constexpr int PLOTTER_PERIOD = 10000;

constexpr int WALL_REFINEMENT_COUNT = 3;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = true;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = false;

// Analytical steady torque magnitude for the periodic annulus.
// This includes cylindrical sidewall torque only; there are no end caps.
constexpr float PI = 3.14159265358979323846f;
constexpr float EXPECTED_TORQUE_MAG_PHYS =
    4.f * PI * RHO_PHYS * NU_PHYS * AXIAL_LENGTH_PHYS
    * OMEGA_MAG_PHYS
    * INNER_RADIUS_PHYS * INNER_RADIUS_PHYS
    * OUTER_RADIUS_PHYS * OUTER_RADIUS_PHYS
    / (OUTER_RADIUS_PHYS * OUTER_RADIUS_PHYS
       - INNER_RADIUS_PHYS * INNER_RADIUS_PHYS);

#include "../../include/types.h"

std::string STLPathStator = "statorSTL.STL"; 
std::string STLPathRotor = "rotorSTL.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( 	const int& iCell, const int& jCell, const int& kCell, 
												bool & refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float rz = std::sqrt( x*x + y*y );
	if ( Info.gridID == 0 )
	{
		refinementMarker = false;
		if ( rz < REFINEMENT_RADIUS ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
											const InfoStruct& Info )
{
	return; // this leaves default zero velocity, zero pressure
}

__cuda_callable__ void getOpenBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	// Periodic along the cylinder axis; the radial boundaries are STL walls.
	BC.periodic = ( kCell == 0 || kCell == Info.cellCountZ - 1 );
}

__cuda_callable__ void getLocalBC( 	BCStruct &BC, const int& iCell, const int& jCell, const int& kCell, 
									const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	// Rigid rotation, converted from mm and rad/s to lattice velocity.
	const float rotationScale = radiansPerSecond * Info.dtPhys / Info.res;
	if ( BC.wallID == 0 ) // stator
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	if ( BC.wallID == 1 ) // rotor
	{
		BC.ux = - rotationScale * y;
		BC.uy = rotationScale * x;
		BC.uz = 0.f;
	}
	BC.collisionLimiter = 0.f; // use K15
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	// Axial section through the annulus
	float xCut = 0.f;
	exportSectionCutPlotZY( grids, xCut, iterationsFinished + 0 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	// Cross-section normal to the cylinder axis
	float zCut = 0.f;
	exportSectionCutPlotXY( grids, zCut, iterationsFinished + 1 );
	if (system("python3 ../../include/plotter/plotGridsFull.py") != 0) {}
	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 2 );
	readSTL( gridStaticSTLs[0], STLPathStator );
	readSTL( gridStaticSTLs[1], STLPathRotor );
	
	std::vector<STLStruct> rotorSTLs( 0 );
	
	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	// The periodic builder copies the first/last planes from the opposite
	// interior planes. Add one ghost plane per end to the requested length.
	const float axialLengthMM = AXIAL_LENGTH_PHYS * 1000.f;
	const float halfLength = 0.5f * axialLengthMM;
	const float boundsEpsilon = 1e-6f * axialLengthMM;
	DomainBounds.zMin = -halfLength - RES_GLOBAL - boundsEpsilon;
	DomainBounds.zMax =  halfLength + RES_GLOBAL + boundsEpsilon;

	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );

	std::cout << "Taylor-Couette: Re = " << REYNOLDS
	          << ", periodic length = " << axialLengthMM << " mm"
	          << ", Omega = " << radiansPerSecond << " rad/s" << std::endl;
	std::cout << "Expected steady torque on fluid: wall 1 (inner) = "
	          << ROTATION_SIGN * EXPECTED_TORQUE_MAG_PHYS
	          << " Nm, wall 0 (outer) = " << -ROTATION_SIGN * EXPECTED_TORQUE_MAG_PHYS
	          << " Nm" << std::endl;
	
	TrackerStruct Tracker;
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
