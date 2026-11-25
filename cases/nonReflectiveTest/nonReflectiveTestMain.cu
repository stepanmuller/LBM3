constexpr float RHO_REFLECTION_TOLERANCE = 1.e-4f; // Density tolerance for the impedance scheme

constexpr float reynoldsNumber = 1000.f; // based on mean velocity and pipe diameter

constexpr int GRID_LEVEL_COUNT = 1;
constexpr float RES_GLOBAL = 0.2f; // mm

constexpr float uzInlet = 0.01f; // mean lattice velocity; centreline velocity is twice this
constexpr int ITERATION_COUNT = 50000;
constexpr int PLOTTER_PERIOD = 50000;

constexpr int WALL_REFINEMENT_COUNT = 0;
constexpr int TRACKER_PERIOD = 1;
constexpr bool TRACK_WALL_FORCE = false;
constexpr bool TRACK_ROTOR_FORCE = false;
constexpr bool TRACK_OPEN_BOUNDARIES = true;

constexpr float pipeDiameterPhys = 20.f; // mm
constexpr float pipeLengthPhys = 100.f;  // mm
constexpr float RInletPhys = 0.5f * pipeDiameterPhys; // mm
constexpr float NU_PHYS = 1.5e-5f; // m2/s, air
constexpr float RHO_PHYS = 1.225f; // kg/m3, air
constexpr float uzInletPhys = reynoldsNumber * NU_PHYS / (pipeDiameterPhys / 1000.f); // m/s, mean velocity
constexpr float DT_PHYS_GLOBAL = (uzInlet / uzInletPhys) * (RES_GLOBAL / 1000.f); // s

#include "../../include/types.h"

std::string STLPathPipe = "pipeD20L200STL.STL";

#include "../../include/STLFunctions.h"
#include "../../include/voxelizerFunctions.h"
#include "../../include/cellFunctions.h"

__cuda_callable__ void getRefinementModifier( const int& iCell, const int& jCell, const int& kCell,
											bool &refinementMarker, const InfoStruct& Info )
{
	float x, y, z;
	getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	const float r2 = x*x + y*y;
	const float refinementR2 = ( 0.5f * RInletPhys ) *  ( 0.5f * RInletPhys );
	const float refinementZ = pipeLengthPhys * 0.25f;
	refinementMarker = false;
	if ( Info.gridID == 0 )
	{
		if ( r2 < refinementR2 && z > - refinementZ && z < refinementZ ) refinementMarker = true;
	}
}

__cuda_callable__ void getInitialCondition( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
										const InfoStruct& Info )
{
	// leave default zeros
	//float x, y, z;
	//getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
	//const float r2 = x * x + y * y;
	//BC.uz = 2.f * uzInlet * std::max( 0.f, 1.f - r2 / (RInletPhys * RInletPhys) );
}

__cuda_callable__ void getOpenBC( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
								const InfoStruct& Info )
{
	if ( kCell == 0 ) // Inlet
	{
		float x, y, z;
		getXYZFromIJKCellIndex( iCell, jCell, kCell, x, y, z, Info );
		const float r2 = x * x + y * y;

		BC.dirichletU = true;
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 2.f * uzInlet * std::max( 0.f, 1.f - r2 / (RInletPhys * RInletPhys) );
		BC.nonReflective = true;
		BC.openBCID = 0;
	}
	else if ( kCell == Info.cellCountZ-1 ) // Outlet
	{
		BC.dirichletRho = true;
		BC.dRho = 0.f;
		BC.nonReflective = false;
		BC.openBCID = 1;
	}
}

__cuda_callable__ void getLocalBC( BCStruct &BC, const int& iCell, const int& jCell, const int& kCell,
								const InfoStruct& Info )
{
	BC.collisionLimiter = 0.f; // use K15
	if ( BC.wallID >= 0 )
	{
		BC.ux = 0.f;
		BC.uy = 0.f;
		BC.uz = 0.f;
	}
	BC.useIBB = false;
}

#include "../../include/gridBuilderFunctions.h"
#include "../../include/updateGrid.h"
#include "../../include/trackerFunctions.h"
#include "../../include/plotter/exportSectionCutPlot.h"
#include "../../include/plotter/plotTracker.h"

void plotGrids( const int &iterationsFinished, std::vector<GridStruct>& grids )
{
	// ZY section cut
	const float xCut = 0.f;
	exportSectionCutPlotZY( grids, xCut, iterationsFinished );
	if (system("python3 plotGridsFullCustom.py") != 0) {}
	//const float zCut = pipeLengthPhys * 0.5f - RES_GLOBAL * 0.5f;
	//exportSectionCutPlotXY( grids, zCut, iterationsFinished+1 );
	//if (system("python3 plotGridsFullCustom.py") != 0) {}
	std::cout << std::endl;
}

float getInletProfileError( const GridStruct &Grid )
{
    using ErrorSums = TNL::Containers::StaticArray<2, float>;

    const InfoStruct Info = Grid.Info;
    const auto &Inlet = Grid.openBCs.at( 0 );
    const bool esotwistFlipper = Info.esotwistFlipper;

    auto indexView = Inlet.indexArray.getConstView();
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

    auto fetch = [=] __cuda_callable__ ( const int index ) mutable -> ErrorSums
    {
        ErrorSums result( 0.0 );
        const int cell = indexView( index );

        // Use the same active-fluid mask as the boundary tracker.
        const int wallMap = wallMapView( cell );
        if ( wallMap == -3 || wallMap == -2 ) return result;
        if ( wallMap >= 0 )
        {
            int wallID;
            bool interfaceOverlapMarker;
            unpackWallID( wallDataView( wallMap ), wallID, interfaceOverlapMarker );
            if ( interfaceOverlapMarker ) return result;
        }

        int iCell, jCell, kCell;
        NBRStruct NBR;
        getCompressedIJKNBR( cell, iCell, jCell, kCell, NBR,
                            shifterView, iView, jView, kView,
                            jPlusView, kPlusView, jkPlusView, Info );
        if ( kCell != 0 ) return result; // This test's inlet is the lower Z face.

        // Read the completed state using the current esotwist layout.
        float f[27];
        int cellIndex[27], fIndex[27];
        getPreCollisionIndex( cellIndex, fIndex, NBR, esotwistFlipper );
        for ( int direction = 0; direction < 27; direction++ )
            f[direction] = fView( fIndex[direction], cellIndex[direction] );

        float dRho, ux, uy, uz;
        getDRhoUxUyUz( dRho, ux, uy, uz, f );

        // Obtain the prescribed profile before any non-reflecting adjustment.
        BCStruct Target;
        getOpenBC( Target, iCell, jCell, kCell, Info );

        const float uzTarget = static_cast<float>( Target.uz );
        const float error = static_cast<float>( uz ) - uzTarget;
        result[0] = error * error;
        result[1] = uzTarget * uzTarget;
        return result;
    };

    auto reduction = [] __cuda_callable__
        ( const ErrorSums &a, const ErrorSums &b ) -> ErrorSums
    {
        ErrorSums result;
        result[0] = a[0] + b[0];
        result[1] = a[1] + b[1];
        return result;
    };

    const ErrorSums sums = TNL::Algorithms::reduce<TNL::Devices::Cuda>(
        0, Inlet.openBCCount, fetch, reduction, ErrorSums( 0.0 ) );

    // A zero target norm makes the relative error undefined.
    if ( !( sums[1] > 0.0 ) )
        return std::numeric_limits<float>::quiet_NaN();

    return std::sqrt( sums[0] / sums[1] );
}

int main(int argc, char **argv)
{
	// STLs
	std::vector<STLStruct> gridStaticSTLs( 1 );
	readSTL( gridStaticSTLs[0], STLPathPipe );
	std::vector<STLStruct> rotorSTLs( 0 );

	// grids
	std::vector<GridStruct> grids( GRID_LEVEL_COUNT );
	BoundsStruct DomainBounds;
	DomainBounds = gridStaticSTLs[0].Bounds;
	DomainBounds.zMin = - pipeLengthPhys*0.5f;
	DomainBounds.zMax = pipeLengthPhys*0.5f;

	long long fluidUpdatesPerIteration = buildGrids( grids, gridStaticSTLs, rotorSTLs, DomainBounds );

	TrackerStruct Tracker;
	Tracker.TRACK_CUSTOM_VARIABLES = false;
	initializeTracker( Tracker, grids );

	// The tracker already provides area averages in physical units.
	std::ofstream inletOutput( "inletHistory.txt" );
	if ( !inletOutput ) throw std::runtime_error( "Cannot open inletHistory.txt" );
	inletOutput.exceptions( std::ios::failbit | std::ios::badbit );
	inletOutput.precision( std::numeric_limits<float>::max_digits10 );
	inletOutput << "Iteration;Area averaged inlet velocity [m/s];Area averaged pressure [Pa];Normalized inlet velocity error [1]\n";

	TNL::Timer lapTimer;
	lapTimer.reset();
	lapTimer.start();

	for ( int iterationsFinished = 1; iterationsFinished <= ITERATION_COUNT; iterationsFinished++ )
	{
		updateAllGrids( grids, 0 );

		updateTracker( Tracker, grids );
		const float inletProfileError = getInletProfileError( grids[0] );

		inletOutput << Tracker.iterationsFinished << ';'
					<< Tracker.normalVelocity( 0 ) << ';'
					<< Tracker.pressure( 0 ) << ';'
					<< inletProfileError << '\n';

		if ( iterationsFinished % PLOTTER_PERIOD == 0 )
		{
			lapTimer.stop();
			std::cout << "Iterations finished: " << iterationsFinished << std::endl;
			auto lapTime = lapTimer.getRealTime();
			const float updateCount = (float)fluidUpdatesPerIteration * (float)PLOTTER_PERIOD;
			const float glups = updateCount / lapTime / 1000000000.f;
			std::cout << "GLUPS: " << glups << std::endl;
			
			plotGrids( iterationsFinished, grids );
			plotTracker( Tracker );

			lapTimer.reset();
			lapTimer.start();
		}
	}
	
	inletOutput.close();

	return EXIT_SUCCESS;
}
