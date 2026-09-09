#pragma once

#include "./types.h"
#include "./D3Q27Directions.h"
#include "./cellFunctions.h"

// TO DO:
// The current issue is that during voxelization, x and y coordinates of the STL get snapped to an integer grid
// to prevent ray from hitting a vertex, and to eliminate floating point errors.
// This way of voxelizing has so far never shown failure for STLs that passed the shared edge check.
// However, doing the shift in the voxelizer and not when calculating the link distances
// introduces errors: Some links between fluid and solid cells do not contain the triangle surface
// when evaluated via the original STL float coordinates.
// A likely better way to do this would be to transform the STL to a fixed integer grid immediately
// after opening, and then do all calculations that involve the STL using the integer grid.
// The voxelization is running on the integer grid mostly already, just the intersection coordinte 
// would need a rewrite. The link length evaluation below would require a serious rewrite
// that would evaluate the link lengths using integer arithmetic.

int countOnesInBoolArray2D( const BoolArray2DType &boolArray )
{
	const int firstBound = boolArray.getSizes()[0];
	const int secondBound = boolArray.getSizes()[1];
	auto boolView = boolArray.getConstView();
	auto fetch = [ = ] __cuda_callable__( const int singleIndex )
	{
		const int i = singleIndex % firstBound;
		const int j = singleIndex / firstBound;
		if ( boolView( i, j ) ) return 1;
		else return 0;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	const int start = 0;
	const int end = firstBound * secondBound;
	const int onesCount = TNL::Algorithms::reduce<TNL::Devices::Cuda>( start, end, fetch, reduction, 0 );
	return onesCount;
}

void buildLinkExistenceMarkerArray( GridBuilderStruct &GridBuilder )
{
	const InfoStruct &Info = GridBuilder.Info;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	auto jPlusView = GridBuilder.NBR.jPlusArray.getConstView();
	auto kPlusView = GridBuilder.NBR.kPlusArray.getConstView();
	auto jMinusView = GridBuilder.NBR.jMinusArray.getConstView();
	auto kMinusView = GridBuilder.NBR.kMinusArray.getConstView();
	auto parentMapView = GridBuilder.parentMapArray.getConstView();
	auto fineToCoarseMarkerView = GridBuilder.fineToCoarseMarkerArray.getConstView();
	auto coarseToFineMarkerView = GridBuilder.coarseToFineMarkerArray.getConstView();
	const bool iAmCoarsest = (GridBuilder.parentMapArray.getSize() == 0);
	const bool iAmFinest = (GridBuilder.fineToCoarseMarkerArray.getSize() == 0);
	auto wallMarkerView = GridBuilder.wallMarkerArray.getConstView();
	auto wallAdjacentCellListView = GridBuilder.wallAdjacentCellList.getConstView();
	const int wallAdjacentCellCount = GridBuilder.wallAdjacentCellList.getSize();
	
	GridBuilder.linkExistenceMarkerArray.setValue( false );
	auto linkExistenceMarkerView = GridBuilder.linkExistenceMarkerArray.getView();
	GridBuilder.linkPiercesInterfaceMarkerArray.setValue( false );
	auto linkPiercesInterfaceMarkerView = GridBuilder.linkPiercesInterfaceMarkerArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = wallAdjacentCellListView( index );
		
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
	
		bool iAmInterface = false;
		if ( !iAmCoarsest )
		{
			if ( parentMapView( cell ) >= 0 ) iAmInterface = true;
		}
		if ( !iAmFinest )
		{ 
			if ( fineToCoarseMarkerView( cell ) ) iAmInterface = true;
			if ( coarseToFineMarkerView( cell ) ) iAmInterface = true;
		}
		
		NBRStruct NBR;
		NBR.self = cell;
		NBR.jPlus = jPlusView( cell );
		NBR.kPlus = kPlusView( cell );
		NBR.jkPlus = jPlusView( kPlusView( cell ) );
		NBR.jMinus = jMinusView( cell );
		NBR.kMinus = kMinusView( cell );
		finishNBRAll( NBR, Info );
		
		int fullNBRList[27];
		fullNBRList[OOO]  = cell;	
		fullNBRList[POO]  = NBR.iPlus;  
		fullNBRList[MOO]  = NBR.iMinus; 			
		fullNBRList[OOM]  = NBR.kMinus; 
		fullNBRList[OOP]  = NBR.kPlus;  					
		fullNBRList[OMO]  = NBR.jMinus; 	
		fullNBRList[OPO]  = NBR.jPlus;  			
		fullNBRList[POM]  = kMinusView( NBR.iPlus );	
		fullNBRList[MOP]  = kPlusView( NBR.iMinus );	
		fullNBRList[POP] = kPlusView( NBR.iPlus ); 	
		fullNBRList[MOM]  = kMinusView( NBR.iMinus );
		fullNBRList[MMO] = jMinusView( NBR.iMinus );	
		fullNBRList[PPO] = jPlusView( NBR.iPlus ); 	
		fullNBRList[OPM] = kMinusView( NBR.jPlus );	
		fullNBRList[OMP] = kPlusView( NBR.jMinus );	
		fullNBRList[MPO] = jPlusView( NBR.iMinus );	
		fullNBRList[PMO] = jMinusView( NBR.iPlus );	
		fullNBRList[OPP] = jPlusView( NBR.kPlus ); 	
		fullNBRList[OMM] = kMinusView( NBR.jMinus );	
		fullNBRList[MPM] = kMinusView( jPlusView( NBR.iMinus ) ); 	
		fullNBRList[PMP] = kPlusView( jMinusView( NBR.iPlus ) ); 	
		fullNBRList[MMP] = kPlusView( jMinusView( NBR.iMinus ) ); 
		fullNBRList[PPM] = kMinusView( jPlusView( NBR.iPlus ) ); 		
		fullNBRList[PMM] = kMinusView( jMinusView( NBR.iPlus ) ); 	
		fullNBRList[MPP] = jPlusView( kPlusView( NBR.iMinus ) ); 	
		fullNBRList[MMM] = kMinusView( jMinusView( NBR.iMinus ) );	
		fullNBRList[PPP] = jPlusView( kPlusView( NBR.iPlus ) );  	
		
		// now look at each neighbour if they are a true geometric wall neighbour 
		// -> that means the fluid-wall link exists
		for ( int direction = 1; direction < 27; direction++ )
		{
			const int nbr = fullNBRList[direction];
			if ( !wallMarkerView(nbr) ) continue; 
			
			bool nbrIsInterface = false;
			if ( !iAmCoarsest )
			{
				if ( parentMapView( nbr ) >= 0 ) nbrIsInterface = true;
			}
			if ( !iAmFinest )
			{ 
				if ( fineToCoarseMarkerView( nbr ) ) nbrIsInterface = true;
				if ( coarseToFineMarkerView( nbr ) ) nbrIsInterface = true;
			}
			
			// if we got here, nbr is a wall, check its position
			const int cx = CX_DIRECTIONS[direction]; 
			const int cy = CY_DIRECTIONS[direction]; 
			const int cz = CZ_DIRECTIONS[direction];
			const int iExpected = iCell + cx; 
			const int jExpected = jCell + cy; 
			const int kExpected = kCell + cz;
			const int iActual = iView( nbr ); 
			const int jActual = jView( nbr ); 
			const int kActual = kView( nbr );
			if ( iActual == iExpected && jActual == jExpected && kActual == kExpected ) 
			{
				// nbr is a true geometric wall neighbour -> mark the link existence
				linkExistenceMarkerView( direction, index ) = true;
				// if me or nbr is interface, mark that this link pierces an interface 
				// -> linkLength will be forced to 0.5f later
				if ( iAmInterface || nbrIsInterface ) linkPiercesInterfaceMarkerView( direction, index ) = true;
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, wallAdjacentCellCount, cellLambda );	
}

__host__ __device__ bool intersectRayTriangle(
    const float ax, const float ay, const float az,
    const float bx, const float by, const float bz,
    const float cx, const float cy, const float cz,
    const float ex, const float ey, const float ez,
    const float edgeTol,
    float &hitX, float &hitY, float &hitZ, float &distance, float &t )
{
    // Triangle edges AB and AC.
    const float e1x = bx - ax;
    const float e1y = by - ay;
    const float e1z = bz - az;

    const float e2x = cx - ax;
    const float e2y = cy - ay;
    const float e2z = cz - az;

    // Triangle normal; its magnitude is twice the triangle area.
    const float nx = e1y * e2z - e1z * e2y;
    const float ny = e1z * e2x - e1x * e2z;
    const float nz = e1x * e2y - e1y * e2x;
    const float area2 = sqrtf(nx * nx + ny * ny + nz * nz);

    if (area2 == 0.f) return false;

    // P = ray direction x AC.
    const float px = ey * e2z - ez * e2y;
    const float py = ez * e2x - ex * e2z;
    const float pz = ex * e2y - ey * e2x;

    const float det = e1x * px + e1y * py + e1z * pz;

    // Preserve your existing parallel-ray rejection.
    const float detEps = 1e-8f;
    if (det > -detEps && det < detEps) return false;

    const float invDet = 1.f / det;

    // Convert the physical edge tolerance into barycentric tolerances.
    const float lenAB = sqrtf(e1x * e1x + e1y * e1y + e1z * e1z);
    const float lenAC = sqrtf(e2x * e2x + e2y * e2y + e2z * e2z);

    const float bcx = cx - bx;
    const float bcy = cy - by;
    const float bcz = cz - bz;
    const float lenBC = sqrtf(bcx * bcx + bcy * bcy + bcz * bcz);

    const float uTol = edgeTol * lenAC / area2;
    const float vTol = edgeTol * lenAB / area2;
    const float wTol = edgeTol * lenBC / area2;

    // Ray origin is the cell center, at (0, 0, 0).
    const float tx = -ax;
    const float ty = -ay;
    const float tz = -az;

    const float u = (tx * px + ty * py + tz * pz) * invDet;
    if (u < -uTol) return false;

    // Q = T x AB.
    const float qx = ty * e1z - tz * e1y;
    const float qy = tz * e1x - tx * e1z;
    const float qz = tx * e1y - ty * e1x;

    const float v = (ex * qx + ey * qy + ez * qz) * invDet;
    if (v < -vTol || u + v > 1.f + wTol) return false;

    t = (e2x * qx + e2y * qy + e2z * qz) * invDet;

    // Allow zero and negative t here.
    // The caller applies qTol to decide whether the hit is acceptable.

    hitX = t * ex;
    hitY = t * ey;
    hitZ = t * ez;

    const float normD = sqrtf(ex * ex + ey * ey + ez * ez);
    distance = t * normD;

    return true;
}

void buildLinkLengthArray( GridBuilderStruct &GridBuilder, std::vector<STLStruct> &gridStaticSTLs )
{
	InfoStruct &Info = GridBuilder.Info;
	auto iView = GridBuilder.IJK.iArray.getConstView();
	auto jView = GridBuilder.IJK.jArray.getConstView();
	auto kView = GridBuilder.IJK.kArray.getConstView();
	auto indexList = GridBuilder.wallAdjacentCellList.getConstView();
	GridBuilder.linkLengthArray.setValue( 2.f ); // purposedly large value that will get overwritten by actual intersections
	auto linkLengthView = GridBuilder.linkLengthArray.getView();
	auto linkExistenceMarkerView = GridBuilder.linkExistenceMarkerArray.getConstView();
	auto linkPiercesInterfaceMarkerView = GridBuilder.linkPiercesInterfaceMarkerArray.getConstView();
	const int wallAdjacentCellCount = GridBuilder.wallAdjacentCellList.getSize();
	
	const int STLCount = gridStaticSTLs.size();
	for ( int STLIndex = 0; STLIndex < STLCount; STLIndex++ )
	{
		STLStruct &STL = gridStaticSTLs[ STLIndex ];
		// const int triangleCount = STL.triangleCount;
		auto axView = STL.axArray.getConstView();
		auto ayView = STL.ayArray.getConstView();
		auto azView = STL.azArray.getConstView();
		auto bxView = STL.bxArray.getConstView();
		auto byView = STL.byArray.getConstView();
		auto bzView = STL.bzArray.getConstView();
		auto cxView = STL.cxArray.getConstView();
		auto cyView = STL.cyArray.getConstView();
		auto czView = STL.czArray.getConstView();
		
		const float &oxBin = STL.oxBin;
		const float &oyBin = STL.oyBin;
		const float &ozBin = STL.ozBin;
		
		auto binView = STL.binArray.getConstView();
		auto firstInBinView = STL.firstInBinArray.getConstView();
		const int &binCountX = STL.binCountX;
		const int &binCountY = STL.binCountY;
		const int &binCountZ = STL.binCountZ;
		const int binCountXY = binCountX * binCountY;
		const float &binSize = STL.binSize;
		
		auto cellLambda = [ = ] __cuda_callable__( const int index ) mutable
		{
			const int cell = indexList( index );
			const int iCell = iView( cell );
			const int jCell = jView( cell );
			const int kCell = kView( cell );
			float xCell, yCell, zCell;
			getXYZFromIJKCellIndex( iCell, jCell, kCell, xCell, yCell, zCell, Info );
			
			const int iBin = (int)(( xCell - oxBin ) / binSize);
			const int jBin = (int)(( yCell - oyBin ) / binSize);
			const int kBin = (int)(( zCell - ozBin ) / binSize);
			if ( iBin < 0 || iBin >= binCountX || jBin < 0 || jBin >= binCountY || kBin < 0 || kBin >= binCountZ ) return;
			
			const int bin = binCountXY * kBin + binCountX * jBin + iBin;
			const int startReadIndex = firstInBinView( bin );
			const int endReadIndex = firstInBinView( bin + 1 );
			
			// loop through triangles in the bin
			for ( int readIndex = startReadIndex; readIndex < endReadIndex; readIndex++ )
			{
				const int triangleIndex = binView( readIndex );
				// transform into the coordinate system of the cell
				const float ax = axView[ triangleIndex ] - xCell;
				const float ay = ayView[ triangleIndex ] - yCell;
				const float az = azView[ triangleIndex ] - zCell;
				const float bx = bxView[ triangleIndex ] - xCell;
				const float by = byView[ triangleIndex ] - yCell;
				const float bz = bzView[ triangleIndex ] - zCell;
				const float cx = cxView[ triangleIndex ] - xCell;
				const float cy = cyView[ triangleIndex ] - yCell;
				const float cz = czView[ triangleIndex ] - zCell;
				// loop through directions
				for ( int direction = 1; direction < 27; direction++ )
				{
					if ( linkExistenceMarkerView( direction, index ) ) // this means there is a wall in this direction
					{
						if ( linkPiercesInterfaceMarkerView( direction, index ) )
						{
							// force any link lengths which pierce interface to safe 0.5f
							// this way interface geometry appears the same for parent and child
							linkLengthView( direction, index ) = 0.5f;  
							continue;
						}
						const float ex = (float)CX_DIRECTIONS[direction]; 
						const float ey = (float)CY_DIRECTIONS[direction]; 
						const float ez = (float)CZ_DIRECTIONS[direction];
						// Now, the triangle is defined by those 3 points and we are searching for an intersection with 
						// a line from the origin (=cell) pointing in the direction ex, ey, ez
						
						// Physical allowance outside each triangle edge:
						// 0.1 means 10% of this grid level's cell spacing.
						const float edgeTolCells = 0.1f;
						const float edgeTol = edgeTolCells * Info.res;

						float hitX, hitY, hitZ, distance, t;

						if ( intersectRayTriangle( ax, ay, az, bx, by, bz, cx, cy, cz, ex, ey, ez, edgeTol, hitX, hitY, hitZ, distance, t ) )
						{
							float q = t / Info.res;
							
							const float qTol = 0.2f;
							
							if (q > -qTol && q <= 1.f + qTol)
							{
								q = std::clamp(q, 0.00001f, 1.f);

								const float qPrev = linkLengthView(direction, index);
								if (q < qPrev) linkLengthView(direction, index) = q;
							}
						}
					}
				}
			}
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, wallAdjacentCellCount, cellLambda );	
	}
	// catch any misbehaving links that stayed on 2.f and make them 0.5f as a safe fallback, count how many such cases occur
	auto fetch = [ = ] __cuda_callable__( const int index ) mutable
	{
		int linkNotFound = 0;
		for ( int direction = 1; direction < 27; direction++ )
		{
			if ( !linkExistenceMarkerView( direction, index ) ) continue;
			const float qPrev = linkLengthView( direction, index );
			if ( qPrev > 1.f ) 
			{
				linkLengthView( direction, index ) = 0.5f;
				linkNotFound++;
			}
		}
		return linkNotFound;
	};
	auto reduction = [] __cuda_callable__( const int& a, const int& b )
	{
		return a + b;
	};
	const int linksNotFoundCount = TNL::Algorithms::reduce<TNL::Devices::Cuda>( 0, wallAdjacentCellCount, fetch, reduction, 0 );
	// compare that to the number of existing links
	const int linksTotalCount = countOnesInBoolArray2D( GridBuilder.linkExistenceMarkerArray );
	
	if ( linksNotFoundCount == 0 ) std::cout << "	Level " << GridBuilder.Info.gridID << " found all " << linksTotalCount << " IBB links" << std::endl;
	else std::cout << "	Level " << GridBuilder.Info.gridID << " failed to find " << linksNotFoundCount << " IBB links out of " << linksTotalCount << std::endl;
}

__cuda_callable__ inline void packWallData( uint32_t (&packed)[4], 
											const bool (&linkExists)[26], const float (&linkLength)[26], 
											const int wallID, const bool parentInterfaceMarker )
{
    constexpr uint32_t divider = 23u;
    uint32_t digits[28]; // 1 wallID, 26 links, 1 parentInterfaceMarker
    digits[0] = static_cast<uint32_t>( wallID );
   
    for( int i = 0; i < 26; ++i )
    {
        if( !linkExists[i] )
        {
            digits[i + 1] = 0u;
            continue;
        }
        const float q = linkLength[i];
        // Round to nearest multiple of 0.05:
        // q = 0 -> code 1, q = 0.5 -> code 11, q = 1 -> code 21.
        digits[i + 1] = 1u + static_cast<uint32_t>( q * 20.0f + 0.5f );
    }
    
    digits[27] = parentInterfaceMarker ? 1u : 0u;
   
    for( int packedIndex = 0; packedIndex < 4; packedIndex++ )
    {
        uint32_t value = 0u;
        for( int digitIndex = 6; digitIndex >= 0; digitIndex-- )
        {
            value = value * divider + digits[7 * packedIndex + digitIndex];
		}
        packed[packedIndex] = value;
    }
}

__cuda_callable__ inline void unpackWallData( 	const uint32_t (&packed)[4],
												bool (&linkExists)[26], float (&linkLength)[26],
												int &wallID, bool &parentInterfaceMarker )
{
    constexpr uint32_t divider = 23u;
    uint32_t digits[28]; // 1 wallID, 26 links, 1 parentInterfaceMarker

    for( int packedIndex = 0; packedIndex < 4; packedIndex++ )
    {
        uint32_t value = packed[packedIndex];
        for( int digitIndex = 0; digitIndex < 7; digitIndex++ )
        {
            digits[7 * packedIndex + digitIndex] = value % divider;
            value /= divider;
        }
    }

    wallID = static_cast<int>( digits[0] );

    for( int i = 0; i < 26; ++i )
    {
        const uint32_t code = digits[i + 1];
        linkExists[i] = ( code != 0u );
        if ( !linkExists[i] ) linkLength[i] = 0.f;
        else linkLength[i] = std::clamp( static_cast<float>( code - 1u ) / 20.0f, 0.00001f, 1.f); 
    }

    parentInterfaceMarker = ( digits[27] == 1u );
}
