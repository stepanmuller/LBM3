#pragma once

#include "./types.h"
#include "./D3Q27Directions.h"
#include "./cellFunctions.h"

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
	auto wallMarkerView = GridBuilder.wallMarkerArray.getConstView();
	auto wallAdjacentCellListView = GridBuilder.wallAdjacentCellList.getConstView();
	const int wallAdjacentCellCount = GridBuilder.wallAdjacentCellList.getSize();
	
	GridBuilder.linkExistenceMarkerArray.setValue( false );
	auto linkExistenceMarkerView = GridBuilder.linkExistenceMarkerArray.getView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cell = wallAdjacentCellListView( index );
		
		const int iCell = iView[ cell ];
		const int jCell = jView[ cell ];
		const int kCell = kView[ cell ];
		
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
		fullNBRList[OPP] = kPlusView( NBR.jPlus ); 	
		fullNBRList[OMM] = kMinusView( NBR.jMinus );	
		fullNBRList[MPM] = kMinusView( jPlusView( NBR.iMinus ) ); 	
		fullNBRList[PMP] = kPlusView( jMinusView( NBR.iPlus ) ); 	
		fullNBRList[MMP] = kPlusView( jMinusView( NBR.iMinus ) ); 
		fullNBRList[PPM] = kMinusView( jPlusView( NBR.iPlus ) ); 		
		fullNBRList[PMM] = kMinusView( jMinusView( NBR.iPlus ) ); 	
		fullNBRList[MPP] = kPlusView( jPlusView( NBR.iMinus ) ); 	
		fullNBRList[MMM] = kMinusView( jMinusView( NBR.iMinus ) );	
		fullNBRList[PPP] = kPlusView( jPlusView( NBR.iPlus ) );  	
		
		// now look at each neighbour if they are a true geometric wall neighbour 
		// -> that means the fluid-wall link exists
		for ( int direction = 1; direction < 27; direction++ )
		{
			const int nbr = fullNBRList[direction];
			if ( !wallMarkerView(nbr) ) continue; 
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
    const float eps,
    float &hitX, float &hitY, float &hitZ, float &distance, float &t )
{
    // Edge vectors of the triangle
    const float e1x = bx - ax;
    const float e1y = by - ay;
    const float e1z = bz - az;

    const float e2x = cx - ax;
    const float e2y = cy - ay;
    const float e2z = cz - az;

    // Cross product of ray direction and e2 ( P = D x e2 )
    const float px = ey * e2z - ez * e2y;
    const float py = ez * e2x - ex * e2z;
    const float pz = ex * e2y - ey * e2x;

    // Determinant
    const float det = e1x * px + e1y * py + e1z * pz;

    // If det is close to zero, the ray is parallel to the triangle plane
    const float detEps = 1e-8f; 
    if ( det > -detEps && det < detEps ) return false;

    const float invDet = 1.0f / det;

    // Vector from origin (cell center) to A
    const float tx = -ax;
    const float ty = -ay;
    const float tz = -az;

    // Calculate u parameter and test bounds
    const float u = (tx * px + ty * py + tz * pz) * invDet;
    if ( u < -eps || u > 1.0f + eps ) return false;

    // Cross product of T and e1 ( Q = T x e1 )
    const float qx = ty * e1z - tz * e1y;
    const float qy = tz * e1x - tx * e1z;
    const float qz = tx * e1y - ty * e1x;

    // Calculate v parameter and test bounds
    const float v = (ex * qx + ey * qy + ez * qz) * invDet;
    if ( v < -eps || u + v > 1.0f + eps ) return false;

    // Calculate t parameter (scale along the ray vector)
    t = (e2x * qx + e2y * qy + e2z * qz) * invDet;

    // If t is negative, the triangle is behind the cell center
    if ( t <= 0.0f ) return false;

    // Calculate coords of the hit (relative to the cell center)
    hitX = t * ex;
    hitY = t * ey;
    hitZ = t * ez;

    // Calculate physical signed distance
    const float normD = TNL::sqrt( ex * ex + ey * ey + ez * ez );
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
	const int wallAdjacentCellCount = GridBuilder.wallAdjacentCellList.getSize();
	
	const int STLCount = gridStaticSTLs.size();
	for ( int STLIndex = 0; STLIndex < STLCount; STLIndex++ )
	{
		STLStruct &STL = gridStaticSTLs[ STLIndex ];
		const int triangleCount = STL.triangleCount;
		auto axView = STL.axArray.getConstView();
		auto ayView = STL.ayArray.getConstView();
		auto azView = STL.azArray.getConstView();
		auto bxView = STL.bxArray.getConstView();
		auto byView = STL.byArray.getConstView();
		auto bzView = STL.bzArray.getConstView();
		auto cxView = STL.cxArray.getConstView();
		auto cyView = STL.cyArray.getConstView();
		auto czView = STL.czArray.getConstView();
		
		auto cellLambda = [ = ] __cuda_callable__( const int index ) mutable
		{
			const int cell = indexList( index );
			const int iCell = iView( cell );
			const int jCell = jView( cell );
			const int kCell = kView( cell );
			float xCell, yCell, zCell;
			getXYZFromIJKCellIndex( iCell, jCell, kCell, xCell, yCell, zCell, Info );
			// loop through all triangles (very slow, yes I know, maybe I will have to redo this if some STL gets very large)
			for ( int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++ )
			{
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
						const float ex = (float)CX_DIRECTIONS[direction]; 
						const float ey = (float)CY_DIRECTIONS[direction]; 
						const float ez = (float)CZ_DIRECTIONS[direction];
						// Now, the triangle is defined by those 3 points and we are searching for an intersection with 
						// a line from the origin (=cell) pointing in the direction ex, ey, ez
						
						const float eps = 1e-5f; // dimensionless tolerance for triangle edges
						// unlike during the voxelization, here we dont mind counting some intersection multiple times
						// better add some eps to the triangle to make sure each likely intersection is counted
						// the closest found intersection wins
						float hitX, hitY, hitZ, distance, t;

						if ( intersectRayTriangle( ax, ay, az, bx, by, bz, cx, cy, cz, ex, ey, ez, eps, hitX, hitY, hitZ, distance, t ) )
						{
							float q = t / Info.res;
							if ( q > 0.f && q <= 1.f ) 
							{
								const float qPrev = linkLengthView( direction, index );
								if ( q < qPrev ) linkLengthView( direction, index ) = q; // prefer the closer intersection
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
	
	std::cout << "	Links not found: " << linksNotFoundCount << " out of " << linksTotalCount << std::endl;
}
