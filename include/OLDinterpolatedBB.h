#pragma once

#include "./types.h"
#include "./cellFunctions.h"

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

void buildLinkLengths( GridStruct &Grid, STLStruct &STL )
{
	InfoStruct &Info = Grid.Info;
	auto iView = Grid.IJK.iArray.getConstView();
	auto jView = Grid.IJK.jArray.getConstView();
	auto kView = Grid.IJK.kArray.getConstView();
	auto indexList = Grid.interpolatedBBCellList.getConstView();
	auto interpolatedBBLinkLengthsView = Grid.interpolatedBBLinkLengths.getView();
	auto bitPackedMarkerView = Grid.bitPackedMarkerArray.getConstView();
	
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
	
	// directions
	const int cxDir[27] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
	const int cyDir[27] = { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
	const int czDir[27] = { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };
	
	auto cellLambda = [ = ] __cuda_callable__( const int index ) mutable
    {
		const int cell = indexList( index );
		const int iCell = iView( cell );
		const int jCell = jView( cell );
		const int kCell = kView( cell );
		if ( kCell < 10 ) return; 
		float xCell, yCell, zCell;
		getXYZFromIJKCellIndex( iCell, jCell, kCell, xCell, yCell, zCell, Info );
		const int bitPackedMarkerInt = bitPackedMarkerView( cell );
		bool bitPackedMarkerBits[32];
		intToBools( bitPackedMarkerInt, bitPackedMarkerBits );
		// loop through all triangles (slow, yes I know, this is just a first attempt to build the links at all)
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
				if ( !bitPackedMarkerBits[direction] ) // this means there is no fluid coming from this direction
				{
					const float ex = -(float)cxDir[direction]; const float ey = -(float)cyDir[direction]; const float ez = -(float)czDir[direction];
					// Now, the triangle is defined by those 3 points and we are searching for an intersection with 
					// a line from the origin (=cell) pointing in the direction ex, ey, ez
					
					// Now, the triangle is defined by those 3 points and we are searching for an intersection with 
					// a line from the origin (=cell) pointing in the direction ex, ey, ez
					
					const float eps = Info.res / 1000.f;
					float hitX, hitY, hitZ, distance, t;

					if ( intersectRayTriangle( ax, ay, az, bx, by, bz, cx, cy, cz, ex, ey, ez, eps, hitX, hitY, hitZ, distance, t ) )
					{
						// NOTE: 't' is mathematically equal to (distance / normD). 
						// Therefore, you can skip the square root and division entirely here!
						float q = t / Info.res;
						
						if ( q > 0.f && q < 1.f ) 
						{
							interpolatedBBLinkLengthsView( direction, index ) = q;
						}
					}
					
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Grid.interpolatedBBCellList.getSize(), cellLambda );	
}
