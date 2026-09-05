#pragma once

#include "./types.h"

__host__ __device__ bool getRayHitYesNo( 	const int &i, const int &j, 
											const long long &wab, const long long &wbc, const long long &wca,
											const int &axInt, const int &ayInt, 
											const int &bxInt, const int &byInt,
											const int &cxInt, const int &cyInt,
											const long long &abx, const long long &aby,
											const long long &bcx, const long long &bcy,
											const long long &cax, const long long &cay )
{
	// Identify if triangle is hit by a ray
	if ( wab > 0 && wbc > 0 && wca > 0 ) return true;
    if ( wab < 0 || wbc < 0 || wca < 0 ) return false;
    // If these two checks did not produce a return, it means ray is hitting exactly an edge
	// Translate the triangle so that the ray lies at [0, 0]
	const int rayXInt = i * 100;
	const int rayYInt = j * 100;
    const long long ax0 = axInt - rayXInt;
    const long long ay0 = ayInt - rayYInt;
    const long long bx0 = bxInt - rayXInt;
    const long long by0 = byInt - rayYInt;
    const long long cx0 = cxInt - rayXInt;
    const long long cy0 = cyInt - rayYInt;
            
    if ( wab == 0LL )
    {
		if ( abx == 0LL ) // vertical edge
		{
			if ( cx0 < 0LL ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( bx0 > 0LL && cy0 * bx0 > by0 * cx0 ) return true;
		if ( bx0 < 0LL && cy0 * bx0 < by0 * cx0 ) return true;
	}
	
	if ( wbc == 0LL )
    {
		if ( bcx == 0LL ) // vertical edge
		{
			if ( ax0 < 0LL ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( cx0 > 0LL && ay0 * cx0 > cy0 * ax0 ) return true;
		if ( cx0 < 0LL && ay0 * cx0 < cy0 * ax0 ) return true;
	}
	
	if ( wca == 0LL )
    {
		if ( cax == 0LL ) // vertical edge
		{
			if ( bx0 < 0LL ) return true;
			else return false;
		}
		// if we got here the edge is not vertical, so we can determine above or below
		if ( ax0 > 0LL && by0 * ax0 > ay0 * bx0 ) return true;
		if ( ax0 < 0LL && by0 * ax0 < ay0 * bx0 ) return true;
	}

    return false; 
}

void voxelizeSTL( rayMapStruct &rayMap, STLStruct &STL, VoxelizerStruct &Voxelizer )
{
	InfoStruct &Info = Voxelizer.Info;
	IntArrayType &rayMapArray = rayMap.rayMapArray;
	IntArrayType hitCounterTempArray;
	IntArrayType &hitCounterScanArray = rayMap.hitCounterScanArray;
	int &totalHitCount = rayMap.totalHitCount;
	
	hitCounterTempArray.setSize( Info.cellCountX * Info.cellCountY );
	hitCounterTempArray.setValue( 0 );
	hitCounterScanArray.setSize( Info.cellCountX * Info.cellCountY + 1 );
	hitCounterScanArray.setValue( 0 );
	
	auto hitCounterTempView = hitCounterTempArray.getView();
	auto hitCounterScanView = hitCounterScanArray.getView();
	
	auto axView = STL.axArray.getConstView();
	auto ayView = STL.ayArray.getConstView();
	auto azView = STL.azArray.getConstView();
	auto bxView = STL.bxArray.getConstView();
	auto byView = STL.byArray.getConstView();
	auto bzView = STL.bzArray.getConstView();
	auto cxView = STL.cxArray.getConstView();
	auto cyView = STL.cyArray.getConstView();
	auto czView = STL.czArray.getConstView();
	
	IntArrayType &raysPerTriangleCounterArray = STL.raysPerTriangleCounterArray;
	auto raysPerTriangleCounterView = raysPerTriangleCounterArray.getView();
	IntArrayType &threadToTriangleMapArray = STL.threadToTriangleMapArray;
	const int threadCountMax = threadToTriangleMapArray.getSize();
	auto threadToTriangleMapView = threadToTriangleMapArray.getView();
	
	// first find rays per triangle to be able to distribute workload on threads evenly later
	auto raysPerTriangleCounterLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		// transform into the coordinate system of the LBM grid
		const float ax = axView[ triangleIndex ] - Info.ox;
		const float ay = ayView[ triangleIndex ] - Info.oy;
		const float bx = bxView[ triangleIndex ] - Info.ox;
		const float by = byView[ triangleIndex ] - Info.oy;
		const float cx = cxView[ triangleIndex ] - Info.ox;
		const float cy = cyView[ triangleIndex ] - Info.oy;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const int axInt = (int)(round( ax * scale )) * 2 + 1;
		const int ayInt = (int)(round( ay * scale )) * 2 + 1;
		const int bxInt = (int)(round( bx * scale )) * 2 + 1;
		const int byInt = (int)(round( by * scale )) * 2 + 1;
		const int cxInt = (int)(round( cx * scale )) * 2 + 1;
		const int cyInt = (int)(round( cy * scale )) * 2 + 1;
		// now figure out which rays can possibly hit the triangle -> get bounds
		const int xIntMin = TNL::max( 0, TNL::min( axInt, bxInt, cxInt, (int)(Info.cellCountX-1)*100 ) );
		const int xIntMax = TNL::min( (int)(Info.cellCountX-1)*100, TNL::max( axInt, bxInt, cxInt, 0 ) );
		const int yIntMin = TNL::max( 0, TNL::min( ayInt, byInt, cyInt, (int)(Info.cellCountY-1)*100 ) );
		const int yIntMax = TNL::min( (int)(Info.cellCountY-1)*100, TNL::max( ayInt, byInt, cyInt, 0 ) );
		const int iStart = (xIntMin + 99) / 100;
		const int iEnd = xIntMax / 100 + 1;
		const int jStart = (yIntMin + 99) / 100;
		const int jEnd = yIntMax / 100 + 1;
		const int raysPerTriangleCount = TNL::max( 1, ( jEnd - jStart ) * ( iEnd - iStart ) );
		
		raysPerTriangleCounterView[ triangleIndex ] = raysPerTriangleCount;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, raysPerTriangleCounterLambda );
	
	const int taskCount = TNL::sum( raysPerTriangleCounterArray );
	// set worst case scenario limit for rays per thread so that thread count will never exceed size of the threadToTriangleMapArray
	const int raysPerThreadLimit = TNL::max(16, TNL::max(0, taskCount - STL.triangleCount) / ( threadCountMax - STL.triangleCount )); 
			
	IntArrayType &threadsPerTriangleScanArray = raysPerTriangleCounterArray;
	auto &threadsPerTriangleScanView = raysPerTriangleCounterView;
	
	threadsPerTriangleScanArray = ( raysPerTriangleCounterArray + raysPerThreadLimit - 1 ) / raysPerThreadLimit;
	const int threadsPerLastTriangle = threadsPerTriangleScanArray.getElement( STL.triangleCount - 1 );
	TNL::Algorithms::inplaceExclusiveScan( threadsPerTriangleScanArray );
	const int threadCount = threadsPerTriangleScanArray.getElement( STL.triangleCount - 1 ) + threadsPerLastTriangle;
	
	threadToTriangleMapArray.setValue( 0 );
	auto threadToTriangleMapLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
	{
		const int firstThreadOnTriangle = threadsPerTriangleScanView[ triangleIndex ];
		threadToTriangleMapView[ firstThreadOnTriangle ] = triangleIndex;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, threadToTriangleMapLambda );
	
	TNL::Algorithms::inplaceInclusiveScan( threadToTriangleMapArray, 0, threadCount, TNL::Max{} );
	
	auto rayHitCounterLambda = [ = ] __cuda_callable__( const int threadIndex ) mutable
    {
		// first, find which triangle our thread is working on
		const int triangleIndex = threadToTriangleMapView[ threadIndex ];
		// what is the first thread working on our triangle?
		const int firstThreadOnTriangle = threadsPerTriangleScanView[ triangleIndex ];
		// where should our portion start?
		const int taskStart = ( threadIndex - firstThreadOnTriangle ) * raysPerThreadLimit;
		
		// transform into the coordinate system of the LBM grid
		const float ax = axView[ triangleIndex ] - Info.ox;
		const float ay = ayView[ triangleIndex ] - Info.oy;
		//const float az = azView[ triangleIndex ] - Info.oz;
		const float bx = bxView[ triangleIndex ] - Info.ox;
		const float by = byView[ triangleIndex ] - Info.oy;
		//const float bz = bzView[ triangleIndex ] - Info.oz;
		const float cx = cxView[ triangleIndex ] - Info.ox;
		const float cy = cyView[ triangleIndex ] - Info.oy;
		//const float cz = czView[ triangleIndex ] - Info.oz;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const int axInt = (int)(round( ax * scale )) * 2 + 1;
		const int ayInt = (int)(round( ay * scale )) * 2 + 1;
		const int bxInt = (int)(round( bx * scale )) * 2 + 1;
		const int byInt = (int)(round( by * scale )) * 2 + 1;
		const int cxInt = (int)(round( cx * scale )) * 2 + 1;
		const int cyInt = (int)(round( cy * scale )) * 2 + 1;
		// now figure out which rays can possibly hit the triangle -> get bounds
		const int xIntMin = TNL::max( 0, TNL::min( axInt, bxInt, cxInt, (int)(Info.cellCountX-1)*100 ) );
		const int xIntMax = TNL::min( (int)(Info.cellCountX-1)*100, TNL::max( axInt, bxInt, cxInt, 0 ) );
		const int yIntMin = TNL::max( 0, TNL::min( ayInt, byInt, cyInt, (int)(Info.cellCountY-1)*100 ) );
		const int yIntMax = TNL::min( (int)(Info.cellCountY-1)*100, TNL::max( ayInt, byInt, cyInt, 0 ) );
		const int iStartGlobal = (xIntMin + 99) / 100;
		const int iEndGlobal = xIntMax / 100 + 1;
		const int jStartGlobal = (yIntMin + 99) / 100;
		const int jEndGlobal = yIntMax / 100 + 1;
		// Prepare cayIntculation of the intersection yes no detection
		// Here we will have to switch to long long because they get multiplied and an integer could overflow if a triangle is bigger than 300 cells
		// A long long is large enough if the triangle is up to about 20M cells
		// Transform the triangle into coordinate system where the first ray is [iMin, jMin]
		const long long xLongMin = 100LL * (long long)iStartGlobal;
		const long long yLongMin = 100LL * (long long)jStartGlobal;
		const long long axLong = axInt - xLongMin;
		const long long ayLong = ayInt - yLongMin;
		const long long bxLong = bxInt - xLongMin;
		const long long byLong = byInt - yLongMin;
		const long long cxLong = cxInt - xLongMin;
		const long long cyLong = cyInt - yLongMin;
		const long long abxLong = bxLong - axLong;
		const long long abyLong = byLong - ayLong;
		const long long bcxLong = cxLong - bxLong;
		const long long bcyLong = cyLong - byLong;
		const long long caxLong = axLong - cxLong;
		const long long cayLong = ayLong - cyLong;
		// Projection of the triangle area, if it is zero, there are no intersections
		const long long signedArea = abxLong * bcyLong - abyLong * bcxLong;
		if ( signedArea == 0LL ) return; 
		long long qzLong = 1LL;
		if ( signedArea < 0LL ) qzLong = -1LL; // now ABCQ has positive volume
		// Calculate edge functions for the first ray [iMin, jMin]
		const long long wab0 = qzLong * ( abxLong * ( -byLong ) - abyLong * ( -bxLong ) );
		const long long wbc0 = qzLong * ( bcxLong * ( -cyLong ) - bcyLong * ( -cxLong ) );
		const long long wca0 = qzLong * ( caxLong * ( -ayLong ) - cayLong * ( -axLong ) );
		// Derivatives of the edge function with respect to i and j
		// We will be adding this each time we do a step in i or j direction
		const long long dwab_di = - qzLong * abyLong * 100LL;
		const long long dwab_dj = qzLong * abxLong * 100LL;
		const long long dwbc_di = - qzLong * bcyLong * 100LL;
		const long long dwbc_dj = qzLong * bcxLong * 100LL;
		const long long dwca_di = - qzLong * cayLong * 100LL;
		const long long dwca_dj = qzLong * caxLong * 100LL;
		// Prepare calculation of the intersection coordinate
		//const float v1x = bx - ax;
		//const float v1y = by - ay;
		//const float v1z = bz - az;
		//const float v2x = cx - ax;
		//const float v2y = cy - ay;
		//const float v2z = cz - az;
		//const float nx = v1y * v2z - v1z * v2y;
		//const float ny = v1z * v2x - v1x * v2z;
		//const float nz = v1x * v2y - v1y * v2x;
		//const float maxZ = TNL::max(az, bz, cz);
		//const float minZ = TNL::min(az, bz, cz);
		//const float midZ = 0.5f * (maxZ + minZ);
		
		//float rayZ;
		
		// Everything about the triangle is prepared now
		// Find where we need to start from
		const int iSpan = iEndGlobal - iStartGlobal;
		const int jSpan = jEndGlobal - jStartGlobal;
		
		const int taskLast = TNL::min(taskStart + raysPerThreadLimit, iSpan * jSpan) - 1;
		
		const int jStartThread = jStartGlobal + (taskStart / iSpan);
		const int iStartThread = iStartGlobal + (taskStart % iSpan);

		const int jEndThread = jStartGlobal + (taskLast / iSpan) + 1;
		const int iEndThread = iStartGlobal + (taskLast % iSpan) + 1;
		
		// we will be changing this each time j changes
		int iStartJ, iEndJ;
		long long wabJ = wab0 + (long long)(jStartThread - jStartGlobal) * dwab_dj; 
		long long wbcJ = wbc0 + (long long)(jStartThread - jStartGlobal) * dwbc_dj;
		long long wcaJ = wca0 + (long long)(jStartThread - jStartGlobal) * dwca_dj;
		
		// we will be changing this each time j or i changes
		long long wab, wbc, wca;
		bool rayHit;
		
		for ( int j = jStartThread; j < jEndThread; j++ )
		{
			if ( j == jStartThread ) 
			{
				iStartJ = iStartThread;
				wab = wabJ + (long long)(iStartThread - iStartGlobal) * dwab_di;
				wbc = wbcJ + (long long)(iStartThread - iStartGlobal) * dwbc_di;
				wca = wcaJ + (long long)(iStartThread - iStartGlobal) * dwca_di;
			}
			else 
			{
				iStartJ = iStartGlobal;
				wab = wabJ;
				wbc = wbcJ;
				wca = wcaJ;
			}
			if ( j == jEndThread - 1 ) 
				iEndJ = iEndThread; 
			else 
				iEndJ = iEndGlobal;
			// possible optimization for later:
			// here we are starting to check all elements of the row
			// instead we can just find the first true element and last true element
			for ( int i = iStartJ; i < iEndJ; i++ )
			{
				rayHit = getRayHitYesNo( i, j, wab, wbc, wca, 
										axInt, ayInt, bxInt, byInt, cxInt, cyInt, 
										abxLong, abyLong, bcxLong, bcyLong, caxLong, cayLong );
				if ( rayHit ) 
				{
					const int rayIndex = j * Info.cellCountX + i;
					TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(hitCounterScanView( rayIndex ), 1);
				}
				// add the increments after ending one i pass - we will be increasing i by 1
				wab = wab + dwab_di;
				wbc = wbc + dwbc_di;
				wca = wca + dwca_di;
			}
			// add the increments after ending one j pass - we will be increasing j by 1
			wabJ = wabJ + dwab_dj;
			wbcJ = wbcJ + dwbc_dj;
			wcaJ = wcaJ + dwca_dj;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, threadCount, rayHitCounterLambda );
	
	TNL::Algorithms::inplaceExclusiveScan( hitCounterScanArray, 0, Info.cellCountX * Info.cellCountY + 1, TNL::Plus{} );
	totalHitCount = hitCounterScanArray.getElement( Info.cellCountX * Info.cellCountY );
	
	rayMapArray.setSize( totalHitCount );
	auto rayMapView = rayMapArray.getView();
		
	auto rayHitIndexLambda = [ = ] __cuda_callable__( const int threadIndex ) mutable
    {
		// first, find which triangle our thread is working on
		const int triangleIndex = threadToTriangleMapView[ threadIndex ];
		// what is the first thread working on our triangle?
		const int firstThreadOnTriangle = threadsPerTriangleScanView[ triangleIndex ];
		// where should our portion start?
		const int taskStart = ( threadIndex - firstThreadOnTriangle ) * raysPerThreadLimit;
		
		// transform into the coordinate system of the LBM grid
		const float ax = axView[ triangleIndex ] - Info.ox;
		const float ay = ayView[ triangleIndex ] - Info.oy;
		const float az = azView[ triangleIndex ] - Info.oz;
		const float bx = bxView[ triangleIndex ] - Info.ox;
		const float by = byView[ triangleIndex ] - Info.oy;
		const float bz = bzView[ triangleIndex ] - Info.oz;
		const float cx = cxView[ triangleIndex ] - Info.ox;
		const float cy = cyView[ triangleIndex ] - Info.oy;
		const float cz = czView[ triangleIndex ] - Info.oz;
		// transform STL floats to integer grid that is 100x finer than the LBM grid to prevent float errors
		// make the STL coords odd, rays will be even, this prevents hitting a vortex
		const float scale = 50.0f / Info.res;
		const int axInt = (int)(round( ax * scale )) * 2 + 1;
		const int ayInt = (int)(round( ay * scale )) * 2 + 1;
		const int bxInt = (int)(round( bx * scale )) * 2 + 1;
		const int byInt = (int)(round( by * scale )) * 2 + 1;
		const int cxInt = (int)(round( cx * scale )) * 2 + 1;
		const int cyInt = (int)(round( cy * scale )) * 2 + 1;
		// now figure out which rays can possibly hit the triangle -> get bounds
		const int xIntMin = TNL::max( 0, TNL::min( axInt, bxInt, cxInt, (int)(Info.cellCountX-1)*100 ) );
		const int xIntMax = TNL::min( (int)(Info.cellCountX-1)*100, TNL::max( axInt, bxInt, cxInt, 0 ) );
		const int yIntMin = TNL::max( 0, TNL::min( ayInt, byInt, cyInt, (int)(Info.cellCountY-1)*100 ) );
		const int yIntMax = TNL::min( (int)(Info.cellCountY-1)*100, TNL::max( ayInt, byInt, cyInt, 0 ) );
		const int iStartGlobal = (xIntMin + 99) / 100;
		const int iEndGlobal = xIntMax / 100 + 1;
		const int jStartGlobal = (yIntMin + 99) / 100;
		const int jEndGlobal = yIntMax / 100 + 1;
		// Prepare cayIntculation of the intersection yes no detection
		// Here we will have to switch to long long because they get multiplied and an integer could overflow if a triangle is bigger than 300 cells
		// A long long is large enough if the triangle is up to about 20M cells
		// Transform the triangle into coordinate system where the first ray is [iMin, jMin]
		const long long xLongMin = 100LL * (long long)iStartGlobal;
		const long long yLongMin = 100LL * (long long)jStartGlobal;
		const long long axLong = axInt - xLongMin;
		const long long ayLong = ayInt - yLongMin;
		const long long bxLong = bxInt - xLongMin;
		const long long byLong = byInt - yLongMin;
		const long long cxLong = cxInt - xLongMin;
		const long long cyLong = cyInt - yLongMin;
		const long long abxLong = bxLong - axLong;
		const long long abyLong = byLong - ayLong;
		const long long bcxLong = cxLong - bxLong;
		const long long bcyLong = cyLong - byLong;
		const long long caxLong = axLong - cxLong;
		const long long cayLong = ayLong - cyLong;
		// Projection of the triangle area, if it is zero, there are no intersections
		const long long signedArea = abxLong * bcyLong - abyLong * bcxLong;
		if ( signedArea == 0LL ) return; 
		long long qzLong = 1LL;
		if ( signedArea < 0LL ) qzLong = -1LL; // now ABCQ has positive volume
		// Calculate edge functions for the first ray [iMin, jMin]
		const long long wab0 = qzLong * ( abxLong * ( -byLong ) - abyLong * ( -bxLong ) );
		const long long wbc0 = qzLong * ( bcxLong * ( -cyLong ) - bcyLong * ( -cxLong ) );
		const long long wca0 = qzLong * ( caxLong * ( -ayLong ) - cayLong * ( -axLong ) );
		// Derivatives of the edge function with respect to i and j
		// We will be adding this each time we do a step in i or j direction
		const long long dwab_di = - qzLong * abyLong * 100LL;
		const long long dwab_dj = qzLong * abxLong * 100LL;
		const long long dwbc_di = - qzLong * bcyLong * 100LL;
		const long long dwbc_dj = qzLong * bcxLong * 100LL;
		const long long dwca_di = - qzLong * cayLong * 100LL;
		const long long dwca_dj = qzLong * caxLong * 100LL;
		// Prepare calculation of the intersection coordinate
		const float v1x = bx - ax;
		const float v1y = by - ay;
		const float v1z = bz - az;
		const float v2x = cx - ax;
		const float v2y = cy - ay;
		const float v2z = cz - az;
		const float nx = v1y * v2z - v1z * v2y;
		const float ny = v1z * v2x - v1x * v2z;
		const float nz = v1x * v2y - v1y * v2x;
		const float maxZ = TNL::max(az, bz, cz);
		const float minZ = TNL::min(az, bz, cz);
		const float midZ = 0.5f * (maxZ + minZ);
		
		float rayZ;
		
		// Everything about the triangle is prepared now
		// Find where we need to start from
		const int iSpan = iEndGlobal - iStartGlobal;
		const int jSpan = jEndGlobal - jStartGlobal;
		
		const int taskLast = TNL::min(taskStart + raysPerThreadLimit, iSpan * jSpan) - 1;
		
		const int jStartThread = jStartGlobal + (taskStart / iSpan);
		const int iStartThread = iStartGlobal + (taskStart % iSpan);

		const int jEndThread = jStartGlobal + (taskLast / iSpan) + 1;
		const int iEndThread = iStartGlobal + (taskLast % iSpan) + 1;
		
		// we will be changing this each time j changes
		int iStartJ, iEndJ;
		long long wabJ = wab0 + (long long)(jStartThread - jStartGlobal) * dwab_dj; 
		long long wbcJ = wbc0 + (long long)(jStartThread - jStartGlobal) * dwbc_dj;
		long long wcaJ = wca0 + (long long)(jStartThread - jStartGlobal) * dwca_dj;
		
		// we will be changing this each time j or i changes
		long long wab, wbc, wca;
		bool rayHit;
		
		for ( int j = jStartThread; j < jEndThread; j++ )
		{
			if ( j == jStartThread ) 
			{
				iStartJ = iStartThread;
				wab = wabJ + (long long)(iStartThread - iStartGlobal) * dwab_di;
				wbc = wbcJ + (long long)(iStartThread - iStartGlobal) * dwbc_di;
				wca = wcaJ + (long long)(iStartThread - iStartGlobal) * dwca_di;
			}
			else 
			{
				iStartJ = iStartGlobal;
				wab = wabJ;
				wbc = wbcJ;
				wca = wcaJ;
			}
			if ( j == jEndThread - 1 ) 
				iEndJ = iEndThread; 
			else 
				iEndJ = iEndGlobal;
			// possible optimization for later:
			// here we are starting to check all elements of the row
			// instead we can just find the first true element and last true element
			for ( int i = iStartJ; i < iEndJ; i++ )
			{
				rayHit = getRayHitYesNo( i, j, wab, wbc, wca, 
										axInt, ayInt, bxInt, byInt, cxInt, cyInt, 
										abxLong, abyLong, bcxLong, bcyLong, caxLong, cayLong );
				if ( rayHit ) 
				{
					const int rayIndex = j * Info.cellCountX + i;
					int writePosition = hitCounterScanView( rayIndex );
					writePosition += TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(hitCounterTempView( rayIndex ), 1);
					const float rayX = i * Info.res;
					const float rayY = j * Info.res;
					
					if (nz != 0.0f) 
					{
						rayZ = az - (nx * (rayX - ax) + ny * (rayY - ay)) / nz;
						if ( rayZ > maxZ ) rayZ = maxZ;
						else if ( rayZ < minZ ) rayZ = minZ;
					}
					else rayZ = midZ;
					int k = (int)ceilf(rayZ / Info.res);
					rayMapView( writePosition ) = k;
				}
				// add the increments after ending one i pass - we will be increasing i by 1
				wab = wab + dwab_di;
				wbc = wbc + dwbc_di;
				wca = wca + dwca_di;
			}
			// add the increments after ending one j pass - we will be increasing j by 1
			wabJ = wabJ + dwab_dj;
			wbcJ = wbcJ + dwbc_dj;
			wcaJ = wcaJ + dwca_dj;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, threadCount, rayHitIndexLambda );
	
	// sort the intersections in ascending order
	auto rayLambda = [=] __cuda_callable__ ( const int rayIndex ) mutable
	{
		const int startingPoint = hitCounterScanView( rayIndex );
		const int hitCount = hitCounterScanView( rayIndex + 1 ) - startingPoint;
		// sort
		for ( int layer = 1; layer < hitCount; layer++ ) 
		{
			int key = rayMapView[startingPoint + layer];
			int slider = layer - 1;
			while ( slider >= 0 && rayMapView[startingPoint + slider] > key ) 
			{
				rayMapView[startingPoint + slider + 1] = rayMapView[startingPoint + slider];
				slider = slider - 1;
			}
			rayMapView[startingPoint + slider + 1] = key;
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, Info.cellCountX * Info.cellCountY, rayLambda );	
}

void sumRayMaps( rayMapStruct &rayMapSum, rayMapStruct &rayMapBonus )
{
	// add rayMapBonus into rayMapSum as unification of all their solid intervals
	const int rayCountSum =	rayMapSum.hitCounterScanArray.getSize() - 1;
	const int rayCountBonus = rayMapBonus.hitCounterScanArray.getSize() - 1;
	if ( rayCountSum != rayCountBonus ) throw std::runtime_error("sumRayMaps failed: ray maps have different ray counts.");
	const int rayCount = rayCountSum;

	// Keep the original input arrays unchanged until both passes are finished.
	auto rayMapSumView = rayMapSum.rayMapArray.getConstView();
	auto hitCounterScanSumView = rayMapSum.hitCounterScanArray.getConstView();
	auto rayMapBonusView = rayMapBonus.rayMapArray.getConstView();
	auto hitCounterScanBonusView = rayMapBonus.hitCounterScanArray.getConstView();

	// -------------------------------------------------------------------------
	// First pass: count the number of result intersections on every ray
	// -------------------------------------------------------------------------

	IntArrayType resultHitCounterScanArray;
	resultHitCounterScanArray.setSize( rayCount + 1 );
	resultHitCounterScanArray.setValue( 0 );

	auto resultHitCounterScanView = resultHitCounterScanArray.getView();

	auto countLambda = [=] __cuda_callable__ ( const int rayIndex ) mutable
	{
		int sumIndex = hitCounterScanSumView( rayIndex );
		const int sumEnd = hitCounterScanSumView( rayIndex + 1 );
		int bonusIndex = hitCounterScanBonusView( rayIndex );
		const int bonusEnd = hitCounterScanBonusView( rayIndex + 1 );
		int resultHitCount = 0;
		bool intervalActive = false;
		int lastEnd = 0;
		
		// Merge two sorted lists of intervals without storing them locally.
		while ( sumIndex < sumEnd || bonusIndex < bonusEnd )
		{
			int currentStart;
			int currentEnd;

			// Pick the interval with the earlier start.
			if ( bonusIndex >= bonusEnd || ( sumIndex < sumEnd && rayMapSumView( sumIndex ) <= rayMapBonusView( bonusIndex )))
			{
				currentStart = rayMapSumView( sumIndex );
				currentEnd   = rayMapSumView( sumIndex + 1 );
				sumIndex += 2;
			}
			else
			{
				currentStart = rayMapBonusView( bonusIndex );
				currentEnd   = rayMapBonusView( bonusIndex + 1 );
				bonusIndex += 2;
			}

			if ( ! intervalActive )
			{
				// Begin the first result interval.
				lastEnd = currentEnd;
				intervalActive = true;
			}
			else if ( currentStart <= lastEnd )
			{
				// The intervals overlap or touch.
				if ( currentEnd > lastEnd )
					lastEnd = currentEnd;
			}
			else
			{
				// The previous interval is complete.
				resultHitCount += 2;
				lastEnd = currentEnd;
			}
		}

		if ( intervalActive ) resultHitCount += 2;
		resultHitCounterScanView( rayIndex ) = resultHitCount;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, rayCount, countLambda );

	TNL::Algorithms::inplaceExclusiveScan( resultHitCounterScanArray, 0, rayCount + 1, TNL::Plus{} );

	const int resultTotalHitCount = resultHitCounterScanArray.getElement( rayCount );

	// -------------------------------------------------------------------------
	// Allocate the exact amount of result storage
	// -------------------------------------------------------------------------

	IntArrayType resultRayMapArray;
	resultRayMapArray.setSize( resultTotalHitCount );
	auto resultRayMapView =	resultRayMapArray.getView();
	auto resultHitCounterScanConstView = resultHitCounterScanArray.getConstView();

	// -------------------------------------------------------------------------
	// Second pass: repeat the merge and write the result intervals
	// -------------------------------------------------------------------------

	auto fillLambda = [=] __cuda_callable__ ( const int rayIndex ) mutable
	{
		int sumIndex = hitCounterScanSumView( rayIndex );
		const int sumEnd = hitCounterScanSumView( rayIndex + 1 );
		int bonusIndex = hitCounterScanBonusView( rayIndex );
		const int bonusEnd = hitCounterScanBonusView( rayIndex + 1 );
		int resultIndex = resultHitCounterScanConstView( rayIndex );
		bool intervalActive = false;
		int lastStart = 0;
		int lastEnd = 0;

		while ( sumIndex < sumEnd || bonusIndex < bonusEnd )
		{
			int currentStart;
			int currentEnd;

			// Pick the interval with the earlier start.
			if ( bonusIndex >= bonusEnd || ( sumIndex < sumEnd && rayMapSumView( sumIndex ) <= rayMapBonusView( bonusIndex )))
			{
				currentStart = rayMapSumView( sumIndex );
				currentEnd   = rayMapSumView( sumIndex + 1 );
				sumIndex += 2;
			}
			else
			{
				currentStart = rayMapBonusView( bonusIndex );
				currentEnd   = rayMapBonusView( bonusIndex + 1 );
				bonusIndex += 2;
			}

			if ( ! intervalActive )
			{
				lastStart = currentStart;
				lastEnd = currentEnd;
				intervalActive = true;
			}
			else if ( currentStart <= lastEnd )
			{
				// Extend the active interval if necessary.
				if ( currentEnd > lastEnd )
					lastEnd = currentEnd;
			}
			else
			{
				// Write the completed interval.
				resultRayMapView( resultIndex     ) = lastStart;
				resultRayMapView( resultIndex + 1 ) = lastEnd;
				resultIndex += 2;

				// Begin the next interval.
				lastStart = currentStart;
				lastEnd = currentEnd;
			}
		}

		// Write the final active interval.
		if ( intervalActive )
		{
			resultRayMapView( resultIndex     ) = lastStart;
			resultRayMapView( resultIndex + 1 ) = lastEnd;
		}
	};

	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, rayCount, fillLambda );

	// -------------------------------------------------------------------------
	// Replace rayMapSum with the compact result
	// -------------------------------------------------------------------------

	rayMapSum.rayMapArray = resultRayMapArray;
	rayMapSum.hitCounterScanArray = resultHitCounterScanArray;
	rayMapSum.totalHitCount = resultTotalHitCount;
}

void initializeVoxelizers( std::vector<VoxelizerStruct> &voxelizers, const std::vector<GridStruct> &grids, std::vector<STLStruct> &gridStaticSTLs, const int level )
{
	std::cout << "Initializing voxelizer for grid level " << level << std::endl; 
	const bool iAmFinest = ( level == GRID_LEVEL_COUNT - 1 );
	
	VoxelizerStruct &Voxelizer = voxelizers[ level ];
	Voxelizer.Info = grids[ level ].Info;
	
	const int rayMapCount = gridStaticSTLs.size();
	Voxelizer.rayMaps.resize( rayMapCount );
	
	unsigned long long totalElementCount = 0LL;
	for ( int rayMapIndex = 0; rayMapIndex < rayMapCount; rayMapIndex++ ) 
	{
		Voxelizer.rayMaps[rayMapIndex].gridID = Voxelizer.Info.gridID;
		voxelizeSTL( Voxelizer.rayMaps[rayMapIndex], gridStaticSTLs[rayMapIndex], Voxelizer );
		totalElementCount += (long long)Voxelizer.rayMaps[rayMapIndex].rayMapArray.getSize() + (long long)Voxelizer.rayMaps[rayMapIndex].hitCounterScanArray.getSize();
	}
	
	Voxelizer.rayMapTotal.gridID = Voxelizer.Info.gridID;
	Voxelizer.rayMapTotal = Voxelizer.rayMaps[0];
	for ( int bonusIndex = 1; bonusIndex < rayMapCount; bonusIndex++ )
	{
		sumRayMaps( Voxelizer.rayMapTotal, Voxelizer.rayMaps[bonusIndex] );
	}
	totalElementCount += (long long)Voxelizer.rayMapTotal.rayMapArray.getSize() + (long long)Voxelizer.rayMapTotal.hitCounterScanArray.getSize();
	
	unsigned long long memoryBytes = 4LL * totalElementCount; // 1 int has 4 Bytes
	std::cout << "	Done, allocated on GPU, it takes " << memoryBytes / 1048576.0 << " MiB" << std::endl;
	std::cout << std::endl;
	
	if ( !iAmFinest ) initializeVoxelizers( voxelizers, grids, gridStaticSTLs, level + 1 );
}
