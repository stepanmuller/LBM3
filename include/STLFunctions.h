#pragma once

#include "./types.h"

void checkSTLEdges( STLStruct &STL )
// For every edge, counts number of triangles that share it. Must be always 2 for a closed STL.
{
	std::cout << "	Starting STL check for faulty edges" << std::endl;
	auto axArrayView = STL.axArray.getConstView();
	auto ayArrayView = STL.ayArray.getConstView();
	auto azArrayView = STL.azArray.getConstView();
	auto bxArrayView = STL.bxArray.getConstView();
	auto byArrayView = STL.byArray.getConstView();
	auto bzArrayView = STL.bzArray.getConstView();
	auto cxArrayView = STL.cxArray.getConstView();
	auto cyArrayView = STL.cyArray.getConstView();
	auto czArrayView = STL.czArray.getConstView();
	
	IntArray2DType edgeCounterArray;
	edgeCounterArray.setSizes( STL.triangleCount, 3 );
	edgeCounterArray.setValue(0);
	auto edgeCounterArrayView = edgeCounterArray.getView();

    auto counterLambda = [ = ] __cuda_callable__( const int triangle1Index ) mutable
    {		
		float triangle1[9];
		triangle1[0] = axArrayView[ triangle1Index ];
		triangle1[1] = ayArrayView[ triangle1Index ];
		triangle1[2] = azArrayView[ triangle1Index ];
		triangle1[3] = bxArrayView[ triangle1Index ];
		triangle1[4] = byArrayView[ triangle1Index ];
		triangle1[5] = bzArrayView[ triangle1Index ];
		triangle1[6] = cxArrayView[ triangle1Index ];
		triangle1[7] = cyArrayView[ triangle1Index ];
		triangle1[8] = czArrayView[ triangle1Index ];
		
		for ( int triangle2Index = 0; triangle2Index < STL.triangleCount; triangle2Index++ ) 
		{
			float triangle2[9];
			triangle2[0] = axArrayView[ triangle2Index ];
			triangle2[1] = ayArrayView[ triangle2Index ];
			triangle2[2] = azArrayView[ triangle2Index ];
			triangle2[3] = bxArrayView[ triangle2Index ];
			triangle2[4] = byArrayView[ triangle2Index ];
			triangle2[5] = bzArrayView[ triangle2Index ];
			triangle2[6] = cxArrayView[ triangle2Index ];
			triangle2[7] = cyArrayView[ triangle2Index ];
			triangle2[8] = czArrayView[ triangle2Index ];
			
			for ( int edge1 = 0; edge1 < 3; edge1++ )
			{
				for ( int edge2 = 0; edge2 < 3; edge2++ )
				{
					float ax1 = triangle1[3*edge1];
					float ay1 = triangle1[3*edge1+1];
					float az1 = triangle1[3*edge1+2];
					float bx1 = triangle1[3*((edge1+1)%3)];
					float by1 = triangle1[3*((edge1+1)%3)+1];
					float bz1 = triangle1[3*((edge1+1)%3)+2];
					
					float ax2 = triangle2[3*edge2];
					float ay2 = triangle2[3*edge2+1];
					float az2 = triangle2[3*edge2+2];
					float bx2 = triangle2[3*((edge2+1)%3)];
					float by2 = triangle2[3*((edge2+1)%3)+1];
					float bz2 = triangle2[3*((edge2+1)%3)+2];
					// testing same orientation of the edge
					if (ax1 == ax2 && ay1 == ay2 && az1 == az2 && bx1 == bx2 && by1 == by2 && bz1 == bz2) 
					{
						TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(edgeCounterArrayView(triangle1Index, edge1), 1);
					}
					// testing reverse orientation of the edge
					if (ax1 == bx2 && ay1 == by2 && az1 == bz2 && bx1 == ax2 && by1 == ay2 && bz1 == az2) 
					{
						TNL::Algorithms::AtomicOperations<TNL::Devices::Cuda>::add(edgeCounterArrayView(triangle1Index, edge1), 1);
					}
				}
			}
		}
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, counterLambda );
	int errorCounter = 0;
	for ( int triangleIndex = 0; triangleIndex < STL.triangleCount; triangleIndex++ )
    {
		int ABcount = edgeCounterArray.getElement( triangleIndex, 0 );
		int BCcount = edgeCounterArray.getElement( triangleIndex, 1 );
		int CAcount = edgeCounterArray.getElement( triangleIndex, 2 );
		if (ABcount != 2 || BCcount != 2 || CAcount != 2)
		{
			errorCounter++;
			std::cout << "	Faulty edge on triangle " << triangleIndex << ", ABcount: " << ABcount << ", BCcount: " << BCcount << ", CAcount: " << CAcount << std::endl;
		}
	}    
	if ( errorCounter == 0 ) std::cout<< "	Check finished, number of faulty edges: " << errorCounter << std::endl; 
	else std::cout<< "	Check failed, number of faulty edges: " << errorCounter << std::endl; 
	if ( errorCounter > 0 ) throw std::runtime_error("Check failed, the STL has some faulty edges which aren't shared between exactly two triangles. This means the STL is not closed. Please fix the STL file.");
}

void readSTL( STLStruct &STL, const std::string &filename )
{
	STLStructCPU STLCPU;
	
	std::cout << "Reading STL: " << filename << std::endl;
	std::ifstream file(filename, std::ios::binary);
	if ( !file.is_open() ) throw std::runtime_error("Failed to open STL file");
	
	 // Skip header
    char header[80];
    file.read( header, 80 );

    uint32_t triangleCount32;
	file.read( reinterpret_cast<char*>(&triangleCount32), sizeof(uint32_t) );
	
	int initialTriangleCount = static_cast<int>( triangleCount32 );
	std::cout<<"	Initial triangle count: " << initialTriangleCount << std::endl;
	// Track excluded triangles (exclude triangles whose at least 2 points are identical)
	int excludedTriangleCount = 0;
	
    STLCPU.axArray = FloatArrayTypeCPU( initialTriangleCount );
    STLCPU.ayArray = FloatArrayTypeCPU( initialTriangleCount );
    STLCPU.azArray = FloatArrayTypeCPU( initialTriangleCount );

    STLCPU.bxArray = FloatArrayTypeCPU( initialTriangleCount );
    STLCPU.byArray = FloatArrayTypeCPU( initialTriangleCount );
    STLCPU.bzArray = FloatArrayTypeCPU( initialTriangleCount );

    STLCPU.cxArray = FloatArrayTypeCPU( initialTriangleCount );
    STLCPU.cyArray = FloatArrayTypeCPU( initialTriangleCount );
    STLCPU.czArray = FloatArrayTypeCPU( initialTriangleCount );
    
    // Initialize minmax
    STLCPU.Bounds.xMin = std::numeric_limits<float>::max();
	STLCPU.Bounds.yMin = std::numeric_limits<float>::max();
	STLCPU.Bounds.zMin = std::numeric_limits<float>::max();

	STLCPU.Bounds.xMax = std::numeric_limits<float>::lowest();
	STLCPU.Bounds.yMax = std::numeric_limits<float>::lowest();
	STLCPU.Bounds.zMax = std::numeric_limits<float>::lowest();
	
	STLCPU.Bounds.rxMax = std::numeric_limits<float>::lowest();
	STLCPU.Bounds.ryMax = std::numeric_limits<float>::lowest();
	STLCPU.Bounds.rzMax = std::numeric_limits<float>::lowest();
	
	int writeIndex = 0;

    for ( int triangle = 0; triangle < initialTriangleCount; triangle++ )
    {
        float ax, ay, az;
        float bx, by, bz;
        float cx, cy, cz;
        uint16_t attr;

		// Skip normal (nx, ny, nz) = 3 floats = 12 bytes
		file.seekg(12, std::ios::cur);

        file.read( reinterpret_cast<char*>(&ax), 4 );
        file.read( reinterpret_cast<char*>(&ay), 4 );
        file.read( reinterpret_cast<char*>(&az), 4 );

        file.read( reinterpret_cast<char*>(&bx), 4 );
        file.read( reinterpret_cast<char*>(&by), 4 );
        file.read( reinterpret_cast<char*>(&bz), 4 );

        file.read( reinterpret_cast<char*>(&cx), 4 );
        file.read( reinterpret_cast<char*>(&cy), 4 );
        file.read( reinterpret_cast<char*>(&cz), 4 );

        file.read( reinterpret_cast<char*>(&attr), 2 );
		
		if ( (ax==bx && ay==by && az==bz) || (cx==bx && cy==by && cz==bz) || (cx==ax && cy==ay && cz==az) )
        {
			excludedTriangleCount++;
			continue;
		}
		
        STLCPU.axArray[writeIndex] = ax;
        STLCPU.ayArray[writeIndex] = ay;
        STLCPU.azArray[writeIndex] = az;

        STLCPU.bxArray[writeIndex] = bx;
        STLCPU.byArray[writeIndex] = by;
        STLCPU.bzArray[writeIndex] = bz;

        STLCPU.cxArray[writeIndex] = cx;
        STLCPU.cyArray[writeIndex] = cy;
        STLCPU.czArray[writeIndex] = cz;
        
        writeIndex++;
        
         // Update bounding box (vertices only)
        STLCPU.Bounds.xMin = std::min(STLCPU.Bounds.xMin, std::min({ax, bx, cx}));
        STLCPU.Bounds.yMin = std::min(STLCPU.Bounds.yMin, std::min({ay, by, cy}));
        STLCPU.Bounds.zMin = std::min(STLCPU.Bounds.zMin, std::min({az, bz, cz}));

        STLCPU.Bounds.xMax = std::max(STLCPU.Bounds.xMax, std::max({ax, bx, cx}));
        STLCPU.Bounds.yMax = std::max(STLCPU.Bounds.yMax, std::max({ay, by, cy}));
        STLCPU.Bounds.zMax = std::max(STLCPU.Bounds.zMax, std::max({az, bz, cz}));
        
        float rx = std::sqrt( std::max({ ay*ay+az*az, by*by+bz*bz, cy*cy+cz*cz }) );
        float ry = std::sqrt( std::max({ ax*ax+az*az, bx*bx+bz*bz, cx*cx+cz*cz }) );
        float rz = std::sqrt( std::max({ ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy }) );
        STLCPU.Bounds.rxMax = std::max(STLCPU.Bounds.rxMax, rx);
        STLCPU.Bounds.ryMax = std::max(STLCPU.Bounds.ryMax, ry);
        STLCPU.Bounds.rzMax = std::max(STLCPU.Bounds.rzMax, rz);
    }
    int triangleCount = initialTriangleCount - excludedTriangleCount;
    STLCPU.triangleCount = triangleCount;
    STLCPU.axArray.resize(triangleCount);
    STLCPU.ayArray.resize(triangleCount);
    STLCPU.azArray.resize(triangleCount);
    STLCPU.bxArray.resize(triangleCount);
    STLCPU.byArray.resize(triangleCount);
    STLCPU.bzArray.resize(triangleCount);
    STLCPU.cxArray.resize(triangleCount);
    STLCPU.cyArray.resize(triangleCount);
    STLCPU.czArray.resize(triangleCount);
    
    std::cout<<"	Excluded triangles with zero area: " << excludedTriangleCount << std::endl;    
    std::cout<<"	Final triangle count: " << triangleCount << std::endl;
    std::cout << "	xMin xMax: " << STLCPU.Bounds.xMin << " " << STLCPU.Bounds.xMax << "\n";
    std::cout << "	yMin yMax: " << STLCPU.Bounds.yMin << " " << STLCPU.Bounds.yMax << "\n";
    std::cout << "	zMin zMax: " << STLCPU.Bounds.zMin << " " << STLCPU.Bounds.zMax << "\n";
    std::cout << "	rxMax ryMax rzMax: " << STLCPU.Bounds.rxMax << " " << STLCPU.Bounds.ryMax << " " << STLCPU.Bounds.rzMax << "\n";
   
    STL = STLStruct( STLCPU );
	checkSTLEdges( STL );
	STL.raysPerTriangleCounterArray.setSize( STL.triangleCount );
	STL.raysPerTriangleCounterArray.setValue( 0 );
	STL.threadToTriangleMapArray.setSize( STL.triangleCount * STL.threadsToTrianglesRatio );
	STL.threadToTriangleMapArray.setValue( 0 );
	unsigned long long memoryBytes = 4LL * 9LL * STL.triangleCount; // 1 float has 4 Bytes, 9 floats per triangle
	std::cout << "	Done, allocated on GPU, it takes " << memoryBytes / 1048576.0 << " MiB" << std::endl;
	std::cout << std::endl;
}

// Rotate STL, overwrite it
void rotateSTLAlongX( STLStruct &STL, float &radians )
{
	auto ayView = STL.ayArray.getView();
	auto azView = STL.azArray.getView();
	auto byView = STL.byArray.getView();
	auto bzView = STL.bzArray.getView();
	auto cyView = STL.cyArray.getView();
	auto czView = STL.czArray.getView();
	
	const float s = sinf(radians);
	const float c = cosf(radians);
	
	auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
	{
		const float ay = ayView[ triangleIndex ];
		const float az = azView[ triangleIndex ];
		const float by = byView[ triangleIndex ];
		const float bz = bzView[ triangleIndex ];
		const float cy = cyView[ triangleIndex ];
		const float cz = czView[ triangleIndex ];
		
		const float newAy = ay * c - az * s;
		const float newAz = ay * s + az * c;
		const float newBy = by * c - bz * s;
		const float newBz = by * s + bz * c;
		const float newCy = cy * c - cz * s;
		const float newCz = cy * s + cz * c;
		
		ayView[ triangleIndex ] = newAy;
		azView[ triangleIndex ] = newAz;
		byView[ triangleIndex ] = newBy;
		bzView[ triangleIndex ] = newBz;
		cyView[ triangleIndex ] = newCy;
		czView[ triangleIndex ] = newCz;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rotateLambda );
}

void rotateSTLAlongY( STLStruct &STL, float &radians )
{
	auto axView = STL.axArray.getView();
	auto azView = STL.azArray.getView();
	auto bxView = STL.bxArray.getView();
	auto bzView = STL.bzArray.getView();
	auto cxView = STL.cxArray.getView();
	auto czView = STL.czArray.getView();
	
	const float s = sinf(radians);
	const float c = cosf(radians);
	
	auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
	{
		const float ax = axView[ triangleIndex ];
		const float az = azView[ triangleIndex ];
		const float bx = bxView[ triangleIndex ];
		const float bz = bzView[ triangleIndex ];
		const float cx = cxView[ triangleIndex ];
		const float cz = czView[ triangleIndex ];
		
		const float newAx = ax * c + az * s;
		const float newAz = -ax * s + az * c;
		const float newBx = bx * c + bz * s;
		const float newBz = -bx * s + bz * c;
		const float newCx = cx * c + cz * s;
		const float newCz = -cx * s + cz * c;
		
		axView[ triangleIndex ] = newAx;
		azView[ triangleIndex ] = newAz;
		bxView[ triangleIndex ] = newBx;
		bzView[ triangleIndex ] = newBz;
		cxView[ triangleIndex ] = newCx;
		czView[ triangleIndex ] = newCz;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rotateLambda );
}

void rotateSTLAlongZ( STLStruct &STL, const float &radians )
{
	auto axView = STL.axArray.getView();
	auto ayView = STL.ayArray.getView();
	auto bxView = STL.bxArray.getView();
	auto byView = STL.byArray.getView();
	auto cxView = STL.cxArray.getView();
	auto cyView = STL.cyArray.getView();
	
	const float s = sinf(radians);
	const float c = cosf(radians);
	
    auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
		const float ax = axView[ triangleIndex ];
		const float ay = ayView[ triangleIndex ];
		const float bx = bxView[ triangleIndex ];
		const float by = byView[ triangleIndex ];
		const float cx = cxView[ triangleIndex ];
		const float cy = cyView[ triangleIndex ];
		
		const float newAx = ax * c - ay * s;
		const float newAy = ax * s + ay * c;
		const float newBx = bx * c - by * s;
		const float newBy = bx * s + by * c;
		const float newCx = cx * c - cy * s;
		const float newCy = cx * s + cy * c;
		
		axView[ triangleIndex ] = newAx;
		ayView[ triangleIndex ] = newAy;
		bxView[ triangleIndex ] = newBx;
		byView[ triangleIndex ] = newBy;
		cxView[ triangleIndex ] = newCx;
		cyView[ triangleIndex ] = newCy;
	};
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STL.triangleCount, rotateLambda );
}

// Versions with source and target that don't overwrite the source STL
void rotateSTLAlongX( STLStruct &STLTarget, const STLStruct &STLSource, const float radians )
{
    auto sourceAyView = STLSource.ayArray.getConstView();
    auto sourceAzView = STLSource.azArray.getConstView();
    auto sourceByView = STLSource.byArray.getConstView();
    auto sourceBzView = STLSource.bzArray.getConstView();
    auto sourceCyView = STLSource.cyArray.getConstView();
    auto sourceCzView = STLSource.czArray.getConstView();
    
    auto targetAyView = STLTarget.ayArray.getView();
    auto targetAzView = STLTarget.azArray.getView();
    auto targetByView = STLTarget.byArray.getView();
    auto targetBzView = STLTarget.bzArray.getView();
    auto targetCyView = STLTarget.cyArray.getView();
    auto targetCzView = STLTarget.czArray.getView();
    
    const float s = sinf(radians);
    const float c = cosf(radians);
    
    auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
        const float ay = sourceAyView[ triangleIndex ];
        const float az = sourceAzView[ triangleIndex ];
        const float by = sourceByView[ triangleIndex ];
        const float bz = sourceBzView[ triangleIndex ];
        const float cy = sourceCyView[ triangleIndex ];
        const float cz = sourceCzView[ triangleIndex ];
        
        const float newAy = ay * c - az * s;
        const float newAz = ay * s + az * c;
        const float newBy = by * c - bz * s;
        const float newBz = by * s + bz * c;
        const float newCy = cy * c - cz * s;
        const float newCz = cy * s + cz * c;
        
        targetAyView[ triangleIndex ] = newAy;
        targetAzView[ triangleIndex ] = newAz;
        targetByView[ triangleIndex ] = newBy;
        targetBzView[ triangleIndex ] = newBz;
        targetCyView[ triangleIndex ] = newCy;
        targetCzView[ triangleIndex ] = newCz;
    };
    
    TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STLSource.triangleCount, rotateLambda );
}

void rotateSTLAlongY( STLStruct &STLTarget, const STLStruct &STLSource, const float radians )
{
    auto sourceAxView = STLSource.axArray.getConstView();
    auto sourceAzView = STLSource.azArray.getConstView();
    auto sourceBxView = STLSource.bxArray.getConstView();
    auto sourceBzView = STLSource.bzArray.getConstView();
    auto sourceCxView = STLSource.cxArray.getConstView();
    auto sourceCzView = STLSource.czArray.getConstView();
    
    auto targetAxView = STLTarget.axArray.getView();
    auto targetAzView = STLTarget.azArray.getView();
    auto targetBxView = STLTarget.bxArray.getView();
    auto targetBzView = STLTarget.bzArray.getView();
    auto targetCxView = STLTarget.cxArray.getView();
    auto targetCzView = STLTarget.czArray.getView();
    
    const float s = sinf(radians);
    const float c = cosf(radians);
    
    auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
        const float ax = sourceAxView[ triangleIndex ];
        const float az = sourceAzView[ triangleIndex ];
        const float bx = sourceBxView[ triangleIndex ];
        const float bz = sourceBzView[ triangleIndex ];
        const float cx = sourceCxView[ triangleIndex ];
        const float cz = sourceCzView[ triangleIndex ];

        const float newAx = ax * c + az * s;
        const float newAz = -ax * s + az * c;
        const float newBx = bx * c + bz * s;
        const float newBz = -bx * s + bz * c;
        const float newCx = cx * c + cz * s;
        const float newCz = -cx * s + cz * c;

        targetAxView[ triangleIndex ] = newAx;
        targetAzView[ triangleIndex ] = newAz;
        targetBxView[ triangleIndex ] = newBx;
        targetBzView[ triangleIndex ] = newBz;
        targetCxView[ triangleIndex ] = newCx;
        targetCzView[ triangleIndex ] = newCz;
    };
    
    TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STLSource.triangleCount, rotateLambda );
}

void rotateSTLAlongZ( STLStruct &STLTarget, const STLStruct &STLSource, const float radians )
{
    auto sourceAxView = STLSource.axArray.getConstView();
    auto sourceAyView = STLSource.ayArray.getConstView();
    auto sourceBxView = STLSource.bxArray.getConstView();
    auto sourceByView = STLSource.byArray.getConstView();
    auto sourceCxView = STLSource.cxArray.getConstView();
    auto sourceCyView = STLSource.cyArray.getConstView();
    
    auto targetAxView = STLTarget.axArray.getView();
    auto targetAyView = STLTarget.ayArray.getView();
    auto targetBxView = STLTarget.bxArray.getView();
    auto targetByView = STLTarget.byArray.getView();
    auto targetCxView = STLTarget.cxArray.getView();
    auto targetCyView = STLTarget.cyArray.getView();
    
    const float s = sinf(radians);
    const float c = cosf(radians);
    
    auto rotateLambda = [ = ] __cuda_callable__( const int triangleIndex ) mutable
    {
        const float ax = sourceAxView[ triangleIndex ];
        const float ay = sourceAyView[ triangleIndex ];
        const float bx = sourceBxView[ triangleIndex ];
        const float by = sourceByView[ triangleIndex ];
        const float cx = sourceCxView[ triangleIndex ];
        const float cy = sourceCyView[ triangleIndex ];
        
        const float newAx = ax * c - ay * s;
        const float newAy = ax * s + ay * c;
        const float newBx = bx * c - by * s;
        const float newBy = bx * s + by * c;
        const float newCx = cx * c - cy * s;
        const float newCy = cx * s + cy * c;
        
        targetAxView[ triangleIndex ] = newAx;
        targetAyView[ triangleIndex ] = newAy;
        targetBxView[ triangleIndex ] = newBx;
        targetByView[ triangleIndex ] = newBy;
        targetCxView[ triangleIndex ] = newCx;
        targetCyView[ triangleIndex ] = newCy;
    };
    
    TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, STLSource.triangleCount, rotateLambda );
}
