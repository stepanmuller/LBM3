#pragma once

#include "./types.h"
#include "./markerFunctions.h"
#include "./voxelizerFunctions.h"


__cuda_callable__ void projectXYZIntoRotorFrame( float &xRotor, float &yRotor, float &zRotor, const RotorInfoStruct &InfoRotor, const InfoStruct &InfoGlobal )
{
	const float angle = (float)InfoRotor.radiansElapsed;
	const float c = cosf(angle);
    const float s = sinf(angle);
    // Original position relative to the rotation pivot.
    const float x = xRotor - InfoRotor.ox;
    const float y = yRotor - InfoRotor.oy;
    const float z = zRotor - InfoRotor.oz;
	// Rotor rotates along an axis parallel to one of the main axes
	// The axis of rotation pierces the point InfoRotor.ox, InfoRotor.oy, InfoRotor.oz
	// radiansPerSecond tell how fast the rotor rotates, in the rotor frame it seems that the global domain rotates with negative of that
	if ( InfoRotor.rotateAlongX )
	{
		yRotor = InfoRotor.oy + c * y + s * z;
        zRotor = InfoRotor.oz - s * y + c * z;
	}
	else if ( InfoRotor.rotateAlongY )
	{
		xRotor = InfoRotor.ox + c * x - s * z;
        zRotor = InfoRotor.oz + s * x + c * z;
	}
	else if ( InfoRotor.rotateAlongZ )
	{
		xRotor = InfoRotor.ox + c * x + s * y;
        yRotor = InfoRotor.oy - s * x + c * y;
	}
}

__cuda_callable__ void projectForcingIntoRotorFrame( float& gxRotor, float& gyRotor, float& gzRotor, const RotorInfoStruct& InfoRotor, const InfoStruct& InfoGlobal)
{
    const float angle = (float)InfoRotor.radiansElapsed;
    const float c = cosf(angle);
    const float s = sinf(angle);

    if (InfoRotor.rotateAlongX)
    {
        const float gy = gyRotor;
        gyRotor =  c * gy + s * gzRotor;
        gzRotor = -s * gy + c * gzRotor;
    }
    else if (InfoRotor.rotateAlongY)
    {
        const float gx = gxRotor;
        gxRotor = c * gx - s * gzRotor;
        gzRotor = s * gx + c * gzRotor;
    }
    else if (InfoRotor.rotateAlongZ)
    {
        const float gx = gxRotor;
        gxRotor =  c * gx + s * gyRotor;
        gyRotor = -s * gx + c * gyRotor;
    }
}

__cuda_callable__ void getRotorForcing( float& gxRotor, float& gyRotor, float& gzRotor, 
						const float& xGlobal, const float& yGlobal, const float& zGlobal,
						const float& rho, const float& uxPreRotor, const float& uyPreRotor, const float& uzPreRotor, 
						const RotorInfoStruct& InfoRotor, const InfoStruct& InfoGlobal)
{
    // Position relative to the rotation axis.
    const float x = xGlobal - InfoRotor.ox;
    const float y = yGlobal - InfoRotor.oy;
    const float z = zGlobal - InfoRotor.oz;

    // Converts angular velocity × distance into lattice velocity
    const float scale = InfoRotor.radiansPerSecond * InfoGlobal.dtPhys / InfoGlobal.res;

    float uxTarget = 0.f;
    float uyTarget = 0.f;
    float uzTarget = 0.f;

    if (InfoRotor.rotateAlongX)
    {
        uyTarget = -scale * z;
        uzTarget =  scale * y;
    }
    else if (InfoRotor.rotateAlongY)
    {
        uxTarget =  scale * z;
        uzTarget = -scale * x;
    }
    else if (InfoRotor.rotateAlongZ)
    {
        uxTarget = -scale * y;
        uyTarget =  scale * x;
    }

    // Momentum correction for the complete collision step
    gxRotor = rho * (uxTarget - uxPreRotor);
    gyRotor = rho * (uyTarget - uyPreRotor);
    gzRotor = rho * (uzTarget - uzPreRotor);
}

__cuda_callable__ void getRotorVelocity( float& uxTarget, float& uyTarget, float& uzTarget, 
						const float& xGlobal, const float& yGlobal, const float& zGlobal,
						const RotorInfoStruct& InfoRotor, const InfoStruct& InfoGlobal)
{
    // Position relative to the rotation axis.
    const float x = xGlobal - InfoRotor.ox;
    const float y = yGlobal - InfoRotor.oy;
    const float z = zGlobal - InfoRotor.oz;

    // Converts angular velocity × distance into lattice velocity
    const float scale = InfoRotor.radiansPerSecond * InfoGlobal.dtPhys / InfoGlobal.res;

    uxTarget = 0.f;
    uyTarget = 0.f;
    uzTarget = 0.f;

    if (InfoRotor.rotateAlongX)
    {
        uyTarget = -scale * z;
        uzTarget =  scale * y;
    }
    else if (InfoRotor.rotateAlongY)
    {
        uxTarget =  scale * z;
        uzTarget = -scale * x;
    }
    else if (InfoRotor.rotateAlongZ)
    {
        uxTarget = -scale * y;
        uyTarget =  scale * x;
    }
}

__cuda_callable__ void interpolateRotorCube( float& rotorFraction, const float x, const float y, const float z, const uint32_t packed)
{
    // Unpack the eight corner counts, each in [0, 8].
    float v[8];
    for (unsigned int corner = 0; corner < 8; corner++) v[corner] = static_cast<float>((packed >> (4u * corner)) & 0xFu);

    // Interpolate along X.
    const float v00 = v[0] + x * (v[1] - v[0]);
    const float v10 = v[2] + x * (v[3] - v[2]);
    const float v01 = v[4] + x * (v[5] - v[4]);
    const float v11 = v[6] + x * (v[7] - v[6]);

    // Interpolate along Y.
    const float v0 = v00 + y * (v10 - v00);
    const float v1 = v01 + y * (v11 - v01);

    // Interpolate along Z and normalize the count to [0, 1].
    rotorFraction = (v0 + z * (v1 - v0)) * 0.125f;
}

__cuda_callable__ void getRotorFraction( float& rotorFraction, 
										 const float& xGlobal, const float& yGlobal, const float& zGlobal, 
										 const InfoStruct& InfoGlobal, const RotorViewStruct& RotorView )
{
	const RotorInfoStruct& InfoRotor = RotorView.Info;
    const BoundsStruct& Bounds = InfoRotor.Bounds;
    const auto& rotorMapView = RotorView.rotorMapView;
    const auto& interpolationView = RotorView.interpolationView;
	
	float xRotor = xGlobal; float yRotor = yGlobal; float zRotor = zGlobal;
	projectXYZIntoRotorFrame( xRotor, yRotor, zRotor, InfoRotor, InfoGlobal );
	
    rotorFraction = 0.f;

    if (xRotor < Bounds.xMin || xRotor >= Bounds.xMax ||
        yRotor < Bounds.yMin || yRotor >= Bounds.yMax ||
        zRotor < Bounds.zMin || zRotor >= Bounds.zMax) return;

    // Position relative to rotor Bounds.Min
    const float xRelative = xRotor - Bounds.xMin;
    const float yRelative = yRotor - Bounds.yMin;
    const float zRelative = zRotor - Bounds.zMin;

    const int iInterpolation = static_cast<int>(xRelative / InfoRotor.res);
    const int jInterpolation = static_cast<int>(yRelative / InfoRotor.res);
    const int kInterpolation = static_cast<int>(zRelative / InfoRotor.res);

    // Protect against rounding up at the upper boundary
    if (iInterpolation >= InfoRotor.cellCountX ||
        jInterpolation >= InfoRotor.cellCountY ||
        kInterpolation >= InfoRotor.cellCountZ) return;

    const int blockCountX = InfoRotor.cellCountX / 4;
    const int blockCountY = InfoRotor.cellCountY / 4;

    const int iBlock = iInterpolation / 4;
    const int jBlock = jInterpolation / 4;
    const int kBlock = kInterpolation / 4;

    const int blockIndex = kBlock * blockCountX * blockCountY + jBlock * blockCountX + iBlock;

    const int rotorMap = rotorMapView(blockIndex);
    if (rotorMap < 0) return;

    const int interpolationIndex = rotorMap * 64 + (kInterpolation % 4) * 16 + (jInterpolation % 4) * 4 + (iInterpolation % 4);

    const uint32_t packed = interpolationView(interpolationIndex);

    const float xWithinCube = (xRelative / InfoRotor.res) - iInterpolation;
    const float yWithinCube = (yRelative / InfoRotor.res) - jInterpolation;
    const float zWithinCube = (zRelative / InfoRotor.res) - kInterpolation;

    interpolateRotorCube( rotorFraction, xWithinCube, yWithinCube, zWithinCube, packed );
}

__cuda_callable__ void processRotor( BCStruct &BC, const float &rho, const float &uxPreRotor, const float &uyPreRotor, const float &uzPreRotor, 
										const float& xGlobal, const float& yGlobal, const float& zGlobal, 
										float &rotorFractionCumulative, const bool &trackForce,
										const InfoStruct& InfoGlobal, RotorViewStruct &RotorView )
{
	const RotorInfoStruct& InfoRotor = RotorView.Info;
    const BoundsStruct& Bounds = InfoRotor.Bounds;
    const auto& rotorMapView = RotorView.rotorMapView;
    const auto& interpolationView = RotorView.interpolationView;
	
	float xRotor = xGlobal; float yRotor = yGlobal; float zRotor = zGlobal;
	projectXYZIntoRotorFrame( xRotor, yRotor, zRotor, InfoRotor, InfoGlobal );
	
    float rotorFraction = 0.f;

    if (xRotor < Bounds.xMin || xRotor >= Bounds.xMax ||
        yRotor < Bounds.yMin || yRotor >= Bounds.yMax ||
        zRotor < Bounds.zMin || zRotor >= Bounds.zMax) return;

    // Position relative to rotor Bounds.Min
    const float xRelative = xRotor - Bounds.xMin;
    const float yRelative = yRotor - Bounds.yMin;
    const float zRelative = zRotor - Bounds.zMin;

    const int iInterpolation = static_cast<int>(xRelative / InfoRotor.res);
    const int jInterpolation = static_cast<int>(yRelative / InfoRotor.res);
    const int kInterpolation = static_cast<int>(zRelative / InfoRotor.res);

    // Protect against rounding up at the upper boundary
    if (iInterpolation >= InfoRotor.cellCountX ||
        jInterpolation >= InfoRotor.cellCountY ||
        kInterpolation >= InfoRotor.cellCountZ) return;

    const int blockCountX = InfoRotor.cellCountX / 4;
    const int blockCountY = InfoRotor.cellCountY / 4;

    const int iBlock = iInterpolation / 4;
    const int jBlock = jInterpolation / 4;
    const int kBlock = kInterpolation / 4;

    const int blockIndex = kBlock * blockCountX * blockCountY + jBlock * blockCountX + iBlock;

    const int rotorMap = rotorMapView(blockIndex);
    if (rotorMap < 0) return;

    const int interpolationIndex = rotorMap * 64 + (kInterpolation % 4) * 16 + (jInterpolation % 4) * 4 + (iInterpolation % 4);

    const uint32_t packed = interpolationView(interpolationIndex);

    const float xWithinCube = (xRelative / InfoRotor.res) - iInterpolation;
    const float yWithinCube = (yRelative / InfoRotor.res) - jInterpolation;
    const float zWithinCube = (zRelative / InfoRotor.res) - kInterpolation;

    interpolateRotorCube( rotorFraction, xWithinCube, yWithinCube, zWithinCube, packed );
	if ( rotorFraction == 0.f ) return;
	
	// forcing is calculated in the global frame
	float gxRotor, gyRotor, gzRotor;
	getRotorForcing( gxRotor, gyRotor, gzRotor, xGlobal, yGlobal, zGlobal, rho, uxPreRotor, uyPreRotor, uzPreRotor, InfoRotor, InfoGlobal );
	
	if ( rotorFractionCumulative + rotorFraction > 1.f )
	{
		rotorFraction = 1.f - rotorFractionCumulative;
		rotorFractionCumulative = 1.f;
	}
	else rotorFractionCumulative += rotorFraction;
	
	gxRotor *= rotorFraction;
	gyRotor *= rotorFraction;
	gzRotor *= rotorFraction;

	BC.gx += gxRotor;
	BC.gy += gyRotor;
	BC.gz += gzRotor;
	
	// if trackForce is true, write rotor forcing
	if ( !TRACK_ROTOR_FORCE || !trackForce ) return;
	
	projectForcingIntoRotorFrame( gxRotor, gyRotor, gzRotor, InfoRotor, InfoGlobal );
	
	// Coordinates inside the selected block, in interpolation-cell units.
	const int iForce = TNL::max(0, TNL::min(2, static_cast<int>(0.75f * ((iInterpolation % 4) + xWithinCube))));
	const int jForce = TNL::max(0, TNL::min(2, static_cast<int>(0.75f * ((jInterpolation % 4) + yWithinCube))));
	const int kForce = TNL::max(0, TNL::min(2, static_cast<int>(0.75f * ((kInterpolation % 4) + zWithinCube))));

	const int forceIndex = rotorMap * 27 + kForce * 9 + jForce * 3 + iForce;
    
    RotorView.gxView( forceIndex ) += gxRotor;
    RotorView.gyView( forceIndex ) += gyRotor;
    RotorView.gzView( forceIndex ) += gzRotor;
}

__cuda_callable__ inline void bitPackInterpolationCube( uint32_t& result, const uint8_t (&counter)[8] )
{
    result = 0u;
    for (unsigned int corner = 0; corner < 8; corner++ )
    {
        result |= static_cast<uint32_t>(counter[corner]) << (4u * corner);
    }
}

void buildRotors( GridStruct &Grid, std::vector<STLStruct> &rotorSTLs )
{
	// builds rotors for this grid level
	const int rotorCount = rotorSTLs.size();
	Grid.rotors.resize( rotorCount );
	for ( int rotorID = 0; rotorID < rotorCount; rotorID++ )
	{
		STLStruct &STL = rotorSTLs[ rotorID ];
		RotorStruct &Rotor = Grid.rotors[ rotorID ];
		VoxelizerStruct Voxelizer;
		
		// 1) set bounds
		Voxelizer.Info.ox = STL.Bounds.xMin;
		Voxelizer.Info.oy = STL.Bounds.yMin;
		Voxelizer.Info.oz = STL.Bounds.zMin;
		Voxelizer.Info.res = Grid.Info.res * 0.5f;
		Voxelizer.Info.cellCountX = (int)(( STL.Bounds.xMax - STL.Bounds.xMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.Info.cellCountY = (int)(( STL.Bounds.yMax - STL.Bounds.yMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.Info.cellCountZ = (int)(( STL.Bounds.zMax - STL.Bounds.zMin ) / Voxelizer.Info.res) + 2;
		Voxelizer.rayMaps.resize( 1 );
		voxelizeSTL( Voxelizer.rayMaps[0], STL, Voxelizer );
		
		Rotor.Info.rotorID = rotorID;
		Rotor.Info.res = Voxelizer.Info.res;
		Rotor.Info.Bounds.xMin = Voxelizer.Info.ox - 1.5f * Voxelizer.Info.res;
		Rotor.Info.Bounds.yMin = Voxelizer.Info.oy - 1.5f * Voxelizer.Info.res;
		Rotor.Info.Bounds.zMin = Voxelizer.Info.oz - 1.5f * Voxelizer.Info.res;
		// add overlap of 2 cells and then +3 /4 *4 to get closest upper multiple of 4
		Rotor.Info.cellCountX = (( Voxelizer.Info.cellCountX + 2 + 3 ) / 4 ) * 4; // we want a multiple of 4 here
		Rotor.Info.cellCountY = (( Voxelizer.Info.cellCountY + 2 + 3 ) / 4 ) * 4;
		Rotor.Info.cellCountZ = (( Voxelizer.Info.cellCountZ + 2 + 3 ) / 4 ) * 4;
		Rotor.Info.Bounds.xMax = Rotor.Info.Bounds.xMin + Rotor.Info.res * (float)Rotor.Info.cellCountX;
		Rotor.Info.Bounds.yMax = Rotor.Info.Bounds.yMin + Rotor.Info.res * (float)Rotor.Info.cellCountY;
		Rotor.Info.Bounds.zMax = Rotor.Info.Bounds.zMin + Rotor.Info.res * (float)Rotor.Info.cellCountZ;
		
		const int blockCountX = Rotor.Info.cellCountX/4;
		const int blockCountY = Rotor.Info.cellCountY/4;
		const int blockCountZ = Rotor.Info.cellCountZ/4;
		const int blockCountXY = blockCountX * blockCountY;
		const int blockCount = blockCountX * blockCountY * blockCountZ;
		const int voxelizerCountX = Voxelizer.Info.cellCountX;
		const int voxelizerCountY = Voxelizer.Info.cellCountY;
		
		// 2) Mark rotorMap blocks, if they contain any nonzero value
		BoolArrayType rotorMapMarkerArray( blockCount );
		rotorMapMarkerArray.setValue( false );
		// Each element in rotorMap tracks 4x4x4 interpolation cells
		// At vertices of those cells, there are counter points that count from 0 to 8
		// The count says how many points around the counter point are marked as 1 by the voxelizer
		// 4x4x4 interpolation cells -> 5x5x5 counter points -> 6x6x6 voxelizer points
		// First, mark elements of the rotorMap as 1 if any of the voxelizer points is 1
		auto rayMapView = Voxelizer.rayMaps[0].rayMapArray.getConstView();
		auto hitCounterScanView = Voxelizer.rayMaps[0].hitCounterScanArray.getConstView();
		auto rotorMapMarkerView = rotorMapMarkerArray.getView();
		auto rotorMapMarkerLambda = [=] __cuda_callable__ ( const int block ) mutable
		{
			const int kBlock = block / blockCountXY;
			const int remainder = block % blockCountXY;
			const int jBlock = remainder / blockCountX;
			const int iBlock = remainder % blockCountX;
			
			const int iVoxelizerStart = TNL::max( 0, iBlock * 4 - 2 );
			const int iVoxelizerEnd = TNL::min( voxelizerCountX, iBlock * 4 + 4 );
			const int jVoxelizerStart = TNL::max( 0, jBlock * 4 - 2);
			const int jVoxelizerEnd = TNL::min( voxelizerCountY, jBlock * 4 + 4 );
			const int kVoxelizerStart = kBlock * 4 - 2;
			const int kVoxelizerEnd = kBlock * 4 + 4;
			
			// here iCell, jCell, kCell refers to the voxelizer cells
			for ( int jCell = jVoxelizerStart; jCell < jVoxelizerEnd; jCell++ )
			{
				for ( int iCell = iVoxelizerStart; iCell < iVoxelizerEnd; iCell++ )
				{
					for ( int kCell = kVoxelizerStart; kCell < kVoxelizerEnd; kCell++ )
					{
						int kStart, kEnd;
						const int rayIndex = voxelizerCountX * jCell + iCell;
						const long long startingPoint = hitCounterScanView( rayIndex );
						const long long endingPoint = hitCounterScanView( rayIndex + 1 );
						for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
						{
							kStart = rayMapView( startIndex );
							if ( kStart > kCell ) break;
							kEnd = rayMapView( startIndex + 1LL );
							if ( kEnd > kCell )
							{
								rotorMapMarkerView( block ) = true;
								return;
							}
						}
					}
				}
			}			
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, blockCount, rotorMapMarkerLambda );
		
		// 3) Fill the rotorMap
		// -1 = all underlying interpolation cells are fully zero
		// >= 0 -> compact index of this non zero block 
		// also fill the indexArray
		const int nonZeroBlockCount = TNL::sum( rotorMapMarkerArray );
		Rotor.indexArray.setSize( nonZeroBlockCount );
		IntArrayType scanArray( blockCount );
		intArrayFromBoolArray( scanArray, rotorMapMarkerArray );
		TNL::Algorithms::inplaceExclusiveScan( scanArray, 0, blockCount, TNL::Plus{} );
		Rotor.rotorMapArray.setSize( blockCount );
		auto rotorMapView = Rotor.rotorMapArray.getView();
		auto scanView = scanArray.getConstView();
		auto indexView = Rotor.indexArray.getView();
		
		auto scanLambda = [=] __cuda_callable__ ( const int block ) mutable
		{
			if ( rotorMapMarkerView( block ) ) 
			{
				const int scan = scanView( block );
				rotorMapView( block ) = scan;
				indexView( scan ) = block;
			}
			else rotorMapView( block ) = -1;
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, blockCount, scanLambda );
		
		// 4) Build the interpolationArray
		Rotor.interpolationArray.setSize( nonZeroBlockCount * 64 );
		auto interpolationView = Rotor.interpolationArray.getView();
		// Loop over the non zero blocks captured in the indexArray
		// Inside the block, loop over the 6x6x6 voxelizer cells -> 216 bools
		// Then loop over the 5x5x5 counter points -> 125 uint8_t
		// Finally loop over the 4x4x4 interpolation cells
		// Bit pack the 8 relevant counter points into a single uint32_t
		// Write this uint as result
		auto resultLambda = [=] __cuda_callable__ ( const int index ) mutable
		{
			const int block = indexView( index );
			
			const int kBlock = block / blockCountXY;
			const int remainder = block % blockCountXY;
			const int jBlock = remainder / blockCountX;
			const int iBlock = remainder % blockCountX;
			
			const int iVoxelizerStart = TNL::max( 0, iBlock * 4 - 2 );
			const int iVoxelizerEnd = TNL::min( voxelizerCountX, iBlock * 4 + 4 );
			const int jVoxelizerStart = TNL::max( 0, jBlock * 4 - 2);
			const int jVoxelizerEnd = TNL::min( voxelizerCountY, jBlock * 4 + 4 );
			const int kVoxelizerStart = kBlock * 4 - 2;
			const int kVoxelizerEnd = kBlock * 4 + 4;
			
			bool voxelizerMark[216] = { false };
			
			// here iCell, jCell, kCell refers to the voxelizer cells
			for ( int jCell = jVoxelizerStart; jCell < jVoxelizerEnd; jCell++ )
			{
				for ( int iCell = iVoxelizerStart; iCell < iVoxelizerEnd; iCell++ )
				{
					for ( int kCell = kVoxelizerStart; kCell < kVoxelizerEnd; kCell++ )
					{
						int kStart, kEnd;
						const int rayIndex = voxelizerCountX * jCell + iCell;
						const long long startingPoint = hitCounterScanView( rayIndex );
						const long long endingPoint = hitCounterScanView( rayIndex + 1 );
						for ( long long startIndex = startingPoint; startIndex < endingPoint; startIndex = startIndex + 2LL )
						{
							kStart = rayMapView( startIndex );
							if ( kStart > kCell ) break;
							kEnd = rayMapView( startIndex + 1LL );
							if ( kEnd > kCell )
							{
								const int iLocal = iCell - (iBlock * 4 - 2);
								const int jLocal = jCell - (jBlock * 4 - 2);
								const int kLocal = kCell - (kBlock * 4 - 2);
								const int markIndex = kLocal * 36 + jLocal * 6 + iLocal;
								voxelizerMark[ markIndex ] = true;
							}
						}
					}
				}
			}		
			
			uint8_t counterPoint[125] = { 0u };
			for ( int kCounter = 0; kCounter < 5; kCounter++ )
			{
				for ( int jCounter = 0; jCounter < 5; jCounter++ )
				{
					for ( int iCounter = 0; iCounter < 5; iCounter++ )
					{
						int counter = 0;
						for ( int kMark = kCounter; kMark <= kCounter+1; kMark++ )
						{
							for ( int jMark = jCounter; jMark <= jCounter+1; jMark++ )
							{
								for ( int iMark = iCounter; iMark <= iCounter+1; iMark++ )
								{
									const int markIndex = kMark * 36 + jMark * 6 + iMark;
									if ( voxelizerMark[ markIndex ] ) counter++;
								}
							}
						}
						const int counterIndex = kCounter * 25 + jCounter * 5 + iCounter;
						counterPoint[ counterIndex ] = counter;
					}
				}
			}
			
			for ( int kInterpolation = 0; kInterpolation < 4; kInterpolation++ )
			{
				for ( int jInterpolation = 0; jInterpolation < 4; jInterpolation++ )
				{
					for ( int iInterpolation = 0; iInterpolation < 4; iInterpolation++ )
					{
						uint8_t smallCounter[8];
						int smallIndex = 0;
						for ( int kCounter = kInterpolation; kCounter <= kInterpolation+1; kCounter++ )
						{
							for ( int jCounter = jInterpolation; jCounter <= jInterpolation+1; jCounter++ )
							{
								for ( int iCounter = iInterpolation; iCounter <= iInterpolation+1; iCounter++ )
								{
									const int counterIndex = kCounter * 25 + jCounter * 5 + iCounter;
									smallCounter[ smallIndex ] = counterPoint[ counterIndex ];
									smallIndex++;
								}
							}
						}
						uint32_t result = 0u;
						bitPackInterpolationCube( result, smallCounter );
						const int interpolationIndex = index * 64 + kInterpolation * 16 + jInterpolation * 4 + iInterpolation;
						interpolationView( interpolationIndex ) = result;
					}
				}
			}
			
		};
		TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, nonZeroBlockCount, resultLambda );
		
		Rotor.gxArray.setSize( nonZeroBlockCount * 27 );
		Rotor.gyArray.setSize( nonZeroBlockCount * 27 );
		Rotor.gzArray.setSize( nonZeroBlockCount * 27 );
		Rotor.gxArray.setValue( 0.f );
		Rotor.gyArray.setValue( 0.f );
		Rotor.gzArray.setValue( 0.f );
	}
	
	long long memoryBytes = 0LL;
	for ( int rotorID = 0; rotorID < (int)Grid.rotors.size(); rotorID++ )
	{
		memoryBytes += 1LL * (long long)Grid.rotors[ rotorID ].rotorMapArray.getSize() * 4LL; // rotorMap
		memoryBytes += 65LL * (long long)Grid.rotors[ rotorID ].indexArray.getSize() * 4LL; // indexArray, interpolationArray
		memoryBytes += 81LL * (long long)Grid.rotors[ rotorID ].indexArray.getSize() * 4LL; // rotor force tracker
	}
	
	// prepare rotor views
	Grid.rotorViews.resize(Grid.rotors.size());
	auto& rotorViews = Grid.rotorViews;
	for (int rotorID = 0; rotorID < rotorCount; rotorID++)
	{
		auto& Rotor = Grid.rotors[rotorID];
		auto& RotorView = rotorViews[rotorID];
		RotorView.Info = Rotor.Info;
		RotorView.indexView.bind(Rotor.indexArray.getConstView());
		RotorView.rotorMapView.bind(Rotor.rotorMapArray.getConstView());
		RotorView.interpolationView.bind(Rotor.interpolationArray.getConstView());
		RotorView.gxView.bind(Rotor.gxArray.getView());
		RotorView.gyView.bind(Rotor.gyArray.getView());
		RotorView.gzView.bind(Rotor.gzArray.getView());
	}

	std::cout << "	Allocated rotors for grid level " << Grid.Info.gridID << ", they take " << memoryBytes / 1048576.0 << " MiB" << std::endl;
}
