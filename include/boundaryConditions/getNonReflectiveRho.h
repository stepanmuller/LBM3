#pragma once

#include "./D3Q27Directions.h"

// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// Based on Schlaffer disertation 2013, eq (6.30)

__host__ __device__ float getNonReflectiveRho( const float &rhoZ, const float &rhoPrev, const float &uNormalPrev )
{
	
	const float cs = INVSQRT3;
	const float cs2 = 1.f / 3.f;
	const float bracket = uNormalPrev + cs + 1.f;
	const float root = sqrtf( rhoPrev * rhoPrev * cs2 + 2.f * rhoPrev * rhoZ * bracket - rhoZ * rhoZ );
	const float numerator = rhoPrev * cs2 + rhoZ * bracket + cs * root;
	const float denominator = bracket * bracket + cs2;
	return ( numerator / denominator );
}
