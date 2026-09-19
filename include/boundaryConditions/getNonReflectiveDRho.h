#pragma once

#include "../D3Q27Directions.h"

// id: 		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: 		{ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: 		{ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: 		{ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// Based on Schlaffer disertation 2013, eq (6.30)
/*
// non well conditioned version
__host__ __device__ float getNonReflectiveDRho( const float &dRhoZ, const float &dRhoPrev, const float &uNormalPrev )
{
	const float cs = INVSQRT3;
	const float cs2 = 1.f / 3.f;
	const float bracket = uNormalPrev + cs + 1.f;
	const float rhoZ = dRhoZ + 1.f;
	const float rhoPrev = dRhoPrev + 1.f;
	const float root = sqrtf( rhoPrev * rhoPrev * cs2 + 2.f * rhoPrev * rhoZ * bracket - rhoZ * rhoZ );
	const float numerator = rhoPrev * cs2 + rhoZ * bracket + cs * root;
	const float denominator = bracket * bracket + cs2;
	return ( numerator / denominator ) - 1.f;
}
*/

// well conditioned version
__host__ __device__ float getNonReflectiveDRho( const float &dRhoZ, const float &dRhoPrev, const float &uNormalPrev )
{
    const float cs  = INVSQRT3;
    const float cs2 = cs * cs;

    const float bracket = uNormalPrev + cs + 1.f;
    const float t = uNormalPrev - dRhoZ;

    const float D = fmaf(bracket, bracket, cs2);
    const float B = fmaf(bracket, cs + t, -cs2 * dRhoPrev);
    const float H = fmaf(-t, 2.f * cs + t, 2.f * cs2 * dRhoPrev);

    const float discriminant = fmaf(D, H, B * B);
    return H / (B + sqrtf(discriminant));
}
