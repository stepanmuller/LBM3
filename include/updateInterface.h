#pragma once

#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"

// Helper table for second order moments
//  	id: { 0, 1, 2, 3, 4, 5, 6,		 7, 8, 9,10,11,12,13,14,15,16,17,18,		19,20,21,22,23,24,25,26 };
	
//  	cx: { 0, 1,-1, 0, 0, 0, 0,		 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,		-1, 1,-1, 1, 1,-1,-1, 1 };
//  	cy: { 0, 0, 0, 0, 0,-1, 1,		 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1,		 1,-1,-1, 1,-1, 1,-1, 1 };
//  	cz: { 0, 0, 0,-1, 1, 0, 0,		-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,		-1, 1, 1,-1,-1, 1,-1, 1 };

// cx * cx: { 0, 1, 1, 0, 0, 0, 0,		 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,		 1, 1, 1, 1, 1, 1, 1, 1 };
// cy * cy: { 0, 0, 0, 0, 0, 1, 1,		 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,		 1, 1, 1, 1, 1, 1, 1, 1 };
// cz * cz: { 0, 0, 0, 1, 1, 0, 0,		 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,		 1, 1, 1, 1, 1, 1, 1, 1 };

// cy * cz: { 0, 0, 0, 0, 0, 0, 0,		 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 1,		-1,-1,-1,-1, 1, 1, 1, 1 };
// cx * cz: { 0, 0, 0, 0, 0, 0, 0,		-1,-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,		 1, 1,-1,-1,-1,-1, 1, 1 };
// cx * cy: { 0, 0, 0, 0, 0, 0, 0,		 0, 0, 0, 0, 1, 1, 0, 0,-1,-1, 0, 0,		-1,-1, 1, 1,-1,-1, 1, 1 };

// cx2-cy2: { 0, 1, 1, 0, 0,-1,-1,		 1, 1, 1, 1, 0, 0,-1,-1, 0, 0,-1,-1,		 0, 0, 0, 0, 0, 0, 0, 0 };
// cx2-cz2: { 0, 1, 1,-1,-1, 0, 0,		 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1,		 0, 0, 0, 0, 0, 0, 0, 0 };

__host__ __device__ void reconstructInterpolatedF( 	float (&f)[27], const float &rho, const float &ux, const float &uy, const float &uz, 
													const float &k_011, const float &k_101, const float &k_110, 
													const float &k_200, const float &k_020, const float &k_002 )
{
	const float dRho = rho - 1.f;
	const float ux2 = ux * ux;
	const float uy2 = uy * uy;
	const float uz2 = uz * uz;
	
	// D3Q27 weight moments needed by the well-conditioned transformation.
    const float K_aa0 = 1.f / 36.f;
    const float K_ab0 = 1.f / 9.f;
    const float K_ac0 = 1.f / 36.f;
    const float K_ba0 = 1.f / 9.f;
    const float K_bb0 = 4.f / 9.f;
    const float K_bc0 = 1.f / 9.f;
    const float K_ca0 = 1.f / 36.f;
    const float K_cb0 = 1.f / 9.f;
    const float K_cc0 = 1.f / 36.f;
    const float K_a00 = 1.f / 6.f;
    const float K_b00 = 2.f / 3.f;
    const float K_c00 = 1.f / 6.f;
    const float K_a02 = 1.f / 18.f;
    const float K_b02 = 2.f / 9.f;
    const float K_c02 = 1.f / 18.f;

    // -------------------------------------------------------------------------
    // Cumulants -> well-conditioned central moments, Geier 2017, Eqs. 53-56.
    // -------------------------------------------------------------------------

	const float k_000 = dRho;

    const float k_220 = dRho / 9.f;
    const float k_022 = k_220;
    const float k_202 = k_220;

    const float k_222 = dRho / 27.f;

    // -------------------------------------------------------------------------
    // Well-conditioned central moments -> shifted populations, Eqs. 57-65.
    // -------------------------------------------------------------------------

    const float k_b00 = k_000 * (1.f - ux2) - k_200 - ux2;
    const float k_b01 = - 2.f * ux * k_101;
    const float k_b02 = k_002 * (1.f - ux2) - k_202 - ux2 / 3.f;
    const float k_b10 = - 2.f * ux * k_110;
    const float k_b11 = k_011 * (1.f - ux2);
    const float k_b20 = k_020 * (1.f - ux2) - k_220 - ux2 / 3.f;
    const float k_b22 = k_022 * (1.f - ux2) - k_222 - ux2 / 9.f;

    const float k_a00 = ((k_000 + 1.f) * (ux2 - ux) + k_200) * 0.5f;
    const float k_a01 = (k_101 * (2.f * ux - 1.f)) * 0.5f;
    const float k_a02 = ((k_002 + 1.f / 3.f) * (ux2 - ux) + k_202) * 0.5f;
    const float k_a10 = (k_110 * (2.f * ux - 1.f)) * 0.5f;
    const float k_a11 = (k_011 * (ux2 - ux)) * 0.5f;
    const float k_a20 = ((k_020 + 1.f / 3.f) * (ux2 - ux) + k_220) * 0.5f;
    const float k_a22 = ((k_022 + 1.f / 9.f) * (ux2 - ux) + k_222) * 0.5f;

    const float k_c00 = ((k_000 + 1.f) * (ux2 + ux) + k_200) * 0.5f;
    const float k_c01 = (k_101 * (2.f * ux + 1.f)) * 0.5f;
    const float k_c02 = ((k_002 + 1.f / 3.f) * (ux2 + ux) + k_202) * 0.5f;
    const float k_c10 = (k_110 * (2.f * ux + 1.f)) * 0.5f;
    const float k_c11 = (k_011 * (ux2 + ux)) * 0.5f;
    const float k_c20 = ((k_020 + 1.f / 3.f) * (ux2 + ux) + k_220) * 0.5f;
    const float k_c22 = ((k_022 + 1.f / 9.f) * (ux2 + ux) + k_222) * 0.5f;

    const float k_ab0 = k_a00 * (1.f - uy2) - 2.f * uy * k_a10 - k_a20 - K_a00 * uy2;
    const float k_ab1 = k_a01 * (1.f - uy2) - 2.f * uy * k_a11 - 0.f;
    const float k_ab2 = k_a02 * (1.f - uy2) - 2.f * uy * 0.f - k_a22 - K_a02 * uy2;
    const float k_bb0 = k_b00 * (1.f - uy2) - 2.f * uy * k_b10 - k_b20 - K_b00 * uy2;
    const float k_bb1 = k_b01 * (1.f - uy2) - 2.f * uy * k_b11 - 0.f;
    const float k_bb2 = k_b02 * (1.f - uy2) - 2.f * uy * 0.f - k_b22 - K_b02 * uy2;
    const float k_cb0 = k_c00 * (1.f - uy2) - 2.f * uy * k_c10 - k_c20 - K_c00 * uy2;
    const float k_cb1 = k_c01 * (1.f - uy2) - 2.f * uy * k_c11 - 0.f;
    const float k_cb2 = k_c02 * (1.f - uy2) - 2.f * uy * 0.f - k_c22 - K_c02 * uy2;

    const float k_aa0 = ((k_a00 + K_a00) * (uy2 - uy) + k_a10 * (2.f * uy - 1.f) + k_a20) * 0.5f;
    const float k_aa1 = (k_a01 * (uy2 - uy) + k_a11 * (2.f * uy - 1.f) + 0.f) * 0.5f;
    const float k_aa2 = ((k_a02 + K_a02) * (uy2 - uy) + k_a22) * 0.5f;
    const float k_ba0 = ((k_b00 + K_b00) * (uy2 - uy) + k_b10 * (2.f * uy - 1.f) + k_b20) * 0.5f;
    const float k_ba1 = (k_b01 * (uy2 - uy) + k_b11 * (2.f * uy - 1.f)) * 0.5f;
    const float k_ba2 = ((k_b02 + K_b02) * (uy2 - uy) + k_b22) * 0.5f;
    const float k_ca0 = ((k_c00 + K_c00) * (uy2 - uy) + k_c10 * (2.f * uy - 1.f) + k_c20) * 0.5f;
    const float k_ca1 = (k_c01 * (uy2 - uy) + k_c11 * (2.f * uy - 1.f)) * 0.5f;
    const float k_ca2 = ((k_c02 + K_c02) * (uy2 - uy) + k_c22) * 0.5f;

    const float k_ac0 = ((k_a00 + K_a00) * (uy2 + uy) + k_a10 * (2.f * uy + 1.f) + k_a20) * 0.5f;
    const float k_ac1 = (k_a01 * (uy2 + uy) + k_a11 * (2.f * uy + 1.f)) * 0.5f;
    const float k_ac2 = ((k_a02 + K_a02) * (uy2 + uy) + k_a22) * 0.5f;
    const float k_bc0 = ((k_b00 + K_b00) * (uy2 + uy) + k_b10 * (2.f * uy + 1.f) + k_b20) * 0.5f;
    const float k_bc1 = (k_b01 * (uy2 + uy) + k_b11 * (2.f * uy + 1.f)) * 0.5f;
    const float k_bc2 = ((k_b02 + K_b02) * (uy2 + uy) + k_b22) * 0.5f;
    const float k_cc0 = ((k_c00 + K_c00) * (uy2 + uy) + k_c10 * (2.f * uy + 1.f) + k_c20) * 0.5f;
    const float k_cc1 = (k_c01 * (uy2 + uy) + k_c11 * (2.f * uy + 1.f)) * 0.5f;
    const float k_cc2 = ((k_c02 + K_c02) * (uy2 + uy) + k_c22) * 0.5f;

    f[MMO] = k_aa0 * (1.f - uz2) - 2.f * uz * k_aa1 - k_aa2 - K_aa0 * uz2;
    f[MOO]  = k_ab0 * (1.f - uz2) - 2.f * uz * k_ab1 - k_ab2 - K_ab0 * uz2;
    f[MPO] = k_ac0 * (1.f - uz2) - 2.f * uz * k_ac1 - k_ac2 - K_ac0 * uz2;
    f[OMO]  = k_ba0 * (1.f - uz2) - 2.f * uz * k_ba1 - k_ba2 - K_ba0 * uz2;
    f[OOO]  = k_bb0 * (1.f - uz2) - 2.f * uz * k_bb1 - k_bb2 - K_bb0 * uz2;
    f[OPO]  = k_bc0 * (1.f - uz2) - 2.f * uz * k_bc1 - k_bc2 - K_bc0 * uz2;
    f[PMO] = k_ca0 * (1.f - uz2) - 2.f * uz * k_ca1 - k_ca2 - K_ca0 * uz2;
    f[POO]  = k_cb0 * (1.f - uz2) - 2.f * uz * k_cb1 - k_cb2 - K_cb0 * uz2;
    f[PPO] = k_cc0 * (1.f - uz2) - 2.f * uz * k_cc1 - k_cc2 - K_cc0 * uz2;

    f[MMM] = ((k_aa0 + K_aa0) * (uz2 - uz) + k_aa1 * (2.f * uz - 1.f) + k_aa2) * 0.5f;
    f[MOM] = ((k_ab0 + K_ab0) * (uz2 - uz) + k_ab1 * (2.f * uz - 1.f) + k_ab2) * 0.5f;
    f[MPM] = ((k_ac0 + K_ac0) * (uz2 - uz) + k_ac1 * (2.f * uz - 1.f) + k_ac2) * 0.5f;
    f[OMM] = ((k_ba0 + K_ba0) * (uz2 - uz) + k_ba1 * (2.f * uz - 1.f) + k_ba2) * 0.5f;
    f[OOM]  = ((k_bb0 + K_bb0) * (uz2 - uz) + k_bb1 * (2.f * uz - 1.f) + k_bb2) * 0.5f;
    f[OPM] = ((k_bc0 + K_bc0) * (uz2 - uz) + k_bc1 * (2.f * uz - 1.f) + k_bc2) * 0.5f;
    f[PMM] = ((k_ca0 + K_ca0) * (uz2 - uz) + k_ca1 * (2.f * uz - 1.f) + k_ca2) * 0.5f;
    f[POM]  = ((k_cb0 + K_cb0) * (uz2 - uz) + k_cb1 * (2.f * uz - 1.f) + k_cb2) * 0.5f;
    f[PPM] = ((k_cc0 + K_cc0) * (uz2 - uz) + k_cc1 * (2.f * uz - 1.f) + k_cc2) * 0.5f;

    f[MMP] = ((k_aa0 + K_aa0) * (uz2 + uz) + k_aa1 * (2.f * uz + 1.f) + k_aa2) * 0.5f;
    f[MOP]  = ((k_ab0 + K_ab0) * (uz2 + uz) + k_ab1 * (2.f * uz + 1.f) + k_ab2) * 0.5f;
    f[MPP] = ((k_ac0 + K_ac0) * (uz2 + uz) + k_ac1 * (2.f * uz + 1.f) + k_ac2) * 0.5f;
    f[OMP] = ((k_ba0 + K_ba0) * (uz2 + uz) + k_ba1 * (2.f * uz + 1.f) + k_ba2) * 0.5f;
    f[OOP]  = ((k_bb0 + K_bb0) * (uz2 + uz) + k_bb1 * (2.f * uz + 1.f) + k_bb2) * 0.5f;
    f[OPP] = ((k_bc0 + K_bc0) * (uz2 + uz) + k_bc1 * (2.f * uz + 1.f) + k_bc2) * 0.5f;
    f[PMP] = ((k_ca0 + K_ca0) * (uz2 + uz) + k_ca1 * (2.f * uz + 1.f) + k_ca2) * 0.5f;
    f[POP]  = ((k_cb0 + K_cb0) * (uz2 + uz) + k_cb1 * (2.f * uz + 1.f) + k_cb2) * 0.5f;
    f[PPP] = ((k_cc0 + K_cc0) * (uz2 + uz) + k_cc1 * (2.f * uz + 1.f) + k_cc2) * 0.5f;
}

void updateFineToCoarseInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	// The interpolation and rescaling is based on Martin Schönherr's disertation 2015
	const InfoStruct &InfoCoarse = GridCoarse.Info;
	auto fViewCoarse = GridCoarse.fArray.getView();
	const bool &esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto shifterViewCoarse = GridCoarse.IJKNBR.shifterArray.getConstView();
	auto jPlusViewCoarse = GridCoarse.IJKNBR.jPlusArray.getConstView();
	auto kPlusViewCoarse = GridCoarse.IJKNBR.kPlusArray.getConstView();
	auto jkPlusViewCoarse = GridCoarse.IJKNBR.jkPlusArray.getConstView();
	const float tauCoarse = 3.f * InfoCoarse.nu + 0.5f;
	const float omega1Coarse =  1.f / tauCoarse;
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto shifterViewFine = GridFine.IJKNBR.shifterArray.getConstView();
	auto jPlusViewFine = GridFine.IJKNBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.IJKNBR.kPlusArray.getConstView();
	auto jkPlusViewFine = GridFine.IJKNBR.jkPlusArray.getConstView();
	const float tauFine = 3.f * InfoFine.nu + 0.5f;
	const float omega1Fine =  1.f / tauFine;
	
	auto indexView = GridCoarse.FineToCoarseInterface.indexArray.getConstView();
	auto childMapView = GridCoarse.FineToCoarseInterface.childMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cellCoarse = indexView( index );
		const int cellFine0 = childMapView( index );
		
		NBRStruct NBRStencil;
		getCompressedNBR( cellFine0, NBRStencil, shifterViewFine, jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );
				
		int cellStencil[8];
		cellStencil[0] = NBRStencil.self;
		cellStencil[1] = NBRStencil.iPlus;
		cellStencil[2] = NBRStencil.jPlus;
		cellStencil[3] = NBRStencil.ijPlus;
		cellStencil[4] = NBRStencil.kPlus;
		cellStencil[5] = NBRStencil.ikPlus;
		cellStencil[6] = NBRStencil.jkPlus;
		cellStencil[7] = NBRStencil.ijkPlus;
		
		// Initialize stencil variables
		float dRhoStencil[8]; float uxStencil[8]; float uyStencil[8]; float uzStencil[8];
		float kxyStencil[8]; float kyzStencil[8]; float kxzStencil[8]; float kxxMyyStencil[8]; float kxxMzzStencil[8];
		
		// Extract values from each stencil cell
		for ( int i = 0; i < 8; i++ )
		{
			const int nbr = cellStencil[i];
			NBRStruct NBRofNBR;
			getCompressedNBR( nbr, NBRofNBR, shifterViewFine, jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPreCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperFine );
			float fNbr[27];
			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewFine( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			
			getDRhoUxUyUz( dRhoStencil[i], uxStencil[i], uyStencil[i], uzStencil[i], fNbr );
			const float rho = 1.f + dRhoStencil[i];
			
			kxyStencil[i] = - 3.f * omega1Fine * ( ( 
					+ fNbr[11] + fNbr[12] - fNbr[15] - fNbr[16] 
					- fNbr[19] - fNbr[20] + fNbr[21] + fNbr[22] - fNbr[23] - fNbr[24] + fNbr[25] + fNbr[26]
													) / rho - uxStencil[i] * uyStencil[i] );
			kyzStencil[i] = - 3.f * omega1Fine * ( (
					- fNbr[13] - fNbr[14] + fNbr[17] + fNbr[18] 
					- fNbr[19] - fNbr[20] - fNbr[21] - fNbr[22] + fNbr[23] + fNbr[24] + fNbr[25] + fNbr[26]
													) / rho - uyStencil[i] * uzStencil[i] );
			kxzStencil[i] = - 3.f * omega1Fine * ( (
					- fNbr[7 ] - fNbr[8 ] + fNbr[9 ] + fNbr[10] 
					+ fNbr[19] + fNbr[20] - fNbr[21] - fNbr[22] - fNbr[23] - fNbr[24] + fNbr[25] + fNbr[26]
													) / rho - uxStencil[i] * uzStencil[i] );
			kxxMyyStencil[i] = - 1.5f * omega1Fine * ( (
					+ fNbr[1 ] + fNbr[2 ] - fNbr[5 ] - fNbr[6 ] 
					+ fNbr[7 ] + fNbr[8 ] + fNbr[9 ] + fNbr[10] - fNbr[13] - fNbr[14] - fNbr[17] - fNbr[18]
													) / rho - ( uxStencil[i] * uxStencil[i] - uyStencil[i] * uyStencil[i] ) );
			kxxMzzStencil[i] = - 1.5f * omega1Fine * ( (
					+ fNbr[1 ] + fNbr[2 ] - fNbr[3 ] - fNbr[4 ] 
					+ fNbr[11] + fNbr[12] - fNbr[13] - fNbr[14] + fNbr[15] + fNbr[16] - fNbr[17] - fNbr[18]
													) / rho - ( uxStencil[i] * uxStencil[i] - uzStencil[i] * uzStencil[i] ) );
		}
		
		// get all required coefficients
		// eq Schönherr 2015 (7.10)
		float d000 = 0.f; for ( int i = 0; i < 8; i++ ) d000 += dRhoStencil[i]; d000 *= 0.125f;
		
		// The following is directly taken from VirtualFluids (just renamed variables). https://github.com/irmb/virtualfluids 
		const float a000 = 0.015625f * (2.f * (((kxyStencil[0] - kxyStencil[7]) + (kxyStencil[4] - kxyStencil[3])) +
                                ((kxyStencil[1] - kxyStencil[6]) + (kxyStencil[5] - kxyStencil[2])) +
                                ((kxzStencil[0] - kxzStencil[7]) + (kxzStencil[3] - kxzStencil[4])) +
                                ((kxzStencil[1] - kxzStencil[6]) + (kxzStencil[2] - kxzStencil[5])) +
                                ((uyStencil[7] + uyStencil[0]) + (uyStencil[3] + uyStencil[4])) - ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[5] + uzStencil[2]) - (uzStencil[6] + uzStencil[1]))) +
                        8.f * (((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) + ((uxStencil[6] + uxStencil[1]) + (uxStencil[5] + uxStencil[2]))) +
                        ((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                        ((kxxMyyStencil[6] - kxxMyyStencil[1]) + (kxxMyyStencil[2] - kxxMyyStencil[5])) +
                        ((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                        ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[2] - kxxMzzStencil[5])));
        const float b000 = 0.015625f * (2.f * (((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                                ((kxxMyyStencil[6] - kxxMyyStencil[1]) + (kxxMyyStencil[2] - kxxMyyStencil[5])) +
                                ((kxyStencil[0] - kxyStencil[7]) + (kxyStencil[4] - kxyStencil[3])) +
                                ((kxyStencil[6] - kxyStencil[1]) + (kxyStencil[2] - kxyStencil[5])) +
                                ((kyzStencil[0] - kyzStencil[7]) + (kyzStencil[3] - kyzStencil[4])) +
                                ((kyzStencil[1] - kyzStencil[6]) + (kyzStencil[2] - kyzStencil[5])) +
                                ((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) - ((uxStencil[2] + uxStencil[6]) + (uxStencil[1] + uxStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[2] + uzStencil[5]))) +
                        8.f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[3] + uyStencil[4])) + ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5]))) +
                        ((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                        ((kxxMzzStencil[1] - kxxMzzStencil[6]) + (kxxMzzStencil[5] - kxxMzzStencil[2])));
        const float c000 = 0.015625f * (2.f * (((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                                ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[5] - kxxMzzStencil[2])) +
                                ((kxzStencil[0] - kxzStencil[7]) + (kxzStencil[4] - kxzStencil[3])) +
                                ((kxzStencil[6] - kxzStencil[1]) + (kxzStencil[2] - kxzStencil[5])) +
                                ((kyzStencil[0] - kyzStencil[7]) + (kyzStencil[4] - kyzStencil[3])) +
                                ((kyzStencil[1] - kyzStencil[6]) + (kyzStencil[5] - kyzStencil[2])) +
                                ((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[6] + uxStencil[1])) +
                                ((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[6] + uyStencil[1]) - (uyStencil[2] + uyStencil[5]))) +
                        8.f * (((uzStencil[7] + uzStencil[0]) + (uzStencil[3] + uzStencil[4])) + ((uzStencil[1] + uzStencil[6]) + (uzStencil[5] + uzStencil[2]))) +
                        ((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                        ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[2] - kxxMyyStencil[5])));

        const float ax = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[3] - uxStencil[4])) + ((uxStencil[1] - uxStencil[6]) + (uxStencil[5] - uxStencil[2])));
        const float bx = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[3] - uyStencil[4])) + ((uyStencil[1] - uyStencil[6]) + (uyStencil[5] - uyStencil[2])));
        const float cx = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[3] - uzStencil[4])) + ((uzStencil[1] - uzStencil[6]) + (uzStencil[5] - uzStencil[2])));

        const float ay = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[3] - uxStencil[4])) + ((uxStencil[6] - uxStencil[1]) + (uxStencil[2] - uxStencil[5])));
        const float by = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[3] - uyStencil[4])) + ((uyStencil[6] - uyStencil[1]) + (uyStencil[2] - uyStencil[5])));
        const float cy = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[3] - uzStencil[4])) + ((uzStencil[6] - uzStencil[1]) + (uzStencil[2] - uzStencil[5])));

        const float az = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[4] - uxStencil[3])) + ((uxStencil[6] - uxStencil[1]) + (uxStencil[5] - uxStencil[2])));
        const float bz = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[4] - uyStencil[3])) + ((uyStencil[6] - uyStencil[1]) + (uyStencil[5] - uyStencil[2])));
        const float cz = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[4] - uzStencil[3])) + ((uzStencil[6] - uzStencil[1]) + (uzStencil[5] - uzStencil[2])));
		
		// The rest of the coefficients is not needed for fineToCoarse interpolation, because the coarse cell has coords [0, 0, 0]
		/*
		a200 = 0.0625f * (2.f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[3] - uyStencil[6])) + ((uyStencil[4] - uyStencil[1]) - (uyStencil[2] + uyStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[6])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[4] + uzStencil[1]))) +
                        ((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                        ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[5] - kxxMyyStencil[2])) +
                        ((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                        ((kxxMzzStencil[1] - kxxMzzStencil[6]) + (kxxMzzStencil[5] - kxxMzzStencil[2])));
        b200 = 0.125f * (2.f * (-((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) + ((uxStencil[6] + uxStencil[1]) + (uxStencil[2] + uxStencil[5]))) +
                       ((kxyStencil[7] - kxyStencil[0]) + (kxyStencil[3] - kxyStencil[4])) +
                       ((kxyStencil[1] - kxyStencil[6]) + (kxyStencil[5] - kxyStencil[2])));
        c200 = 0.125f * (2.f * (((uxStencil[3] + uxStencil[4]) - (uxStencil[7] + uxStencil[0])) + ((uxStencil[6] + uxStencil[1]) - (uxStencil[2] + uxStencil[5]))) +
                       ((kxzStencil[7] - kxzStencil[0]) + (kxzStencil[3] - kxzStencil[4])) +
                       ((kxzStencil[1] - kxzStencil[6]) + (kxzStencil[5] - kxzStencil[2])));
        
        a020 = 0.125f * (2.f * (-((uyStencil[7] + uyStencil[0]) + (uyStencil[4] + uyStencil[3])) + ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5]))) +
                       ((kxyStencil[7] - kxyStencil[0]) + (kxyStencil[3] - kxyStencil[4])) +
                       ((kxyStencil[6] - kxyStencil[1]) + (kxyStencil[2] - kxyStencil[5])));
        b020 = 0.0625f * (2.f * (((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                                ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[5] - kxxMyyStencil[2])) +
                                ((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) - ((uxStencil[6] + uxStencil[1]) + (uxStencil[5] + uxStencil[2])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[2] + uzStencil[5]))) +
                        ((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                        ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[2] - kxxMzzStencil[5])));
        c020 = 0.125f * (2.f * (((uyStencil[4] + uyStencil[3]) - (uyStencil[7] + uyStencil[0])) + ((uyStencil[5] + uyStencil[2]) - (uyStencil[6] + uyStencil[1]))) +
                       ((kyzStencil[7] - kyzStencil[0]) + (kyzStencil[3] - kyzStencil[4])) +
                       ((kyzStencil[6] - kyzStencil[1]) + (kyzStencil[2] - kyzStencil[5])));
                 
        a002 = 0.125f * (2.f * (((uzStencil[3] + uzStencil[4]) - (uzStencil[7] + uzStencil[0])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[5] + uzStencil[2]))) +
                       ((kxzStencil[7] - kxzStencil[0]) + (kxzStencil[4] - kxzStencil[3])) +
                       ((kxzStencil[5] - kxzStencil[2]) + (kxzStencil[6] - kxzStencil[1])));
        b002 = 0.125f * (2.f * (((uzStencil[3] + uzStencil[4]) - (uzStencil[7] + uzStencil[0])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[1] + uzStencil[6]))) +
                       ((kyzStencil[7] - kyzStencil[0]) + (kyzStencil[4] - kyzStencil[3])) +
                       ((kyzStencil[5] - kyzStencil[2]) + (kyzStencil[6] - kyzStencil[1])));
        c002 = 0.0625f * (2.f * (((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                                ((kxxMzzStencil[2] - kxxMzzStencil[5]) + (kxxMzzStencil[1] - kxxMzzStencil[6])) +
                                ((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[1] + uxStencil[6])) +
                                ((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[1] + uyStencil[6]) - (uyStencil[2] + uyStencil[5]))) +
                        ((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                        ((kxxMyyStencil[5] - kxxMyyStencil[2]) + (kxxMyyStencil[6] - kxxMyyStencil[1])));
		
        a110 = 0.5f * (((uxStencil[7] + uxStencil[0]) + (uxStencil[4] + uxStencil[3])) - ((uxStencil[2] + uxStencil[5]) + (uxStencil[1] + uxStencil[6])));
        b110 = 0.5f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[4] + uyStencil[3])) - ((uyStencil[2] + uyStencil[5]) + (uyStencil[1] + uyStencil[6])));
        c110 = 0.5f * (((uzStencil[7] + uzStencil[0]) + (uzStencil[4] + uzStencil[3])) - ((uzStencil[2] + uzStencil[5]) + (uzStencil[1] + uzStencil[6])));

        a101 = 0.5f * (((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[1] + uxStencil[6])));
        b101 = 0.5f * (((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[2] + uyStencil[5]) - (uyStencil[1] + uyStencil[6])));
        c101 = 0.5f * (((uzStencil[7] + uzStencil[0]) - (uzStencil[4] + uzStencil[3])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[1] + uzStencil[6])));

        a011 = 0.5f * (((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[1] + uxStencil[6]) - (uxStencil[2] + uxStencil[5])));
        b011 = 0.5f * (((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[1] + uyStencil[6]) - (uyStencil[2] + uyStencil[5])));
        c011 = 0.5f * (((uzStencil[7] + uzStencil[0]) - (uzStencil[4] + uzStencil[3])) + ((uzStencil[1] + uzStencil[6]) - (uzStencil[2] + uzStencil[5])));

        a111 = ((uxStencil[7] - uxStencil[0]) + (uxStencil[4] - uxStencil[3])) + ((uxStencil[2] - uxStencil[5]) + (uxStencil[1] - uxStencil[6]));
        b111 = ((uyStencil[7] - uyStencil[0]) + (uyStencil[4] - uyStencil[3])) + ((uyStencil[2] - uyStencil[5]) + (uyStencil[1] - uyStencil[6]));
        c111 = ((uzStencil[7] - uzStencil[0]) + (uzStencil[4] - uzStencil[3])) + ((uzStencil[2] - uzStencil[5]) + (uzStencil[1] - uzStencil[6]));
		*/
		
		// get average second order moments
		// eq Schönherr 2015 (7.29 - 7.33)
		constexpr float kxyAvg = 0.f;
		constexpr float kyzAvg = 0.f;
		constexpr float kxzAvg = 0.f;
		constexpr float kxxMyyAvg = 0.f;
		constexpr float kxxMzzAvg = 0.f;
		
		const float LaplaceRho = - 3.f * (ax * ax + by * by + cz * cz) - 6.f * (bx * ay + cx * az + cy * bz);
		
		// get interpolated variables for the coarse cell
		const float dRho = d000 - 0.25f * LaplaceRho;
		const float ux = a000; 
		const float uy = b000; 
		const float uz = c000;
		const float rho  = dRho + 1.f;
		
		// calculate second order central moments
		// eq Schönherr 2015 (7.38 - 7.43)
		// note that A, B, C is all zeros because coarse cell is placed [0, 0, 0]
		const float sigma = 2.f; // fine to coarse
		const float k_011 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( (bz + cy) + kyzAvg );
		const float k_101 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( (az + cx) + kxzAvg );
		const float k_110 = - ( sigma * rho ) / ( 3.f * omega1Coarse ) * ( (ay + bx) + kxyAvg );
		
		const float mxxMyy = - (2.f / 3.f) * ((ax - by) + kxxMyyAvg) * sigma / omega1Coarse * rho;
		const float mxxMzz = - (2.f / 3.f) * ((ax - cz) + kxxMzzAvg) * sigma / omega1Coarse * rho;
		
		const float k_200 = (1.f / 3.f) * (       mxxMyy +       mxxMzz + dRho);
		const float k_020 = (1.f / 3.f) * (-2.f * mxxMyy +       mxxMzz + dRho);
		const float k_002 = (1.f / 3.f) * (       mxxMyy - 2.f * mxxMzz + dRho);
		
		// reconstruct f for the coarse cell
		float f[27];
		reconstructInterpolatedF( f, rho, ux, uy, uz, k_011, k_101, k_110, k_200, k_020, k_002 );
		
		// write reconstructed f into the coarse cell
		NBRStruct NBR;
		getCompressedNBR( cellCoarse, NBR, shifterViewCoarse, jPlusViewCoarse, kPlusViewCoarse, jkPlusViewCoarse, InfoCoarse );
		int cellWriteIndex[27];
		int fWriteIndex[27];
		getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperCoarse );
		for ( int direction = 0; direction < 27; direction++ ) fViewCoarse( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
	};
	
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.FineToCoarseInterface.interfaceCount, cellLambda );
}

void updateCoarseToFineInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	// The interpolation and rescaling is based on Martin Schönherr's disertation 2015
	const InfoStruct &InfoCoarse = GridCoarse.Info;
	auto fViewCoarse = GridCoarse.fArray.getView();
	const bool &esotwistFlipperCoarse = GridCoarse.esotwistFlipper;
	auto shifterViewCoarse = GridCoarse.IJKNBR.shifterArray.getConstView();
	auto jPlusViewCoarse = GridCoarse.IJKNBR.jPlusArray.getConstView();
	auto kPlusViewCoarse = GridCoarse.IJKNBR.kPlusArray.getConstView();
	auto jkPlusViewCoarse = GridCoarse.IJKNBR.jkPlusArray.getConstView();
	const float tauCoarse = 3.f * InfoCoarse.nu + 0.5f;
	const float omega1Coarse =  1.f / tauCoarse;
	
	const InfoStruct &InfoFine = GridFine.Info;
	auto fViewFine = GridFine.fArray.getView();
	const bool &esotwistFlipperFine = GridFine.esotwistFlipper;
	auto shifterViewFine = GridFine.IJKNBR.shifterArray.getConstView();
	auto jPlusViewFine = GridFine.IJKNBR.jPlusArray.getConstView();
	auto kPlusViewFine = GridFine.IJKNBR.kPlusArray.getConstView();
	auto jkPlusViewFine = GridFine.IJKNBR.jkPlusArray.getConstView();
	const float tauFine = 3.f * InfoFine.nu + 0.5f;
	const float omega1Fine =  1.f / tauFine;
	
	auto indexView = GridCoarse.CoarseToFineInterface.indexArray.getConstView();
	auto childMapView = GridCoarse.CoarseToFineInterface.childMapArray.getConstView();
	
	auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
	{
		const int cellCoarse0 = indexView( index );
		const int cellFine0 = childMapView( index );
		
		NBRStruct NBRStencil;
		getCompressedNBR( cellCoarse0, NBRStencil, shifterViewCoarse, jPlusViewCoarse, kPlusViewCoarse, jkPlusViewCoarse, InfoCoarse );
				
		int cellStencil[8];
		cellStencil[0] = NBRStencil.self;
		cellStencil[1] = NBRStencil.iPlus;
		cellStencil[2] = NBRStencil.jPlus;
		cellStencil[3] = NBRStencil.ijPlus;
		cellStencil[4] = NBRStencil.kPlus;
		cellStencil[5] = NBRStencil.ikPlus;
		cellStencil[6] = NBRStencil.jkPlus;
		cellStencil[7] = NBRStencil.ijkPlus;
		
		// Initialize stencil variables
		float dRhoStencil[8]; float uxStencil[8]; float uyStencil[8]; float uzStencil[8];
		float kxyStencil[8]; float kyzStencil[8]; float kxzStencil[8]; float kxxMyyStencil[8]; float kxxMzzStencil[8];
		
		// Extract values from each stencil cell
		for ( int i = 0; i < 8; i++ )
		{
			const int nbr = cellStencil[i];
			NBRStruct NBRofNBR;
			getCompressedNBR( nbr, NBRofNBR, shifterViewCoarse, jPlusViewCoarse, kPlusViewCoarse, jkPlusViewCoarse, InfoCoarse );
			int nbrCellReadIndex[27], nbrFReadIndex[27];
			getPreCollisionIndex( nbrCellReadIndex, nbrFReadIndex, NBRofNBR, esotwistFlipperCoarse );
			float fNbr[27];
			for ( int direction = 0; direction < 27; direction++ ) fNbr[direction] = fViewCoarse( nbrFReadIndex[direction], nbrCellReadIndex[direction] );
			
			getDRhoUxUyUz( dRhoStencil[i], uxStencil[i], uyStencil[i], uzStencil[i], fNbr );
			const float rho = 1.f + dRhoStencil[i];
			
			kxyStencil[i] = - 3.f * omega1Coarse * ( ( 
					+ fNbr[11] + fNbr[12] - fNbr[15] - fNbr[16] 
					- fNbr[19] - fNbr[20] + fNbr[21] + fNbr[22] - fNbr[23] - fNbr[24] + fNbr[25] + fNbr[26]
													) / rho - uxStencil[i] * uyStencil[i] );
			kyzStencil[i] = - 3.f * omega1Coarse * ( (
					- fNbr[13] - fNbr[14] + fNbr[17] + fNbr[18] 
					- fNbr[19] - fNbr[20] - fNbr[21] - fNbr[22] + fNbr[23] + fNbr[24] + fNbr[25] + fNbr[26]
													) / rho - uyStencil[i] * uzStencil[i] );
			kxzStencil[i] = - 3.f * omega1Coarse * ( (
					- fNbr[7 ] - fNbr[8 ] + fNbr[9 ] + fNbr[10] 
					+ fNbr[19] + fNbr[20] - fNbr[21] - fNbr[22] - fNbr[23] - fNbr[24] + fNbr[25] + fNbr[26]
													) / rho - uxStencil[i] * uzStencil[i] );
			kxxMyyStencil[i] = - 1.5f * omega1Coarse * ( (
					+ fNbr[1 ] + fNbr[2 ] - fNbr[5 ] - fNbr[6 ] 
					+ fNbr[7 ] + fNbr[8 ] + fNbr[9 ] + fNbr[10] - fNbr[13] - fNbr[14] - fNbr[17] - fNbr[18]
													) / rho - ( uxStencil[i] * uxStencil[i] - uyStencil[i] * uyStencil[i] ) );
			kxxMzzStencil[i] = - 1.5f * omega1Coarse * ( (
					+ fNbr[1 ] + fNbr[2 ] - fNbr[3 ] - fNbr[4 ] 
					+ fNbr[11] + fNbr[12] - fNbr[13] - fNbr[14] + fNbr[15] + fNbr[16] - fNbr[17] - fNbr[18]
													) / rho - ( uxStencil[i] * uxStencil[i] - uzStencil[i] * uzStencil[i] ) );
		}
		
		// get all required coefficients
		// eq Schönherr 2015 (7.10)
		
		// The following is directly taken from VirtualFluids (just renamed variables). https://github.com/irmb/virtualfluids 
		
		const float d000 = 0.125f * (((dRhoStencil[7] + dRhoStencil[0]) + (dRhoStencil[3] + dRhoStencil[4])) + ((dRhoStencil[1] + dRhoStencil[6]) + (dRhoStencil[5] + dRhoStencil[2])));
        const float d100 = 0.25f * (((dRhoStencil[7] - dRhoStencil[0]) + (dRhoStencil[3] - dRhoStencil[4])) + ((dRhoStencil[1] - dRhoStencil[6]) + (dRhoStencil[5] - dRhoStencil[2])));
        const float d010 = 0.25f * (((dRhoStencil[7] - dRhoStencil[0]) + (dRhoStencil[3] - dRhoStencil[4])) + ((dRhoStencil[6] - dRhoStencil[1]) + (dRhoStencil[2] - dRhoStencil[5])));
        const float d001 = 0.25f * (((dRhoStencil[7] - dRhoStencil[0]) + (dRhoStencil[4] - dRhoStencil[3])) + ((dRhoStencil[6] - dRhoStencil[1]) + (dRhoStencil[5] - dRhoStencil[2])));
        const float d110 = 0.5f * (((dRhoStencil[7] + dRhoStencil[0]) + (dRhoStencil[3] + dRhoStencil[4])) - ((dRhoStencil[1] + dRhoStencil[6]) + (dRhoStencil[5] + dRhoStencil[2])));
        const float d101 = 0.5f * (((dRhoStencil[7] + dRhoStencil[0]) - (dRhoStencil[3] + dRhoStencil[4])) + ((dRhoStencil[5] + dRhoStencil[2]) - (dRhoStencil[1] + dRhoStencil[6])));
        const float d011 = 0.5f * (((dRhoStencil[7] + dRhoStencil[0]) - (dRhoStencil[3] + dRhoStencil[4])) + ((dRhoStencil[1] + dRhoStencil[6]) - (dRhoStencil[5] + dRhoStencil[2])));
        const float d111 = (((dRhoStencil[7] - dRhoStencil[0]) + (dRhoStencil[4] - dRhoStencil[3])) + ((dRhoStencil[1] - dRhoStencil[6]) + (dRhoStencil[2] - dRhoStencil[5])));
		
		const float a000 = 0.015625f * (2.f * (((kxyStencil[0] - kxyStencil[7]) + (kxyStencil[4] - kxyStencil[3])) +
                                ((kxyStencil[1] - kxyStencil[6]) + (kxyStencil[5] - kxyStencil[2])) +
                                ((kxzStencil[0] - kxzStencil[7]) + (kxzStencil[3] - kxzStencil[4])) +
                                ((kxzStencil[1] - kxzStencil[6]) + (kxzStencil[2] - kxzStencil[5])) +
                                ((uyStencil[7] + uyStencil[0]) + (uyStencil[3] + uyStencil[4])) - ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[5] + uzStencil[2]) - (uzStencil[6] + uzStencil[1]))) +
                        8.f * (((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) + ((uxStencil[6] + uxStencil[1]) + (uxStencil[5] + uxStencil[2]))) +
                        ((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                        ((kxxMyyStencil[6] - kxxMyyStencil[1]) + (kxxMyyStencil[2] - kxxMyyStencil[5])) +
                        ((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                        ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[2] - kxxMzzStencil[5])));
        const float b000 = 0.015625f * (2.f * (((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                                ((kxxMyyStencil[6] - kxxMyyStencil[1]) + (kxxMyyStencil[2] - kxxMyyStencil[5])) +
                                ((kxyStencil[0] - kxyStencil[7]) + (kxyStencil[4] - kxyStencil[3])) +
                                ((kxyStencil[6] - kxyStencil[1]) + (kxyStencil[2] - kxyStencil[5])) +
                                ((kyzStencil[0] - kyzStencil[7]) + (kyzStencil[3] - kyzStencil[4])) +
                                ((kyzStencil[1] - kyzStencil[6]) + (kyzStencil[2] - kyzStencil[5])) +
                                ((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) - ((uxStencil[2] + uxStencil[6]) + (uxStencil[1] + uxStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[2] + uzStencil[5]))) +
                        8.f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[3] + uyStencil[4])) + ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5]))) +
                        ((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                        ((kxxMzzStencil[1] - kxxMzzStencil[6]) + (kxxMzzStencil[5] - kxxMzzStencil[2])));
        const float c000 = 0.015625f * (2.f * (((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[4] - kxxMzzStencil[3])) +
                                ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[5] - kxxMzzStencil[2])) +
                                ((kxzStencil[0] - kxzStencil[7]) + (kxzStencil[4] - kxzStencil[3])) +
                                ((kxzStencil[6] - kxzStencil[1]) + (kxzStencil[2] - kxzStencil[5])) +
                                ((kyzStencil[0] - kyzStencil[7]) + (kyzStencil[4] - kyzStencil[3])) +
                                ((kyzStencil[1] - kyzStencil[6]) + (kyzStencil[5] - kyzStencil[2])) +
                                ((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[6] + uxStencil[1])) +
                                ((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[6] + uyStencil[1]) - (uyStencil[2] + uyStencil[5]))) +
                        8.f * (((uzStencil[7] + uzStencil[0]) + (uzStencil[3] + uzStencil[4])) + ((uzStencil[1] + uzStencil[6]) + (uzStencil[5] + uzStencil[2]))) +
                        ((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                        ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[2] - kxxMyyStencil[5])));

        const float a100 = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[3] - uxStencil[4])) + ((uxStencil[1] - uxStencil[6]) + (uxStencil[5] - uxStencil[2])));
        const float b100 = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[3] - uyStencil[4])) + ((uyStencil[1] - uyStencil[6]) + (uyStencil[5] - uyStencil[2])));
        const float c100 = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[3] - uzStencil[4])) + ((uzStencil[1] - uzStencil[6]) + (uzStencil[5] - uzStencil[2])));

        const float a010 = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[3] - uxStencil[4])) + ((uxStencil[6] - uxStencil[1]) + (uxStencil[2] - uxStencil[5])));
        const float b010 = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[3] - uyStencil[4])) + ((uyStencil[6] - uyStencil[1]) + (uyStencil[2] - uyStencil[5])));
        const float c010 = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[3] - uzStencil[4])) + ((uzStencil[6] - uzStencil[1]) + (uzStencil[2] - uzStencil[5])));

        const float a001 = 0.25f * (((uxStencil[7] - uxStencil[0]) + (uxStencil[4] - uxStencil[3])) + ((uxStencil[6] - uxStencil[1]) + (uxStencil[5] - uxStencil[2])));
        const float b001 = 0.25f * (((uyStencil[7] - uyStencil[0]) + (uyStencil[4] - uyStencil[3])) + ((uyStencil[6] - uyStencil[1]) + (uyStencil[5] - uyStencil[2])));
        const float c001 = 0.25f * (((uzStencil[7] - uzStencil[0]) + (uzStencil[4] - uzStencil[3])) + ((uzStencil[6] - uzStencil[1]) + (uzStencil[5] - uzStencil[2])));
		
		const float a200 = 0.0625f * (2.f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[3] - uyStencil[6])) + ((uyStencil[4] - uyStencil[1]) - (uyStencil[2] + uyStencil[5])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[6])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[4] + uzStencil[1]))) +
                        ((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[3] - kxxMyyStencil[4])) +
                        ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[5] - kxxMyyStencil[2])) +
                        ((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                        ((kxxMzzStencil[1] - kxxMzzStencil[6]) + (kxxMzzStencil[5] - kxxMzzStencil[2])));
        const float b200 = 0.125f * (2.f * (-((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) + ((uxStencil[6] + uxStencil[1]) + (uxStencil[2] + uxStencil[5]))) +
                       ((kxyStencil[7] - kxyStencil[0]) + (kxyStencil[3] - kxyStencil[4])) +
                       ((kxyStencil[1] - kxyStencil[6]) + (kxyStencil[5] - kxyStencil[2])));
        const float c200 = 0.125f * (2.f * (((uxStencil[3] + uxStencil[4]) - (uxStencil[7] + uxStencil[0])) + ((uxStencil[6] + uxStencil[1]) - (uxStencil[2] + uxStencil[5]))) +
                       ((kxzStencil[7] - kxzStencil[0]) + (kxzStencil[3] - kxzStencil[4])) +
                       ((kxzStencil[1] - kxzStencil[6]) + (kxzStencil[5] - kxzStencil[2])));
        
        const float a020 = 0.125f * (2.f * (-((uyStencil[7] + uyStencil[0]) + (uyStencil[4] + uyStencil[3])) + ((uyStencil[6] + uyStencil[1]) + (uyStencil[2] + uyStencil[5]))) +
                       ((kxyStencil[7] - kxyStencil[0]) + (kxyStencil[3] - kxyStencil[4])) +
                       ((kxyStencil[6] - kxyStencil[1]) + (kxyStencil[2] - kxyStencil[5])));
        const float b020 = 0.0625f * (2.f * (((kxxMyyStencil[0] - kxxMyyStencil[7]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                                ((kxxMyyStencil[1] - kxxMyyStencil[6]) + (kxxMyyStencil[5] - kxxMyyStencil[2])) +
                                ((uxStencil[7] + uxStencil[0]) + (uxStencil[3] + uxStencil[4])) - ((uxStencil[6] + uxStencil[1]) + (uxStencil[5] + uxStencil[2])) +
                                ((uzStencil[7] + uzStencil[0]) - (uzStencil[3] + uzStencil[4])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[2] + uzStencil[5]))) +
                        ((kxxMzzStencil[7] - kxxMzzStencil[0]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                        ((kxxMzzStencil[6] - kxxMzzStencil[1]) + (kxxMzzStencil[2] - kxxMzzStencil[5])));
        const float c020 = 0.125f * (2.f * (((uyStencil[4] + uyStencil[3]) - (uyStencil[7] + uyStencil[0])) + ((uyStencil[5] + uyStencil[2]) - (uyStencil[6] + uyStencil[1]))) +
                       ((kyzStencil[7] - kyzStencil[0]) + (kyzStencil[3] - kyzStencil[4])) +
                       ((kyzStencil[6] - kyzStencil[1]) + (kyzStencil[2] - kyzStencil[5])));
                 
        const float a002 = 0.125f * (2.f * (((uzStencil[3] + uzStencil[4]) - (uzStencil[7] + uzStencil[0])) + ((uzStencil[6] + uzStencil[1]) - (uzStencil[5] + uzStencil[2]))) +
                       ((kxzStencil[7] - kxzStencil[0]) + (kxzStencil[4] - kxzStencil[3])) +
                       ((kxzStencil[5] - kxzStencil[2]) + (kxzStencil[6] - kxzStencil[1])));
        const float b002 = 0.125f * (2.f * (((uzStencil[3] + uzStencil[4]) - (uzStencil[7] + uzStencil[0])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[1] + uzStencil[6]))) +
                       ((kyzStencil[7] - kyzStencil[0]) + (kyzStencil[4] - kyzStencil[3])) +
                       ((kyzStencil[5] - kyzStencil[2]) + (kyzStencil[6] - kyzStencil[1])));
        const float c002 = 0.0625f * (2.f * (((kxxMzzStencil[0] - kxxMzzStencil[7]) + (kxxMzzStencil[3] - kxxMzzStencil[4])) +
                                ((kxxMzzStencil[2] - kxxMzzStencil[5]) + (kxxMzzStencil[1] - kxxMzzStencil[6])) +
                                ((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[1] + uxStencil[6])) +
                                ((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[1] + uyStencil[6]) - (uyStencil[2] + uyStencil[5]))) +
                        ((kxxMyyStencil[7] - kxxMyyStencil[0]) + (kxxMyyStencil[4] - kxxMyyStencil[3])) +
                        ((kxxMyyStencil[5] - kxxMyyStencil[2]) + (kxxMyyStencil[6] - kxxMyyStencil[1])));
		
        const float a110 = 0.5f * (((uxStencil[7] + uxStencil[0]) + (uxStencil[4] + uxStencil[3])) - ((uxStencil[2] + uxStencil[5]) + (uxStencil[1] + uxStencil[6])));
        const float b110 = 0.5f * (((uyStencil[7] + uyStencil[0]) + (uyStencil[4] + uyStencil[3])) - ((uyStencil[2] + uyStencil[5]) + (uyStencil[1] + uyStencil[6])));
        const float c110 = 0.5f * (((uzStencil[7] + uzStencil[0]) + (uzStencil[4] + uzStencil[3])) - ((uzStencil[2] + uzStencil[5]) + (uzStencil[1] + uzStencil[6])));

        const float a101 = 0.5f * (((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[2] + uxStencil[5]) - (uxStencil[1] + uxStencil[6])));
        const float b101 = 0.5f * (((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[2] + uyStencil[5]) - (uyStencil[1] + uyStencil[6])));
        const float c101 = 0.5f * (((uzStencil[7] + uzStencil[0]) - (uzStencil[4] + uzStencil[3])) + ((uzStencil[2] + uzStencil[5]) - (uzStencil[1] + uzStencil[6])));

        const float a011 = 0.5f * (((uxStencil[7] + uxStencil[0]) - (uxStencil[4] + uxStencil[3])) + ((uxStencil[1] + uxStencil[6]) - (uxStencil[2] + uxStencil[5])));
        const float b011 = 0.5f * (((uyStencil[7] + uyStencil[0]) - (uyStencil[4] + uyStencil[3])) + ((uyStencil[1] + uyStencil[6]) - (uyStencil[2] + uyStencil[5])));
        const float c011 = 0.5f * (((uzStencil[7] + uzStencil[0]) - (uzStencil[4] + uzStencil[3])) + ((uzStencil[1] + uzStencil[6]) - (uzStencil[2] + uzStencil[5])));

        const float a111 = ((uxStencil[7] - uxStencil[0]) + (uxStencil[4] - uxStencil[3])) + ((uxStencil[2] - uxStencil[5]) + (uxStencil[1] - uxStencil[6]));
        const float b111 = ((uyStencil[7] - uyStencil[0]) + (uyStencil[4] - uyStencil[3])) + ((uyStencil[2] - uyStencil[5]) + (uyStencil[1] - uyStencil[6]));
        const float c111 = ((uzStencil[7] - uzStencil[0]) + (uzStencil[4] - uzStencil[3])) + ((uzStencil[2] - uzStencil[5]) + (uzStencil[1] - uzStencil[6]));
        
        const float LaplaceRho = -3.f * (a100 * a100 + b010 * b010 + c001 * c001) - 6.f * (b100 * a010 + c100 * a001 + c010 * b001);
		
		constexpr float kxyAvg = 0.f;
		constexpr float kyzAvg = 0.f;
		constexpr float kxzAvg = 0.f;
		constexpr float kxxMyyAvg = 0.f;
		constexpr float kxxMzzAvg = 0.f;
		
		// build list of fine cells and their positions
		NBRStruct NBRTarget;
		getCompressedNBR( cellFine0, NBRTarget, shifterViewFine, jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );
		int cellTarget[8];
		cellTarget[0] = NBRTarget.self;
		cellTarget[1] = NBRTarget.iPlus;
		cellTarget[2] = NBRTarget.jPlus;
		cellTarget[3] = NBRTarget.ijPlus;
		cellTarget[4] = NBRTarget.kPlus;
		cellTarget[5] = NBRTarget.ikPlus;
		cellTarget[6] = NBRTarget.jkPlus;
		cellTarget[7] = NBRTarget.ijkPlus;
		
		const float dxArray[8] = {-0.25f, +0.25f, -0.25f, +0.25f, -0.25f, +0.25f, -0.25f, +0.25f};
		const float dyArray[8] = {-0.25f, -0.25f, +0.25f, +0.25f, -0.25f, -0.25f, +0.25f, +0.25f};
		const float dzArray[8] = {-0.25f, -0.25f, -0.25f, -0.25f, +0.25f, +0.25f, +0.25f, +0.25f};
		
		for ( int i = 0; i < 8; i++ )
		{
			const int cellFine = cellTarget[i];
			const float dx = dxArray[i];
			const float dy = dyArray[i];
			const float dz = dzArray[i];
			// get interpolated variables for the fine cell
			const float dRho = d000 + d100 * dx + d010 * dy + d001 * dz + d110 * dx * dy + d101 * dx * dz + d011 * dy * dz + d111 * dx * dy * dz 
								+ 3.f * dx * dx * LaplaceRho;
			const float ux = a000 + a100 * dx + a010 * dy + a001 * dz + a110 * dx * dy + a101 * dx * dz + a011 * dy * dz + a111 * dx * dy * dz
								+ a200 * dx * dx + a020 * dy * dy + a002 * dz * dz; 
			const float uy = b000 + b100 * dx + b010 * dy + b001 * dz + b110 * dx * dy + b101 * dx * dz + b011 * dy * dz + b111 * dx * dy * dz
								+ b200 * dx * dx + b020 * dy * dy + b002 * dz * dz; 
			const float uz = c000 + c100 * dx + c010 * dy + c001 * dz + c110 * dx * dy + c101 * dx * dz + c011 * dy * dz + c111 * dx * dy * dz
								+ c200 * dx * dx + c020 * dy * dy + c002 * dz * dz; 
			const float rho = dRho + 1.f;
			
			// calculate second order central moments
			
			const float sigma = 0.5f; // coarse to fine
			
			const float k_011 = -(1.f / 3.f) * (b001 + c010 + kyzAvg + b101 * dx + c110 * dx + b011 * dy + 2.f * c020 * dy
					+ b111 * dx * dy + 2.f * b002 * dz + c011 * dz + c111 * dx * dz) * sigma / omega1Fine * (1.f + dRho);
			const float k_101 = -(1.f / 3.f) * (a001 + c100 + kxzAvg + a101 * dx + 2.f * c200 * dx + a011 * dy + c110 * dy
					+ a111 * dx * dy + 2.f * a002 * dz + c101 * dz + c111 * dy * dz) * sigma / omega1Fine * (1.f + dRho);
			const float k_110 = -(1.f / 3.f) * (a010 + b100 + kxyAvg + a110 * dx + 2.f * b200 * dx + 2.f * a020 * dy
					+ b110 * dy + a011 * dz + b101 * dz + a111 * dx * dz + b111 * dy * dz) * sigma / omega1Fine * (1.f + dRho);
			
			const float mxxMyy = -(2.f/3.f) * (a100 - b010 + kxxMyyAvg + 2.f * a200 * dx - b110 * dx + a110 * dy
						  -2.f * b020 * dy + a101 * dz - b011 * dz - b111 * dx * dz + a111 * dy * dz) * sigma / omega1Fine * (1.f + dRho);
			const float mxxMzz = -(2.f/3.f) * (a100 - c001 + kxxMzzAvg + 2.f * a200 * dx - c101 * dx + a110 * dy
						  -c011 * dy - c111 * dx * dy + a101 * dz - 2.f * c002 * dz + a111 * dy * dz) * sigma / omega1Fine * (1.f + dRho);
			
			const float k_200 = (1.f / 3.f) * (       mxxMyy +       mxxMzz + dRho);
			const float k_020 = (1.f / 3.f) * (-2.f * mxxMyy +       mxxMzz + dRho);
			const float k_002 = (1.f / 3.f) * (       mxxMyy - 2.f * mxxMzz + dRho);
			
			// reconstruct f for the fine cell
			float f[27];
			reconstructInterpolatedF( f, rho, ux, uy, uz, k_011, k_101, k_110, k_200, k_020, k_002 );
			
			// write reconstructed f into the fine cell
			NBRStruct NBR;
			getCompressedNBR( cellFine, NBR, shifterViewFine, jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );
			int cellWriteIndex[27];
			int fWriteIndex[27];
			getPreCollisionIndex( cellWriteIndex, fWriteIndex, NBR, esotwistFlipperFine );
			for ( int direction = 0; direction < 27; direction++ ) fViewFine( fWriteIndex[direction], cellWriteIndex[direction] ) = f[direction];
		}
	};
	
	TNL::Algorithms::parallelFor<TNL::Devices::Cuda>(0, GridCoarse.CoarseToFineInterface.interfaceCount, cellLambda );
}

void updateInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
	updateFineToCoarseInterface( GridCoarse, GridFine );
	updateCoarseToFineInterface( GridCoarse, GridFine );
}
