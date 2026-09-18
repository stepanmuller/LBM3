#pragma once

#include "./D3Q27Directions.h"
#include "./NBRFunctions.h"

// id: { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 };
// cx: { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 };
// cy: { 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 };
// cz: { 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 };

// cx * cx: { 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
// cy * cy: { 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
// cz * cz: { 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

// cy * cz: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1 };
// cx * cz: { 0, 0, 0, 0, 0, 0, 0,-1,-1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1,-1,-1, 1, 1 };
// cx * cy: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,-1,-1, 0, 0,-1,-1, 1, 1,-1,-1, 1, 1 };

// w:  { 8/27, 2/27, 2/27, 2/27 , 2/27, 2/27, 2/27, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216 };

__host__ __device__ inline void getCompressedIJKNBR( const int &cell, int& iCell, int& jCell, int& kCell, NBRStruct &NBR,
								IntConstViewType& shifterView, IntConstViewType& iView, IntConstViewType& jView, IntConstViewType& kView,
								IntConstViewType& jPlusView, IntConstViewType& kPlusView, IntConstViewType& jkPlusView, 
								const InfoStruct &Info )
{
    NBR.self = cell;
    const int shift = shifterView(cell);
	if ( shift >= 0 ) 
	{ 
		iCell = iView( shift ); 
		jCell = jView( shift ); 
		kCell = kView( shift ); 
		NBR.jPlus = jPlusView( shift );
		NBR.kPlus = kPlusView( shift );
		NBR.jkPlus = jkPlusView( shift );
	}
	else 
	{ 
		const int firstInRow = cell + shift; 
		const int compressedIndex = shifterView( firstInRow );
		iCell = iView( compressedIndex ) - shift; 
		jCell = jView( compressedIndex ); 
		kCell = kView( compressedIndex );
		NBR.jPlus = jPlusView( compressedIndex ) - shift;
		NBR.kPlus = kPlusView( compressedIndex ) - shift;
		NBR.jkPlus = jkPlusView( compressedIndex ) - shift;
	}
	finishNBRPlus( NBR, Info );
}

__host__ __device__ inline void getCompressedIJK( const int &cell, int& iCell, int& jCell, int& kCell,
								const IntConstViewType& shifterView, const IntConstViewType& iView, const IntConstViewType& jView, const IntConstViewType& kView,
								const InfoStruct &Info )
{
    const int shift = shifterView(cell);
	if ( shift >= 0 ) 
	{ 
		iCell = iView( shift ); 
		jCell = jView( shift ); 
		kCell = kView( shift ); 
	}
	else 
	{ 
		const int firstInRow = cell + shift; 
		const int compressedIndex = shifterView( firstInRow );
		iCell = iView( compressedIndex ) - shift; 
		jCell = jView( compressedIndex ); 
		kCell = kView( compressedIndex );
	}
}

__host__ __device__ inline void getCompressedNBR( const int &cell, NBRStruct &NBR,
								IntConstViewType& shifterView,
								IntConstViewType& jPlusView, IntConstViewType& kPlusView, IntConstViewType& jkPlusView, 
								const InfoStruct &Info )
{
    NBR.self = cell;
    const int shift = shifterView(cell);
	if ( shift >= 0 ) 
	{ 
		NBR.jPlus = jPlusView( shift );
		NBR.kPlus = kPlusView( shift );
		NBR.jkPlus = jkPlusView( shift );
	}
	else 
	{ 
		const int firstInRow = cell + shift; 
		const int compressedIndex = shifterView( firstInRow );
		NBR.jPlus = jPlusView( compressedIndex ) - shift;
		NBR.kPlus = kPlusView( compressedIndex ) - shift;
		NBR.jkPlus = jkPlusView( compressedIndex ) - shift;
	}
	finishNBRPlus( NBR, Info );
}

__host__ __device__ void getIJKCellIndexFromXYZ( int& iCell, int& jCell, int& kCell, const float &x, const float &y, const float &z, const InfoStruct &Info)
{
    iCell = (int)(( x - Info.ox ) / Info.res + 0.5f);
    jCell = (int)(( y - Info.oy ) / Info.res + 0.5f);
    kCell = (int)(( z - Info.oz ) / Info.res + 0.5f);
}

__host__ __device__ void getXYZFromIJKCellIndex( const int& iCell, const int& jCell, const int& kCell, float &x, float &y, float &z, const InfoStruct &Info)
{
    x = iCell * Info.res + Info.ox;
    y = jCell * Info.res + Info.oy;
    z = kCell * Info.res + Info.oz;
}

__host__ __device__ void getOuterNormal( 	const int& iCell, const int& jCell, const int& kCell,
										int& outerNormalX, int& outerNormalY, int& outerNormalZ, const InfoStruct &Info )
{
    outerNormalX = 0;
    outerNormalY = 0;
    outerNormalZ = 0;
    if 			( iCell == 0 ) 						outerNormalX = -1;
    else if 	( iCell == Info.cellCountX - 1 ) 	outerNormalX = 1;
    if 			( jCell == 0 ) 						outerNormalY = -1;
    else if 	( jCell == Info.cellCountY - 1) 	outerNormalY = 1;
    if 			( kCell == 0 ) 						outerNormalZ = -1;
    else if 	( kCell == Info.cellCountZ - 1 ) 	outerNormalZ = 1;
}

__host__ __device__ void getFeq(
	const float &dRho, const float &ux, const float &uy, const float &uz, 
	float (&feq)[27]
	)
{
	const float u2 = ux*ux + uy*uy + uz*uz;

	const float cu0  = 0.f;
	const float cu1  = +ux;
	const float cu2  = -ux;
	const float cu3  = -uz;
	const float cu4  = +uz;
	const float cu5  = -uy;
	const float cu6  = +uy;
	const float cu7  = +ux -uz;
	const float cu8  = -ux +uz;
	const float cu9  = +ux +uz;
	const float cu10 = -ux -uz;
	const float cu11 = -ux -uy;
	const float cu12 = +ux +uy;
	const float cu13 = +uy -uz;
	const float cu14 = -uy +uz;
	const float cu15 = -ux +uy;
	const float cu16 = +ux -uy;
	const float cu17 = +uy +uz;
	const float cu18 = -uy -uz;
	const float cu19 = -ux +uy -uz;
	const float cu20 = +ux -uy +uz;
	const float cu21 = -ux -uy +uz;
	const float cu22 = +ux +uy -uz;
	const float cu23 = +ux -uy -uz;
	const float cu24 = -ux +uy +uz;
	const float cu25 = -ux -uy -uz;
	const float cu26 = +ux +uy +uz;

	constexpr float w0  = 8.f/27.f;
	constexpr float w1  = 2.f/27.f;
	constexpr float w2  = 1.f/54.f;
	constexpr float w3 = 1.f/216.f;

	feq[0]  = w0 * (dRho + (3.f*cu0  + 4.5f*cu0 *cu0  - 1.5f*u2) * (dRho + 1.f));
	feq[1]  = w1 * (dRho + (3.f*cu1  + 4.5f*cu1 *cu1  - 1.5f*u2) * (dRho + 1.f));
	feq[2]  = w1 * (dRho + (3.f*cu2  + 4.5f*cu2 *cu2  - 1.5f*u2) * (dRho + 1.f));
	feq[3]  = w1 * (dRho + (3.f*cu3  + 4.5f*cu3 *cu3  - 1.5f*u2) * (dRho + 1.f));
	feq[4]  = w1 * (dRho + (3.f*cu4  + 4.5f*cu4 *cu4  - 1.5f*u2) * (dRho + 1.f));
	feq[5]  = w1 * (dRho + (3.f*cu5  + 4.5f*cu5 *cu5  - 1.5f*u2) * (dRho + 1.f));
	feq[6]  = w1 * (dRho + (3.f*cu6  + 4.5f*cu6 *cu6  - 1.5f*u2) * (dRho + 1.f));
	feq[7]  = w2 * (dRho + (3.f*cu7  + 4.5f*cu7 *cu7  - 1.5f*u2) * (dRho + 1.f));
	feq[8]  = w2 * (dRho + (3.f*cu8  + 4.5f*cu8 *cu8  - 1.5f*u2) * (dRho + 1.f));
	feq[9]  = w2 * (dRho + (3.f*cu9  + 4.5f*cu9 *cu9  - 1.5f*u2) * (dRho + 1.f));
	feq[10] = w2 * (dRho + (3.f*cu10 + 4.5f*cu10*cu10 - 1.5f*u2) * (dRho + 1.f));
	feq[11] = w2 * (dRho + (3.f*cu11 + 4.5f*cu11*cu11 - 1.5f*u2) * (dRho + 1.f));
	feq[12] = w2 * (dRho + (3.f*cu12 + 4.5f*cu12*cu12 - 1.5f*u2) * (dRho + 1.f));
	feq[13] = w2 * (dRho + (3.f*cu13 + 4.5f*cu13*cu13 - 1.5f*u2) * (dRho + 1.f));
	feq[14] = w2 * (dRho + (3.f*cu14 + 4.5f*cu14*cu14 - 1.5f*u2) * (dRho + 1.f));
	feq[15] = w2 * (dRho + (3.f*cu15 + 4.5f*cu15*cu15 - 1.5f*u2) * (dRho + 1.f));
	feq[16] = w2 * (dRho + (3.f*cu16 + 4.5f*cu16*cu16 - 1.5f*u2) * (dRho + 1.f));
	feq[17] = w2 * (dRho + (3.f*cu17 + 4.5f*cu17*cu17 - 1.5f*u2) * (dRho + 1.f));
	feq[18] = w2 * (dRho + (3.f*cu18 + 4.5f*cu18*cu18 - 1.5f*u2) * (dRho + 1.f));
	feq[19] = w3 * (dRho + (3.f*cu19 + 4.5f*cu19*cu19 - 1.5f*u2) * (dRho + 1.f));
	feq[20] = w3 * (dRho + (3.f*cu20 + 4.5f*cu20*cu20 - 1.5f*u2) * (dRho + 1.f));
	feq[21] = w3 * (dRho + (3.f*cu21 + 4.5f*cu21*cu21 - 1.5f*u2) * (dRho + 1.f));
	feq[22] = w3 * (dRho + (3.f*cu22 + 4.5f*cu22*cu22 - 1.5f*u2) * (dRho + 1.f));
	feq[23] = w3 * (dRho + (3.f*cu23 + 4.5f*cu23*cu23 - 1.5f*u2) * (dRho + 1.f));
	feq[24] = w3 * (dRho + (3.f*cu24 + 4.5f*cu24*cu24 - 1.5f*u2) * (dRho + 1.f));
	feq[25] = w3 * (dRho + (3.f*cu25 + 4.5f*cu25*cu25 - 1.5f*u2) * (dRho + 1.f));
	feq[26] = w3 * (dRho + (3.f*cu26 + 4.5f*cu26*cu26 - 1.5f*u2) * (dRho + 1.f));	
}

__host__ __device__ inline float getFeqSingle( const float &dRho, const float &ux, const float &uy, const float &uz, const int direction )
{
    const float u2 = ux*ux + uy*uy + uz*uz;
    const float cu = CX_DIRECTIONS[direction] * ux + CY_DIRECTIONS[direction] * uy + CZ_DIRECTIONS[direction] * uz;
    return DIRECTION_WEIGHTS[direction] * (dRho + (3.f*cu + 4.5f*cu*cu - 1.5f*u2) * (dRho + 1.f));
}

__host__ __device__ void getFneq(const float (&f)[27], const float (&feq)[27], float (&fneq)[27])
{
	for ( int i = 0; i < 27; i++ ) fneq[i] = f[i] - feq[i];
}

__host__ __device__ void getDRhoUxUyUz( float &dRho, float &ux, float &uy, float &uz, const float (&f)[27])
{
	dRho = (((f[PPP]+f[MMM]) + (f[PMP]+f[MPM])) + ((f[PPM]+f[MMP]) + (f[PMM]+f[MPP])))
					  + (((f[OPP]+f[OMM]) + (f[OPM]+f[OMP])) + ((f[POP]+f[MOM]) + (f[POM]+f[MOP])) + ((f[PPO]+f[MMO]) + (f[PMO]+f[MPO])))
						+ ((f[POO]+f[MOO]) + (f[OPO]+f[OMO]) + (f[OOP]+f[OOM])) + f[OOO];			
    
    const float rhoInv = 1.f / (dRho + 1.f);
    
    const float momentumX = ((((f[PPP]-f[MMM]) + (f[PMP]-f[MPM])) + ((f[PPM]-f[MMP]) + (f[PMM]-f[MPP])))
                          + (((f[POP]-f[MOM]) + (f[POM]-f[MOP])) + ((f[PPO]-f[MMO]) + (f[PMO]-f[MPO])))
                            + (f[POO]-f[MOO]));

    const float momentumY = ((((f[PPP]-f[MMM]) - (f[PMP]-f[MPM])) + ((f[PPM]-f[MMP]) - (f[PMM]-f[MPP])))
                          + (((f[OPP]-f[OMM]) + (f[OPM]-f[OMP])) + ((f[PPO]-f[MMO]) - (f[PMO]-f[MPO])))
                            + (f[OPO]-f[OMO]));

    const float momentumZ = ((((f[PPP]-f[MMM]) + (f[PMP]-f[MPM])) - ((f[PPM]-f[MMP]) + (f[PMM]-f[MPP])))
                          + (((f[OPP]-f[OMM]) - (f[OPM]-f[OMP])) + ((f[POP]-f[MOM]) - (f[POM]-f[MOP])))
                            + (f[OOP]-f[OOM]));
    ux = momentumX * rhoInv;
    uy = momentumY * rhoInv;
    uz = momentumZ * rhoInv;
}

__host__ __device__ void convertToPhysicalVelocity( float &ux, float &uy, float &uz, const InfoStruct &Info )
{
	ux = ux * (Info.res/1000.f) / Info.dtPhys;
	uy = uy * (Info.res/1000.f) / Info.dtPhys;
	uz = uz * (Info.res/1000.f) / Info.dtPhys;
}

__host__ __device__ void convertToPhysicalPressure( float &dRho, const InfoStruct &Info )
{
	// converts LBM dRho to physical pressure, overwrites the variable (LBM dRho -> physical p)
	float soundspeedPhys = INVSQRT3 * (Info.res/1000.f) / Info.dtPhys;
	const float p = dRho * RHO_PHYS * soundspeedPhys * soundspeedPhys;
	dRho = p;
}

__host__ __device__ void convertToPhysicalForce( float &gx, float &gy, float &gz, const InfoStruct &Info )
{
	gx = gx * RHO_PHYS * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
	gy = gy * RHO_PHYS * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
	gz = gz * RHO_PHYS * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) * (Info.res/1000.f) / (Info.dtPhys * Info.dtPhys);
}
