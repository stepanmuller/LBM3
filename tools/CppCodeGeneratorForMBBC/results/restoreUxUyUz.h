__host__ __device__ void restoreUxUyUz(
	const int &outerNormalX, const int &outerNormalY, const int &outerNormalZ,
	BCStruct &BC,
	const float (&f)[27]
)
{
	const float rho = BC.dRho + 1.f;
	float &ux = BC.ux;
	float &uy = BC.uy;
	float &uz = BC.uz;
	
	const int normalCode = (outerNormalX + 5) * 100 + (outerNormalY + 5) * 10 + (outerNormalZ + 5);
	if ( normalCode == 655 )
	{
		// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.
		const float fZero = + f[0] + f[3] + f[4] + f[5] + f[6] + f[13] + f[14] + f[17] + f[18];
		const float fPlus = + f[1] + f[7] + f[9] + f[12] + f[16] + f[20] + f[22] + f[23] + f[26];
		const float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;
		ux = uNormal;
		// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).
		const float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;
		const float jZeroy = - f[5] + f[6] + f[13] - f[14] + f[17] - f[18];
		const float jPlusy = + f[12] - f[16] - f[20] + f[22] - f[23] + f[26];
		uy = ((2.f * uNormal + 1.f) * jZeroy + 4.f * uNormal * jPlusy) / tangentDenominator;
		const float jZeroz = - f[3] + f[4] - f[13] + f[14] + f[17] - f[18];
		const float jPlusz = - f[7] + f[9] + f[20] - f[22] - f[23] + f[26];
		uz = ((2.f * uNormal + 1.f) * jZeroz + 4.f * uNormal * jPlusz) / tangentDenominator;
	}
	else if ( normalCode == 565 )
	{
		// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.
		const float fZero = + f[0] + f[1] + f[2] + f[3] + f[4] + f[7] + f[8] + f[9] + f[10];
		const float fPlus = + f[6] + f[12] + f[13] + f[15] + f[17] + f[19] + f[22] + f[24] + f[26];
		const float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;
		uy = uNormal;
		// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).
		const float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;
		const float jZerox = + f[1] - f[2] + f[7] - f[8] + f[9] - f[10];
		const float jPlusx = + f[12] - f[15] - f[19] + f[22] - f[24] + f[26];
		ux = ((2.f * uNormal + 1.f) * jZerox + 4.f * uNormal * jPlusx) / tangentDenominator;
		const float jZeroz = - f[3] + f[4] - f[7] + f[8] + f[9] - f[10];
		const float jPlusz = - f[13] + f[17] - f[19] - f[22] + f[24] + f[26];
		uz = ((2.f * uNormal + 1.f) * jZeroz + 4.f * uNormal * jPlusz) / tangentDenominator;
	}
	else if ( normalCode == 556 )
	{
		// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.
		const float fZero = + f[0] + f[1] + f[2] + f[5] + f[6] + f[11] + f[12] + f[15] + f[16];
		const float fPlus = + f[4] + f[8] + f[9] + f[14] + f[17] + f[20] + f[21] + f[24] + f[26];
		const float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;
		uz = uNormal;
		// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).
		const float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;
		const float jZerox = + f[1] - f[2] - f[11] + f[12] - f[15] + f[16];
		const float jPlusx = - f[8] + f[9] + f[20] - f[21] - f[24] + f[26];
		ux = ((2.f * uNormal + 1.f) * jZerox + 4.f * uNormal * jPlusx) / tangentDenominator;
		const float jZeroy = - f[5] + f[6] - f[11] + f[12] + f[15] - f[16];
		const float jPlusy = - f[14] + f[17] - f[20] - f[21] + f[24] + f[26];
		uy = ((2.f * uNormal + 1.f) * jZeroy + 4.f * uNormal * jPlusy) / tangentDenominator;
	}
	else if ( normalCode == 455 )
	{
		// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.
		const float fZero = + f[0] + f[3] + f[4] + f[5] + f[6] + f[13] + f[14] + f[17] + f[18];
		const float fPlus = + f[2] + f[8] + f[10] + f[11] + f[15] + f[19] + f[21] + f[24] + f[25];
		const float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;
		ux = -uNormal;
		// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).
		const float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;
		const float jZeroy = - f[5] + f[6] + f[13] - f[14] + f[17] - f[18];
		const float jPlusy = - f[11] + f[15] + f[19] - f[21] + f[24] - f[25];
		uy = ((2.f * uNormal + 1.f) * jZeroy + 4.f * uNormal * jPlusy) / tangentDenominator;
		const float jZeroz = - f[3] + f[4] - f[13] + f[14] + f[17] - f[18];
		const float jPlusz = + f[8] - f[10] - f[19] + f[21] + f[24] - f[25];
		uz = ((2.f * uNormal + 1.f) * jZeroz + 4.f * uNormal * jPlusz) / tangentDenominator;
	}
	else if ( normalCode == 545 )
	{
		// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.
		const float fZero = + f[0] + f[1] + f[2] + f[3] + f[4] + f[7] + f[8] + f[9] + f[10];
		const float fPlus = + f[5] + f[11] + f[14] + f[16] + f[18] + f[20] + f[21] + f[23] + f[25];
		const float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;
		uy = -uNormal;
		// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).
		const float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;
		const float jZerox = + f[1] - f[2] + f[7] - f[8] + f[9] - f[10];
		const float jPlusx = - f[11] + f[16] + f[20] - f[21] + f[23] - f[25];
		ux = ((2.f * uNormal + 1.f) * jZerox + 4.f * uNormal * jPlusx) / tangentDenominator;
		const float jZeroz = - f[3] + f[4] - f[7] + f[8] + f[9] - f[10];
		const float jPlusz = + f[14] - f[18] + f[20] + f[21] - f[23] - f[25];
		uz = ((2.f * uNormal + 1.f) * jZeroz + 4.f * uNormal * jPlusz) / tangentDenominator;
	}
	else if ( normalCode == 554 )
	{
		// Face restoration: rhoZ = sum(c.n=0) f + 2 sum(c.n=+1) f.
		const float fZero = + f[0] + f[1] + f[2] + f[5] + f[6] + f[11] + f[12] + f[15] + f[16];
		const float fPlus = + f[3] + f[7] + f[10] + f[13] + f[18] + f[19] + f[22] + f[23] + f[25];
		const float uNormal = (fZero + 2.f * fPlus) / rho - 1.f;
		uz = -uNormal;
		// Set k_nnt = 0 for each tangent (LBM (20), Eq. 2.52).
		const float tangentDenominator = fZero + 2.f * (uNormal + 1.f) * rho * uNormal;
		const float jZerox = + f[1] - f[2] - f[11] + f[12] - f[15] + f[16];
		const float jPlusx = + f[7] - f[10] - f[19] + f[22] + f[23] - f[25];
		ux = ((2.f * uNormal + 1.f) * jZerox + 4.f * uNormal * jPlusx) / tangentDenominator;
		const float jZeroy = - f[5] + f[6] - f[11] + f[12] + f[15] - f[16];
		const float jPlusy = + f[13] - f[18] + f[19] + f[22] - f[23] - f[25];
		uy = ((2.f * uNormal + 1.f) * jZeroy + 4.f * uNormal * jPlusy) / tangentDenominator;
	}
	else if ( normalCode == 566 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + f[1] + f[2] + (2.f) * f[4] + (2.f) * f[6] + (2.f) * f[8] + (2.f) * f[9] + (2.f) * f[12] + (2.f) * f[15] + (4.f) * f[17] + (4.f) * f[24] + (4.f) * f[26]) / rho;
		const float scmf1 = (  - f[0] - f[1] - f[2] + (-2.f) * f[4] + (-2.f) * f[8] + (-2.f) * f[9]) / rho;
		const float scmf2 = (  + f[1] - f[2] + (-2.f) * f[8] + (2.f) * f[9] + (2.f) * f[12] + (-2.f) * f[15] + (-4.f) * f[24] + (4.f) * f[26]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s2;
		uy = + s0 + (3.f/2.f) * s1;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 656 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[1] + (2.f) * f[4] + f[5] + f[6] + (4.f) * f[9] + (2.f) * f[12] + (2.f) * f[14] + (2.f) * f[16] + (2.f) * f[17] + (4.f) * f[20] + (4.f) * f[26]) / rho;
		const float scmf1 = (  - f[0] + (-2.f) * f[4] - f[5] - f[6] + (-2.f) * f[14] + (-2.f) * f[17]) / rho;
		const float scmf2 = (  - f[5] + f[6] + (2.f) * f[12] + (-2.f) * f[14] + (-2.f) * f[16] + (2.f) * f[17] + (-4.f) * f[20] + (4.f) * f[26]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (3.f/2.f) * s1;
		uy = + s2;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 665 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[1] + f[3] + f[4] + (2.f) * f[6] + (2.f) * f[7] + (2.f) * f[9] + (4.f) * f[12] + (2.f) * f[13] + (2.f) * f[17] + (4.f) * f[22] + (4.f) * f[26]) / rho;
		const float scmf1 = (  - f[0] - f[3] - f[4] + (-2.f) * f[6] + (-2.f) * f[13] + (-2.f) * f[17]) / rho;
		const float scmf2 = (  - f[3] + f[4] + (-2.f) * f[7] + (2.f) * f[9] + (-2.f) * f[13] + (2.f) * f[17] + (-4.f) * f[22] + (4.f) * f[26]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (3.f/2.f) * s1;
		uy = + (-3.f/2.f) * s1;
		uz = + s2;
	}
	else if ( normalCode == 544 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + f[1] + f[2] + (2.f) * f[3] + (2.f) * f[5] + (2.f) * f[7] + (2.f) * f[10] + (2.f) * f[11] + (2.f) * f[16] + (4.f) * f[18] + (4.f) * f[23] + (4.f) * f[25]) / rho;
		const float scmf1 = (  + f[0] + f[1] + f[2] + (2.f) * f[3] + (2.f) * f[7] + (2.f) * f[10]) / rho;
		const float scmf2 = (  + f[1] - f[2] + (2.f) * f[7] + (-2.f) * f[10] + (-2.f) * f[11] + (2.f) * f[16] + (4.f) * f[23] + (-4.f) * f[25]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s2;
		uy = - s0 + (3.f/2.f) * s1;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 454 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[2] + (2.f) * f[3] + f[5] + f[6] + (4.f) * f[10] + (2.f) * f[11] + (2.f) * f[13] + (2.f) * f[15] + (2.f) * f[18] + (4.f) * f[19] + (4.f) * f[25]) / rho;
		const float scmf1 = (  + f[0] + (2.f) * f[3] + f[5] + f[6] + (2.f) * f[13] + (2.f) * f[18]) / rho;
		const float scmf2 = (  - f[5] + f[6] + (-2.f) * f[11] + (2.f) * f[13] + (2.f) * f[15] + (-2.f) * f[18] + (4.f) * f[19] + (-4.f) * f[25]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (3.f/2.f) * s1;
		uy = + s2;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 445 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[2] + f[3] + f[4] + (2.f) * f[5] + (2.f) * f[8] + (2.f) * f[10] + (4.f) * f[11] + (2.f) * f[14] + (2.f) * f[18] + (4.f) * f[21] + (4.f) * f[25]) / rho;
		const float scmf1 = (  + f[0] + f[3] + f[4] + (2.f) * f[5] + (2.f) * f[14] + (2.f) * f[18]) / rho;
		const float scmf2 = (  - f[3] + f[4] + (2.f) * f[8] + (-2.f) * f[10] + (2.f) * f[14] + (-2.f) * f[18] + (4.f) * f[21] + (-4.f) * f[25]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (3.f/2.f) * s1;
		uy = + (-3.f/2.f) * s1;
		uz = + s2;
	}
	else if ( normalCode == 564 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] - f[1] - f[2] + (-2.f) * f[3] + (-2.f) * f[6] + (-2.f) * f[7] + (-2.f) * f[10] + (-2.f) * f[12] + (-4.f) * f[13] + (-2.f) * f[15] + (-4.f) * f[19] + (-4.f) * f[22]) / rho;
		const float scmf1 = (  + f[0] + f[1] + f[2] + (2.f) * f[3] + (2.f) * f[7] + (2.f) * f[10]) / rho;
		const float scmf2 = (  - f[1] + f[2] + (-2.f) * f[7] + (2.f) * f[10] + (-2.f) * f[12] + (2.f) * f[15] + (4.f) * f[19] + (-4.f) * f[22]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s2;
		uy = - s0 + (-3.f/2.f) * s1;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 654 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[1] + (-2.f) * f[3] - f[5] - f[6] + (-4.f) * f[7] + (-2.f) * f[12] + (-2.f) * f[13] + (-2.f) * f[16] + (-2.f) * f[18] + (-4.f) * f[22] + (-4.f) * f[23]) / rho;
		const float scmf1 = (  + f[0] + (2.f) * f[3] + f[5] + f[6] + (2.f) * f[13] + (2.f) * f[18]) / rho;
		const float scmf2 = (  + f[5] - f[6] + (-2.f) * f[12] + (-2.f) * f[13] + (2.f) * f[16] + (2.f) * f[18] + (-4.f) * f[22] + (4.f) * f[23]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (-3.f/2.f) * s1;
		uy = - s2;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 645 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[1] - f[3] - f[4] + (-2.f) * f[5] + (-2.f) * f[7] + (-2.f) * f[9] + (-2.f) * f[14] + (-4.f) * f[16] + (-2.f) * f[18] + (-4.f) * f[20] + (-4.f) * f[23]) / rho;
		const float scmf1 = (  + f[0] + f[3] + f[4] + (2.f) * f[5] + (2.f) * f[14] + (2.f) * f[18]) / rho;
		const float scmf2 = (  + f[3] - f[4] + (2.f) * f[7] + (-2.f) * f[9] + (-2.f) * f[14] + (2.f) * f[18] + (-4.f) * f[20] + (4.f) * f[23]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (-3.f/2.f) * s1;
		uy = + (-3.f/2.f) * s1;
		uz = - s2;
	}
	else if ( normalCode == 546 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] - f[1] - f[2] + (-2.f) * f[4] + (-2.f) * f[5] + (-2.f) * f[8] + (-2.f) * f[9] + (-2.f) * f[11] + (-4.f) * f[14] + (-2.f) * f[16] + (-4.f) * f[20] + (-4.f) * f[21]) / rho;
		const float scmf1 = (  - f[0] - f[1] - f[2] + (-2.f) * f[4] + (-2.f) * f[8] + (-2.f) * f[9]) / rho;
		const float scmf2 = (  - f[1] + f[2] + (2.f) * f[8] + (-2.f) * f[9] + (2.f) * f[11] + (-2.f) * f[16] + (-4.f) * f[20] + (4.f) * f[21]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s2;
		uy = + s0 + (-3.f/2.f) * s1;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 456 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[2] + (-2.f) * f[4] - f[5] - f[6] + (-4.f) * f[8] + (-2.f) * f[11] + (-2.f) * f[14] + (-2.f) * f[15] + (-2.f) * f[17] + (-4.f) * f[21] + (-4.f) * f[24]) / rho;
		const float scmf1 = (  - f[0] + (-2.f) * f[4] - f[5] - f[6] + (-2.f) * f[14] + (-2.f) * f[17]) / rho;
		const float scmf2 = (  + f[5] - f[6] + (2.f) * f[11] + (2.f) * f[14] + (-2.f) * f[15] + (-2.f) * f[17] + (4.f) * f[21] + (-4.f) * f[24]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (-3.f/2.f) * s1;
		uy = - s2;
		uz = + (-3.f/2.f) * s1;
	}
	else if ( normalCode == 465 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[2] - f[3] - f[4] + (-2.f) * f[6] + (-2.f) * f[8] + (-2.f) * f[10] + (-2.f) * f[13] + (-4.f) * f[15] + (-2.f) * f[17] + (-4.f) * f[19] + (-4.f) * f[24]) / rho;
		const float scmf1 = (  - f[0] - f[3] - f[4] + (-2.f) * f[6] + (-2.f) * f[13] + (-2.f) * f[17]) / rho;
		const float scmf2 = (  + f[3] - f[4] + (-2.f) * f[8] + (2.f) * f[10] + (2.f) * f[13] + (-2.f) * f[17] + (4.f) * f[19] + (-4.f) * f[24]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (0.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (-3.f/2.f) * s1;
		uy = + (-3.f/2.f) * s1;
		uz = - s2;
	}
	else if ( normalCode == 666 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[1] + (2.f) * f[4] + (2.f) * f[6] + (4.f) * f[9] + (4.f) * f[12] + (4.f) * f[17] + (8.f) * f[26]) / rho;
		const float scmf1 = (  - f[0] + (-2.f) * f[4] + (-2.f) * f[6] + (-4.f) * f[17]) / rho;
		const float scmf2 = (  - f[0] + (-2.f) * f[1] + (-2.f) * f[4] + (-4.f) * f[9]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (-2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (3.f/2.f) * s1;
		uy = + s0 + (3.f/2.f) * s2;
		uz = - s0 + (-3.f/2.f) * s1 + (-3.f/2.f) * s2;
	}
	else if ( normalCode == 444 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[2] + (-2.f) * f[3] + (-2.f) * f[5] + (-4.f) * f[10] + (-4.f) * f[11] + (-4.f) * f[18] + (-8.f) * f[25]) / rho;
		const float scmf1 = (  - f[0] + (-2.f) * f[3] + (-2.f) * f[5] + (-4.f) * f[18]) / rho;
		const float scmf2 = (  - f[0] + (-2.f) * f[2] + (-2.f) * f[3] + (-4.f) * f[10]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (-2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (-3.f/2.f) * s1;
		uy = + s0 + (-3.f/2.f) * s2;
		uz = - s0 + (3.f/2.f) * s1 + (3.f/2.f) * s2;
	}
	else if ( normalCode == 466 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[2] + (-2.f) * f[4] + (-2.f) * f[6] + (-4.f) * f[8] + (-4.f) * f[15] + (-4.f) * f[17] + (-8.f) * f[24]) / rho;
		const float scmf1 = (  - f[0] + (-2.f) * f[4] + (-2.f) * f[6] + (-4.f) * f[17]) / rho;
		const float scmf2 = (  + f[0] + (2.f) * f[2] + (2.f) * f[4] + (4.f) * f[8]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (-3.f/2.f) * s1;
		uy = - s0 + (-3.f/2.f) * s2;
		uz = + s0 + (-3.f/2.f) * s1 + (3.f/2.f) * s2;
	}
	else if ( normalCode == 646 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[1] + (-2.f) * f[4] + (-2.f) * f[5] + (-4.f) * f[9] + (-4.f) * f[14] + (-4.f) * f[16] + (-8.f) * f[20]) / rho;
		const float scmf1 = (  + f[0] + (2.f) * f[4] + (2.f) * f[5] + (4.f) * f[14]) / rho;
		const float scmf2 = (  - f[0] + (-2.f) * f[1] + (-2.f) * f[4] + (-4.f) * f[9]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (-2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (-3.f/2.f) * s1;
		uy = + s0 + (-3.f/2.f) * s2;
		uz = + s0 + (3.f/2.f) * s1 + (-3.f/2.f) * s2;
	}
	else if ( normalCode == 664 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  - f[0] + (-2.f) * f[1] + (-2.f) * f[3] + (-2.f) * f[6] + (-4.f) * f[7] + (-4.f) * f[12] + (-4.f) * f[13] + (-8.f) * f[22]) / rho;
		const float scmf1 = (  + f[0] + (2.f) * f[3] + (2.f) * f[6] + (4.f) * f[13]) / rho;
		const float scmf2 = (  + f[0] + (2.f) * f[1] + (2.f) * f[3] + (4.f) * f[7]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (-1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (-3.f/2.f) * s1;
		uy = - s0 + (-3.f/2.f) * s2;
		uz = - s0 + (-3.f/2.f) * s1 + (-3.f/2.f) * s2;
	}
	else if ( normalCode == 644 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[1] + (2.f) * f[3] + (2.f) * f[5] + (4.f) * f[7] + (4.f) * f[16] + (4.f) * f[18] + (8.f) * f[23]) / rho;
		const float scmf1 = (  - f[0] + (-2.f) * f[3] + (-2.f) * f[5] + (-4.f) * f[18]) / rho;
		const float scmf2 = (  + f[0] + (2.f) * f[1] + (2.f) * f[3] + (4.f) * f[7]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (-2.f/3.f);
		const float s2 = scmf2 - (2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = + s0 + (3.f/2.f) * s1;
		uy = - s0 + (3.f/2.f) * s2;
		uz = + s0 + (3.f/2.f) * s1 + (-3.f/2.f) * s2;
	}
	else if ( normalCode == 464 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[2] + (2.f) * f[3] + (2.f) * f[6] + (4.f) * f[10] + (4.f) * f[13] + (4.f) * f[15] + (8.f) * f[19]) / rho;
		const float scmf1 = (  + f[0] + (2.f) * f[3] + (2.f) * f[6] + (4.f) * f[13]) / rho;
		const float scmf2 = (  - f[0] + (-2.f) * f[2] + (-2.f) * f[3] + (-4.f) * f[10]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (-2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (3.f/2.f) * s1;
		uy = + s0 + (3.f/2.f) * s2;
		uz = + s0 + (-3.f/2.f) * s1 + (3.f/2.f) * s2;
	}
	else if ( normalCode == 446 )
	{
		// Multiply SCMfk fk
		const float scmf0 = (  + f[0] + (2.f) * f[2] + (2.f) * f[4] + (2.f) * f[5] + (4.f) * f[8] + (4.f) * f[11] + (4.f) * f[14] + (8.f) * f[21]) / rho;
		const float scmf1 = (  + f[0] + (2.f) * f[4] + (2.f) * f[5] + (4.f) * f[14]) / rho;
		const float scmf2 = (  + f[0] + (2.f) * f[2] + (2.f) * f[4] + (4.f) * f[8]) / rho;
		// Subtract 1/rho scmf - scq
		const float s0 = scmf0 - (1.f);
		const float s1 = scmf1 - (2.f/3.f);
		const float s2 = scmf2 - (2.f/3.f);
		// Multiply (Su C^T Qu)^-1 * s to get u
		ux = - s0 + (3.f/2.f) * s1;
		uy = - s0 + (3.f/2.f) * s2;
		uz = - s0 + (3.f/2.f) * s1 + (3.f/2.f) * s2;
	}
}
