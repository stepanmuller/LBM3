#pragma once

// Esotwist streaming step: Just flip the EsotwistFlipper to alter between odd / even iterations
void applyStreaming( GridStruct& Grid )
{
	Grid.esotwistFlipper = !Grid.esotwistFlipper;
}

// Implementation of the esotwist logic:
// for physical distribution function XYZ, 
// cellIndex informs from which cell this XYZ distribution should be loaded,
// fIndex informs from which memory position in that cell should it be loaded
// note that the opposing distributions swap memory positions each iteration

__cuda_callable__ void getPreCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NBRStruct &NBR, const bool &esotwistFlipper, const InfoStruct &Info )
{
    cellIndex[OOO] = NBR.self;
    cellIndex[POO] = NBR.self;
    cellIndex[MOO] = NBR.iPlus;
    cellIndex[OOM] = NBR.kPlus;
    cellIndex[OOP] = NBR.self;
    cellIndex[OMO] = NBR.jPlus;
    cellIndex[OPO] = NBR.self;
    cellIndex[POM] = NBR.kPlus;
    cellIndex[MOP] = NBR.iPlus;
    cellIndex[POP] = NBR.self;
    cellIndex[MOM] = NBR.ikPlus;
    cellIndex[MMO] = NBR.ijPlus;
    cellIndex[PPO] = NBR.self;
    cellIndex[OPM] = NBR.kPlus;
    cellIndex[OMP] = NBR.jPlus;
    cellIndex[MPO] = NBR.iPlus;
    cellIndex[PMO] = NBR.jPlus;
    cellIndex[OPP] = NBR.self;
    cellIndex[OMM] = NBR.jkPlus;
    cellIndex[MPM] = NBR.ikPlus;
    cellIndex[PMP] = NBR.jPlus;
    cellIndex[MMP] = NBR.ijPlus;
    cellIndex[PPM] = NBR.kPlus;
    cellIndex[PMM] = NBR.jkPlus;
    cellIndex[MPP] = NBR.iPlus;
    cellIndex[MMM] = NBR.ijkPlus;
    cellIndex[PPP] = NBR.self;

    if ( !esotwistFlipper )
    {
        fIndex[OOO] = OOO;  fIndex[POO] = POO;  fIndex[MOO] = MOO;
        fIndex[OOM] = OOM;  fIndex[OOP] = OOP;  fIndex[OMO] = OMO;
        fIndex[OPO] = OPO;  fIndex[POM] = POM;  fIndex[MOP] = MOP;
        fIndex[POP] = POP;  fIndex[MOM] = MOM;  fIndex[MMO] = MMO;
        fIndex[PPO] = PPO;  fIndex[OPM] = OPM;  fIndex[OMP] = OMP;
        fIndex[MPO] = MPO;  fIndex[PMO] = PMO;  fIndex[OPP] = OPP;
        fIndex[OMM] = OMM;  fIndex[MPM] = MPM;  fIndex[PMP] = PMP;
        fIndex[MMP] = MMP;  fIndex[PPM] = PPM;  fIndex[PMM] = PMM;
        fIndex[MPP] = MPP;  fIndex[MMM] = MMM;  fIndex[PPP] = PPP;
    }
    else
    {
        fIndex[OOO] = OOO;  fIndex[POO] = MOO;  fIndex[MOO] = POO;
        fIndex[OOM] = OOP;  fIndex[OOP] = OOM;  fIndex[OMO] = OPO;
        fIndex[OPO] = OMO;  fIndex[POM] = MOP;  fIndex[MOP] = POM;
        fIndex[POP] = MOM;  fIndex[MOM] = POP;  fIndex[MMO] = PPO;
        fIndex[PPO] = MMO;  fIndex[OPM] = OMP;  fIndex[OMP] = OPM;
        fIndex[MPO] = PMO;  fIndex[PMO] = MPO;  fIndex[OPP] = OMM;
        fIndex[OMM] = OPP;  fIndex[MPM] = PMP;  fIndex[PMP] = MPM;
        fIndex[MMP] = PPM;  fIndex[PPM] = MMP;  fIndex[PMM] = MPP;
        fIndex[MPP] = PMM;  fIndex[MMM] = PPP;  fIndex[PPP] = MMM;
    }
}

__cuda_callable__ void getPostCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NBRStruct &NBR, const bool &esotwistFlipper, const InfoStruct &Info )
{
    cellIndex[OOO] = NBR.self;
    cellIndex[POO] = NBR.iPlus;
    cellIndex[MOO] = NBR.self;
    cellIndex[OOM] = NBR.self;
    cellIndex[OOP] = NBR.kPlus;
    cellIndex[OMO] = NBR.self;
    cellIndex[OPO] = NBR.jPlus;
    cellIndex[POM] = NBR.iPlus;
    cellIndex[MOP] = NBR.kPlus;
    cellIndex[POP] = NBR.ikPlus;
    cellIndex[MOM] = NBR.self;
    cellIndex[MMO] = NBR.self;
    cellIndex[PPO] = NBR.ijPlus;
    cellIndex[OPM] = NBR.jPlus;
    cellIndex[OMP] = NBR.kPlus;
    cellIndex[MPO] = NBR.jPlus;
    cellIndex[PMO] = NBR.iPlus;
    cellIndex[OPP] = NBR.jkPlus;
    cellIndex[OMM] = NBR.self;
    cellIndex[MPM] = NBR.jPlus;
    cellIndex[PMP] = NBR.ikPlus;
    cellIndex[MMP] = NBR.kPlus;
    cellIndex[PPM] = NBR.ijPlus;
    cellIndex[PMM] = NBR.iPlus;
    cellIndex[MPP] = NBR.jkPlus;
    cellIndex[MMM] = NBR.self;
    cellIndex[PPP] = NBR.ijkPlus;

    if ( !esotwistFlipper )
    {
        fIndex[OOO] = OOO;  fIndex[POO] = MOO;  fIndex[MOO] = POO;
        fIndex[OOM] = OOP;  fIndex[OOP] = OOM;  fIndex[OMO] = OPO;
        fIndex[OPO] = OMO;  fIndex[POM] = MOP;  fIndex[MOP] = POM;
        fIndex[POP] = MOM;  fIndex[MOM] = POP;  fIndex[MMO] = PPO;
        fIndex[PPO] = MMO;  fIndex[OPM] = OMP;  fIndex[OMP] = OPM;
        fIndex[MPO] = PMO;  fIndex[PMO] = MPO;  fIndex[OPP] = OMM;
        fIndex[OMM] = OPP;  fIndex[MPM] = PMP;  fIndex[PMP] = MPM;
        fIndex[MMP] = PPM;  fIndex[PPM] = MMP;  fIndex[PMM] = MPP;
        fIndex[MPP] = PMM;  fIndex[MMM] = PPP;  fIndex[PPP] = MMM;
    }
    else
    {
        fIndex[OOO] = OOO;  fIndex[POO] = POO;  fIndex[MOO] = MOO;
        fIndex[OOM] = OOM;  fIndex[OOP] = OOP;  fIndex[OMO] = OMO;
        fIndex[OPO] = OPO;  fIndex[POM] = POM;  fIndex[MOP] = MOP;
        fIndex[POP] = POP;  fIndex[MOM] = MOM;  fIndex[MMO] = MMO;
        fIndex[PPO] = PPO;  fIndex[OPM] = OPM;  fIndex[OMP] = OMP;
        fIndex[MPO] = MPO;  fIndex[PMO] = PMO;  fIndex[OPP] = OPP;
        fIndex[OMM] = OMM;  fIndex[MPM] = MPM;  fIndex[PMP] = PMP;
        fIndex[MMP] = MMP;  fIndex[PPM] = PPM;  fIndex[PMM] = PMM;
        fIndex[MPP] = MPP;  fIndex[MMM] = MMM;  fIndex[PPP] = PPP;
    }
}

__cuda_callable__ void getFuturePreCollisionIndex( int (&cellIndex)[27], int (&fIndex)[27], const NBRStruct &NBR, const bool &esotwistFlipper, const InfoStruct &Info ) 
{ 
	const bool futureEsotwistFlipper = !esotwistFlipper;
	getPreCollisionIndex( cellIndex, fIndex, NBR, futureEsotwistFlipper, Info ); 
}
