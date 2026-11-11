// LOCAL ITERATIVE INTERFACE VERSION

#pragma once

#include "./types.h"
#include "./esotwistStreamingFunctions.h"
#include "./cellFunctions.h"
#include "./NBRFunctions.h"

// Partial-population interface for 2:1 refinement.
// Mask bit q == 1 means that population q must be supplied by the interface.
// The current builder masks are intended for fluid-fluid interfaces away from
// walls and domain boundaries. Decoding a stencil fallback does not establish
// the validity of its donor populations near a boundary.

namespace InterfaceDetail
{
    // Fixed work per target; no convergence check in the GPU kernels.
    constexpr int iterationCount = 16;

    __host__ __device__ inline void getChildren(
        const NBRStruct &NBR, int (&children)[8] )
    {
        // Child index = iAdd + 2*jAdd + 4*kAdd, as in fillInterface.
        children[0] = NBR.self;
        children[1] = NBR.iPlus;
        children[2] = NBR.jPlus;
        children[3] = NBR.ijPlus;
        children[4] = NBR.kPlus;
        children[5] = NBR.ikPlus;
        children[6] = NBR.jkPlus;
        children[7] = NBR.ijkPlus;
    }

    __host__ __device__ inline float readPopulation(
        const FloatConstView2DType &fView, const NBRStruct &NBR,
        const bool esotwistFlipper, const int q )
    {
        int cellIndex, fIndex;
        getPreCollisionIndexSingle( cellIndex, fIndex, NBR, esotwistFlipper, q );
        return fView( fIndex, cellIndex );
    }

    __host__ __device__ inline void reconstruct(
        float (&f)[27], const float (&source)[27], const uint32_t mask,
        const float gamma, const float alpha )
    {
        // Only requested entries of source need to be initialized.
        for ( int q = 0; q < 27; ++q )
            if ( mask & (uint32_t{1} << q) ) f[q] = source[q];

        for ( int iteration = 0; iteration < iterationCount; ++iteration )
        {
            float dRho, ux, uy, uz;
            getDRhoUxUyUz( dRho, ux, uy, uz, f );
            float feq[27];
            getFeq( dRho, ux, uy, uz, feq );

            for ( int q = 0; q < 27; ++q )
            {
                if ( !(mask & (uint32_t{1} << q)) ) continue;
                // f* = (1-gamma)*feq(f) + gamma*source.
                // Work entirely with shifted populations: f = f_physical - w.
                const float candidate = feq[q] + gamma * (source[q] - feq[q]);
                f[q] += alpha * (candidate - f[q]);
            }
        }
    }

    __host__ __device__ inline void updateTarget(
        FloatArray2DType::ViewType &fView, const NBRStruct &NBR,
        const bool esotwistFlipper, const uint32_t mask,
        const float (&source)[27], const float gamma, const float alpha )
    {
        if ( mask == 0 ) return;

        int cellIndex[27], fIndex[27];
        getPreCollisionIndex( cellIndex, fIndex, NBR, esotwistFlipper );

        float f[27];
        for ( int q = 0; q < 27; ++q )
        {
            // Do not read the old contents of an invalid population.
            if ( !(mask & (uint32_t{1} << q)) )
                f[q] = fView( fIndex[q], cellIndex[q] );
        }

        reconstruct( f, source, mask, gamma, alpha );

        // Known populations are never written, even with their original value.
        for ( int q = 0; q < 27; ++q )
            if ( mask & (uint32_t{1} << q) )
                fView( fIndex[q], cellIndex[q] ) = f[q];
    }
}

inline void updateFineToCoarseInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
    const int interfaceCount = GridCoarse.Interface.interfaceCount;
    if ( interfaceCount == 0 ) return;

    const InfoStruct InfoCoarse = GridCoarse.Info;
    const InfoStruct InfoFine = GridFine.Info;
    auto fViewCoarse = GridCoarse.fArray.getView();
    auto fViewFine = GridFine.fArray.getConstView();
    auto shifterViewCoarse = GridCoarse.IJKNBR.shifterArray.getConstView();
    auto jPlusViewCoarse = GridCoarse.IJKNBR.jPlusArray.getConstView();
    auto kPlusViewCoarse = GridCoarse.IJKNBR.kPlusArray.getConstView();
    auto jkPlusViewCoarse = GridCoarse.IJKNBR.jkPlusArray.getConstView();
    auto shifterViewFine = GridFine.IJKNBR.shifterArray.getConstView();
    auto jPlusViewFine = GridFine.IJKNBR.jPlusArray.getConstView();
    auto kPlusViewFine = GridFine.IJKNBR.kPlusArray.getConstView();
    auto jkPlusViewFine = GridFine.IJKNBR.jkPlusArray.getConstView();
    auto indexView = GridCoarse.Interface.indexArray.getConstView();
    auto childMapView = GridCoarse.Interface.childMapArray.getConstView();
    auto maskView = GridCoarse.Interface.coarseTargetMaskArray.getConstView();

    // Pre-collision nonequilibrium scaling, with dtCoarse / dtFine = 2.
    const float tauCoarse = 0.5f + 3.f * InfoCoarse.nu;
    const float tauFine = 0.5f + 3.f * InfoFine.nu;
    const float gamma = 2.f * tauCoarse / tauFine;
    const float alpha = 2.f / (1.f + gamma);

    auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
    {
        const uint32_t mask = maskView( index );
        if ( mask == 0 ) return;

        NBRStruct NBRCoarse, NBRFine0;
        getCompressedNBR( indexView(index), NBRCoarse, shifterViewCoarse,
                          jPlusViewCoarse, kPlusViewCoarse, jkPlusViewCoarse, InfoCoarse );
        getCompressedNBR( childMapView(index), NBRFine0, shifterViewFine,
                          jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );

        int children[8];
        InterfaceDetail::getChildren( NBRFine0, children );
        NBRStruct NBRChildren[8];
        for ( int child = 0; child < 8; ++child )
            getCompressedNBR( children[child], NBRChildren[child], shifterViewFine,
                              jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );

        float source[27];
        for ( int q = 0; q < 27; ++q )
        {
            if ( !(mask & (uint32_t{1} << q)) ) continue;
            float sum = 0.f;
            for ( int child = 0; child < 8; ++child )
                sum += InterfaceDetail::readPopulation(
                    fViewFine, NBRChildren[child], InfoFine.esotwistFlipper, q );
            source[q] = 0.125f * sum;
        }

        InterfaceDetail::updateTarget( fViewCoarse, NBRCoarse,
            InfoCoarse.esotwistFlipper, mask, source, gamma, alpha );
    };
    TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, interfaceCount, cellLambda );
}

inline void updateCoarseToFineInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
    const int interfaceCount = GridCoarse.Interface.interfaceCount;
    if ( interfaceCount == 0 ) return;

    const InfoStruct InfoCoarse = GridCoarse.Info;
    const InfoStruct InfoFine = GridFine.Info;
    auto fViewCoarse = GridCoarse.fArray.getConstView();
    auto fViewFine = GridFine.fArray.getView();
    auto shifterViewCoarse = GridCoarse.IJKNBR.shifterArray.getConstView();
    auto jPlusViewCoarse = GridCoarse.IJKNBR.jPlusArray.getConstView();
    auto kPlusViewCoarse = GridCoarse.IJKNBR.kPlusArray.getConstView();
    auto jkPlusViewCoarse = GridCoarse.IJKNBR.jkPlusArray.getConstView();
    auto shifterViewFine = GridFine.IJKNBR.shifterArray.getConstView();
    auto jPlusViewFine = GridFine.IJKNBR.jPlusArray.getConstView();
    auto kPlusViewFine = GridFine.IJKNBR.kPlusArray.getConstView();
    auto jkPlusViewFine = GridFine.IJKNBR.jkPlusArray.getConstView();
    auto indexView = GridCoarse.Interface.indexArray.getConstView();
    auto childMapView = GridCoarse.Interface.childMapArray.getConstView();
    auto maskView = GridCoarse.Interface.fineTargetMaskArray.getConstView();
    auto iMinusView = GridCoarse.Interface.iMinusStencilArray.getConstView();
    auto iPlusView = GridCoarse.Interface.iPlusStencilArray.getConstView();
    auto jMinusView = GridCoarse.Interface.jMinusStencilArray.getConstView();
    auto jPlusView = GridCoarse.Interface.jPlusStencilArray.getConstView();
    auto kMinusView = GridCoarse.Interface.kMinusStencilArray.getConstView();
    auto kPlusView = GridCoarse.Interface.kPlusStencilArray.getConstView();

    // Reciprocal of the fine-to-coarse factor.
    const float tauCoarse = 0.5f + 3.f * InfoCoarse.nu;
    const float tauFine = 0.5f + 3.f * InfoFine.nu;
    const float gamma = tauFine / (2.f * tauCoarse);
    const float alpha = 2.f / (1.f + gamma);

    auto cellLambda = [=] __cuda_callable__ ( const int index ) mutable
    {
        NBRStruct NBRCoarse, NBRFine0;
        getCompressedNBR( indexView(index), NBRCoarse, shifterViewCoarse,
                          jPlusViewCoarse, kPlusViewCoarse, jkPlusViewCoarse, InfoCoarse );
        getCompressedNBR( childMapView(index), NBRFine0, shifterViewFine,
                          jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );

        int children[8];
        InterfaceDetail::getChildren( NBRFine0, children );

        // Nonnegative: direct donor (including the centre for constant fallback).
        // Negative: opposite donor encoded as -cell-1; use 2*centre-opposite.
        const int stencil[6] = {
            iMinusView(index), iPlusView(index),
            jMinusView(index), jPlusView(index),
            kMinusView(index), kPlusView(index)
        };
        NBRStruct NBRStencil[6];
        for ( int face = 0; face < 6; ++face )
        {
            const int cell = stencil[face] >= 0 ? stencil[face] : -stencil[face] - 1;
            getCompressedNBR( cell, NBRStencil[face], shifterViewCoarse,
                              jPlusViewCoarse, kPlusViewCoarse, jkPlusViewCoarse, InfoCoarse );
        }

        for ( int child = 0; child < 8; ++child )
        {
            const uint32_t mask = maskView( child, index );
            if ( mask == 0 ) continue;

            const int faces[3] = {
                child & 1, 2 + ((child >> 1) & 1), 4 + ((child >> 2) & 1)
            };
            float source[27];
            for ( int q = 0; q < 27; ++q )
            {
                if ( !(mask & (uint32_t{1} << q)) ) continue;
                const float centre = InterfaceDetail::readPopulation(
                    fViewCoarse, NBRCoarse, InfoCoarse.esotwistFlipper, q );
                float sum = centre;
                for ( int axis = 0; axis < 3; ++axis )
                {
                    const int face = faces[axis];
                    const float value = InterfaceDetail::readPopulation(
                        fViewCoarse, NBRStencil[face], InfoCoarse.esotwistFlipper, q );
                    sum += stencil[face] >= 0 ? value : 2.f * centre - value;
                }
                source[q] = 0.25f * sum;
            }

            NBRStruct NBRFine;
            getCompressedNBR( children[child], NBRFine, shifterViewFine,
                              jPlusViewFine, kPlusViewFine, jkPlusViewFine, InfoFine );
            InterfaceDetail::updateTarget( fViewFine, NBRFine,
                InfoFine.esotwistFlipper, mask, source, gamma, alpha );
        }
    };
    TNL::Algorithms::parallelFor<TNL::Devices::Cuda>( 0, interfaceCount, cellLambda );
}

inline void updateInterface( GridStruct &GridCoarse, GridStruct &GridFine )
{
    updateFineToCoarseInterface( GridCoarse, GridFine );
    updateCoarseToFineInterface( GridCoarse, GridFine );
}
