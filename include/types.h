#pragma once

#include <iostream>
#include <sstream>
#include <cmath>
#include <fstream> 
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <climits>
#include <string>
#include <vector>
#include <algorithm>
#include <new>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Timer.h>

//------------------------------------------------------------------------------------
//--------------------------- ARRAYS, VECTORS  ---------------------------------------
//------------------------------------------------------------------------------------

using BoolArrayType = TNL::Containers::Vector< bool, TNL::Devices::Cuda, size_t >;
using BoolArrayTypeCPU = TNL::Containers::Vector< bool, TNL::Devices::Host, size_t >;
												
using IntArrayType = TNL::Containers::Vector< int, TNL::Devices::Cuda, size_t >;
using IntConstViewType = TNL::Containers::VectorView< const int, TNL::Devices::Cuda, size_t >;
using IntArrayTypeCPU = TNL::Containers::Vector< int, TNL::Devices::Host, size_t >;

using LongLongArrayType = TNL::Containers::Vector< long long, TNL::Devices::Cuda, size_t >;

using IntArray2DType = TNL::Containers::NDArray< int, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Cuda >;
using IntArray2DTypeCPU = TNL::Containers::NDArray< int, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Host >;
												
using IntArray3DType = TNL::Containers::NDArray< int, 
												TNL::Containers::SizesHolder< size_t, 0, 0, 0 >,
												std::index_sequence< 0, 1, 2 >,
												TNL::Devices::Cuda >;

using FloatArrayType = TNL::Containers::Vector< float, TNL::Devices::Cuda, size_t >;
using FloatArrayTypeCPU = TNL::Containers::Vector< float, TNL::Devices::Host, size_t >;

using BoolArray2DType = TNL::Containers::NDArray< bool, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Cuda >;

using FloatArray2DType = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Cuda >;
using FloatArray2DTypeCPU = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0 >,
												std::index_sequence< 0, 1 >,
												TNL::Devices::Host >;
												
using FloatArray3DType = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0, 0 >,
												std::index_sequence< 0, 1, 2 >,
												TNL::Devices::Cuda >;
using FloatArray3DTypeCPU = TNL::Containers::NDArray< float, 
												TNL::Containers::SizesHolder< size_t, 0, 0, 0 >,
												std::index_sequence< 0, 1, 2 >,
												TNL::Devices::Host >;
												
using Uint8_tArrayType = TNL::Containers::Vector< uint8_t, TNL::Devices::Cuda, size_t >;

using Uint4ArrayType = TNL::Containers::Vector< uint4, TNL::Devices::Cuda, size_t >;

using IntPairType = TNL::Containers::StaticArray< 2, int >;											
using IntTripleType = TNL::Containers::StaticArray< 3, int >;

//------------------------------------------------------------------------------------
//--------------------------------- STRUCTS  -----------------------------------------
//------------------------------------------------------------------------------------

struct BoundsStruct { float xMin = 0.f; float yMin = 0.f; float zMin = 0.f; float xMax = 0.f; float yMax = 0.f; float zMax = 0.f; 
						float rxMax = 0.f; float ryMax = 0.f; float rzMax = 0.f; }; 

struct InfoStruct { float gridID = 0; unsigned long long gridMemoryBytes = 0LL; int iterationsFinished = 0;
					float res = 1.f; float ox = 0.f; float oy = 0.f; float oz = 0.f; 
					BoundsStruct Bounds;
					float nu = 1.f; float dtPhys = 1.f; 
					int cellCountX = 0; int cellCountY = 0; int cellCountZ = 0; 
					int cellCount = 0; 
					bool esotwistFlipper = 0; 
					float iRegulatorInlet = 0.f; float iRegulatorOutlet = 0.f; };

struct MarkerStruct { 	bool fluid = 0; bool bounceback = 0; bool movingBounceback = 0; bool forcedVelocity = 0;
						bool BCRho = 0; bool BCU = 0; bool nonReflectiveOutlet = 0; bool nonReflectiveInlet = 0;
						bool refinement = 0; bool deepRefinement = 0; bool fineToCoarse = 0; };
						
struct BCStruct { float rho = 1.f; float ux = 0.f; float uy = 0.f; float uz = 0.f; float gx = 0.f; float gy = 0.f; float gz = 0.f; float nuMultiplier = 1.f; float collisionLimiter = 0.01f; };
					
// IJK holds cell indexes on X, Y, Z axes within the Grid that owns it
struct IJKArrayStructCPU; // just declaring first
struct IJKArrayStruct { IntArrayType iArray; IntArrayType jArray; IntArrayType kArray; 
						IJKArrayStruct() = default;
						// Constructor copies data from IJKArrayStructCPU
						IJKArrayStruct( const IJKArrayStructCPU& IJKCPU );
					};

struct IJKArrayStructCPU { 	IntArrayTypeCPU iArray; IntArrayTypeCPU jArray; IntArrayTypeCPU kArray; 
							IJKArrayStructCPU() = default;
							// Constructor copies data from IJKArrayStruct (GPU)
							IJKArrayStructCPU( const IJKArrayStruct& IJK )
							{
								iArray = IJK.iArray; 
								jArray = IJK.jArray;
								kArray = IJK.kArray;
							}
						};
inline IJKArrayStruct::IJKArrayStruct(const IJKArrayStructCPU& IJKCPU) {
    iArray = IJKCPU.iArray;
    jArray = IJKCPU.jArray;
    kArray = IJKCPU.kArray;
}

struct CompressedIJKArrayStruct { IntArrayType shifter; IntArrayType iArray; IntArrayType jArray; IntArrayType kArray; };

struct RayMapStruct { int gridID = 0; long long totalHitCount = 0LL; IntArrayType rayMapArray; LongLongArrayType hitCounterScanArray; };

struct VoxelizerStruct { InfoStruct Info; std::vector<RayMapStruct> rayMaps; RayMapStruct rayMapTotal; };

// NBR holds:
// Connectivity for Esotwist: indexes of 2 neighbours in the positive direction jPlus, kPlus. 
// jkPlus is always reached as jPlus[kPlus]
// Thanks to cell sorting where X runs the fastest, the indexes for remaining 4 neighbours iPlus, ijPlus, ikPlus, ijkPlus are just +1 to self, jPlus, kPlus, jkPlus.
// Then it holds 2 more neighbour indexes in the main negative directions jMinus, kMinus (iMinus would be self-1)
// In addition to that it holds a bit packed array of uint8_t, whose bits are 1 if the respective neighbour is also truly geometric neighbour (there is no gap between)
// There 7 used bits are ordered as is iPlus, jPlus, ijPlus, kPlus, ikPlus, jkPlus, ijkPlus
struct NBRArrayStruct { IntArrayType jPlusArray; IntArrayType kPlusArray;
						IntArrayType jMinusArray; IntArrayType kMinusArray; 
						Uint8_tArrayType isGeometricBitPackedMarkerArray; }; 
										
struct NBRStruct { 	int self;
					int iPlus; int jPlus; int kPlus; int ijPlus; int ikPlus; int jkPlus; int ijkPlus; 
					int iMinus; int jMinus; int kMinus; }; 
					
struct NBRHoleMapStruct { IntArray3DType holeStartArray; IntArray2DType startCounterArray; IntArray3DType holeEndArray; IntArray2DType endCounterArray; };

struct SkeletonGridStruct { InfoStruct Info; BoolArrayType keepCellMarkerArray; };

struct GridBuilderStruct { 	InfoStruct Info; IJKArrayStruct IJK; NBRArrayStruct NBR; 
							FloatArray2DType fArray; 
							IntArrayType parentMapArray; IntArrayType wallIDArray; IntArrayType wallAdjacentCellList;
							BoolArray2DType linkExistenceMarkerArray; FloatArray2DType linkLengthArray; BoolArray2DType linkPiercesInterfaceMarkerArray;
							BoolArrayType keepCellMarkerArray; BoolArrayType wallMarkerArray; 
							BoolArrayType refinementMarkerArray; BoolArrayType deepRefinementMarkerArray;
							BoolArrayType fineToCoarseMarkerArray; BoolArrayType coarseToFineMarkerArray;
							BoolArrayType parentInterfaceMarkerArray;
							SkeletonGridStruct SkeletonGrid; }; 

struct InterfaceStruct { int cellCount = 0; IntArrayType indexList; IntArrayType jMinusArray; IntArrayType kMinusArray; IntArrayType childMapArray; };
	
struct ForceTrackerArrayStruct { FloatArrayType gxArray; FloatArrayType gyArray; FloatArrayType gzArray; };
					
struct GridStruct { InfoStruct Info; 
					FloatArray2DType fArray; 
					bool esotwistFlipper = false; 
					CompressedIJKArrayStruct IJK;
					NBRArrayStruct NBR; 
					IntArrayType wallMapArray; Uint4ArrayType wallDataArray; ForceTrackerArrayStruct WallForceTracker;
					InterfaceStruct CoarseToFineInterface; InterfaceStruct FineToCoarseInterface; 
					IntArrayType BCIndexList; FloatArrayType BCMemoryArray; }; 	
					
struct STLStructCPU { 	int triangleCount = 0;
						FloatArrayTypeCPU axArray; FloatArrayTypeCPU ayArray; FloatArrayTypeCPU azArray; 
						FloatArrayTypeCPU bxArray; FloatArrayTypeCPU byArray; FloatArrayTypeCPU bzArray; 
						FloatArrayTypeCPU cxArray; FloatArrayTypeCPU cyArray; FloatArrayTypeCPU czArray; 
						BoundsStruct Bounds; }; 

struct STLStruct { 	static constexpr int threadsToTrianglesRatio = 4;
					int triangleCount = 0; float binSize = 0.f; static constexpr int avgTrianglesPerBin = 1;
					int binCountX = 0; int binCountY = 0; int binCountZ = 0; 
					float oxBin = 0.f; float oyBin = 0.f; float ozBin = 0.f;
					FloatArrayType axArray; FloatArrayType ayArray; FloatArrayType azArray; 
					FloatArrayType bxArray; FloatArrayType byArray; FloatArrayType bzArray; 
					FloatArrayType cxArray; FloatArrayType cyArray; FloatArrayType czArray; 
					BoundsStruct Bounds; 
					IntArrayType raysPerTriangleCounterArray;
					IntArrayType threadToTriangleMapArray;
					IntArrayType binArray; IntArrayType firstInBinArray;
					STLStruct() = default;
					// Constructor copies data from STLStructCPU
					STLStruct( const STLStructCPU& STLCPU )
					{
						triangleCount = STLCPU.triangleCount;
						axArray = STLCPU.axArray; ayArray = STLCPU.ayArray;	azArray = STLCPU.azArray; 
						bxArray = STLCPU.bxArray; byArray = STLCPU.byArray;	bzArray = STLCPU.bzArray; 
						cxArray = STLCPU.cxArray; cyArray = STLCPU.cyArray;	czArray = STLCPU.czArray; 
						Bounds = STLCPU.Bounds;	
					}
				};

struct FlowReportStruct { float uxPhys = 0.f; float uyPhys = 0.f; float uzPhys = 0.f; float pPhys = 0.f; float areamm2 = 0.f; 
							float massFlowPhys = 0.f; float normalDirectionMomentumFlowPhys = 0.f; float normalDirectionKineticEnergyFlowPhys = 0.f; };

struct LocalDuStruct { float duxdx = 0.f; float duydy = 0.f; float duzdz = 0.f; float duxdyCross = 0.f; float duydzCross = 0.f; float duxdzCross = 0.f; };

struct SectionCutStruct { 	FloatArray2DType rhoArray; FloatArray2DType uxArray; FloatArray2DType uyArray; FloatArray2DType uzArray; 
							FloatArray2DType markerArray; IntArray2DType gridIDArray; };
							
struct SectionCutStructCPU { 	FloatArray2DTypeCPU rhoArray; FloatArray2DTypeCPU uxArray; FloatArray2DTypeCPU uyArray; FloatArray2DTypeCPU uzArray; 
								FloatArray2DTypeCPU markerArray; IntArray2DTypeCPU gridIDArray; };
