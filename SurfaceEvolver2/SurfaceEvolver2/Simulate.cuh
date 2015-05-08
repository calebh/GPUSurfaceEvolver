#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/nvidia/helper_math.h"
#include "vector_types.h"

__host__ void initDeviceVariables(uint numTriangles,
	uint numVertices,
	float3* areaForce,
	float3* volumeForce,
	uint* triangleOffset,
	uint* triangleCountPerVertex,
	uint2* trianglesByVertex,
	uint3* triangles);

__host__ void stepCudaSimulation(float lambda,
	float3* sourceVertices,
	float3* destinationVertices
	/*int maxTrianglesPerVertex*/);

__host__ float calculateArea(float3* vertices);