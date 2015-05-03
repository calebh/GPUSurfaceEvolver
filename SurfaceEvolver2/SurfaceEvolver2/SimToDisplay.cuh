#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/nvidia/helper_math.h"
#include "vector_types.h"

__host__ void simToDisplay(uint3* trianglesIn, uint numTriangles, float3* verticesIn, float3* verticesOut, float3* normalsOut, float3* barycentricOut);