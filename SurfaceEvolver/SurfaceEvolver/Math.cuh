#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaGeom.h"

__device__ vector subtractPoints(point a, point b);
__device__ vector crossProduct(vector u, vector v);