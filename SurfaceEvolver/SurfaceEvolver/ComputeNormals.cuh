#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Math.cuh"

typedef struct {
	unsigned int a;
	unsigned int b;
	unsigned int c;
} indexedTriangle;

typedef struct {
	point a;
	point b;
	point c;
} unindexedTriangle;

__host__ void simToDisplay(indexedTriangle* trianglesIn, int numTriangles, point* verticesIn, point* verticesOut, vector* normalsOut, vector* barycentricOut);