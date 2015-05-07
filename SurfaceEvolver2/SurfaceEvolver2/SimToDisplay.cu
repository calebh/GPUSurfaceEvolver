#include "SimToDisplay.cuh"

// triangles should have length 3 * numTriangles
// normalsOut should have length 9 * numTriangles

// Triangles is an array of a1, b1, c1, a2, b2, c2 ...
// where A, B and C are vertices of the triangle
// list of indices

// Vertices is an array of v1x, v1y, v1z, v2x, v2y, v2z, ...

// For now just use one thread per block

// Converts indexed vertices to unindex version, and also computes the normals

__global__ void simToDisplayKernel (uint3* trianglesIn,
								    uint numTriangles,
								    float3* verticesIn, 
									float3* verticesOut,
									float3* normalsOut, 
									float3* barycentricOut) {
	// One block per triangle
	// Each triangle has three vertices, each with three components

	//int triangleIndex = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
	int triangleIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);

	if (triangleIndex < numTriangles) {
		int i = triangleIndex;
		float3 a = verticesIn[trianglesIn[i].x];
		float3 b = verticesIn[trianglesIn[i].y];
		float3 c = verticesIn[trianglesIn[i].z];

		int j = triangleIndex * 3;
		verticesOut[j    ] = a;
		verticesOut[j + 1] = b;
		verticesOut[j + 2] = c;

		// u and v are difference float3s. their cross product should give the normals
		// u = b-a
		// v = c-a

		float3 u = b - a;
		float3 v = c - a;

		// Compute the normal float3
		float3 n = cross(u, v);

		normalsOut[j    ] = n;
		normalsOut[j + 1] = n;
		normalsOut[j + 2] = n;

		barycentricOut[j    ] = { 1.0f, 0.0f, 0.0f };
		barycentricOut[j + 1] = { 0.0f, 1.0f, 0.0f };
		barycentricOut[j + 2] = { 0.0f, 0.0f, 1.0f };
	}
}


__host__ void simToDisplay(uint3* trianglesIn,
						   uint numTriangles,
						   float3* verticesIn,
						   float3* verticesOut,
						   float3* normalsOut,
						   float3* barycentricOut) {
	dim3 grid = { numTriangles, 1, 1 };
	dim3 block = { 1, 1, 1  };
	simToDisplayKernel<<<grid, block>>>(trianglesIn, numTriangles, verticesIn, verticesOut, normalsOut, barycentricOut);
}