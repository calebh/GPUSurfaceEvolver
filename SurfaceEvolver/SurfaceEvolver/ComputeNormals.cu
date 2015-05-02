#include "ComputeNormals.cuh"

// triangles should have length 3 * numTriangles
// normalsOut should have length 9 * numTriangles

// Triangles is an array of a1, b1, c1, a2, b2, c2 ...
// where A, B and C are vertices of the triangle
// list of indices

// Vertices is an array of v1x, v1y, v1z, v2x, v2y, v2z, ...

// For now just use one thread per block

// Converts indexed vertices to unindex version, and also computes the normals

__global__ void simToDisplayKernel (indexedTriangle* trianglesIn,
								    int numTriangles,
								    point* verticesIn, 
									point* verticesOut,
									vector* normalsOut, 
									vector* barycentricOut) {
	// One block per triangle
	// Each triangle has three vertices, each with three components

	int triangleIndex = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);

	if (triangleIndex < numTriangles) {
		int i = triangleIndex;
		point a = verticesIn[trianglesIn[i].a];
		point b = verticesIn[trianglesIn[i].b];
		point c = verticesIn[trianglesIn[i].c];

		int j = triangleIndex * 3;
		verticesOut[j  ] = a;
		verticesOut[j+1] = b;
		verticesOut[j+2] = c;

		// u and v are difference vectors. their cross product should give the normals
		// u = b-a
		// v = c-a

		vector u = subtractPoints(b, a);
		vector v = subtractPoints(c, a);

		// Compute the normal vector
		vector n = crossProduct(u, v);

		normalsOut[triangleIndex * 3    ] = n;
		normalsOut[triangleIndex * 3 + 1] = n;
		normalsOut[triangleIndex * 3 + 2] = n;

		barycentricOut[triangleIndex * 3    ] = { 1.0f, 0.0f, 0.0f };
		barycentricOut[triangleIndex * 3 + 1] = { 0.0f, 1.0f, 0.0f };
		barycentricOut[triangleIndex * 3 + 2] = { 0.0f, 0.0f, 1.0f };
	}
}


__host__ void simToDisplay(indexedTriangle* trianglesIn,
						   int numTriangles,
						   point* verticesIn,
						   point* verticesOut,
						   vector* normalsOut,
						   vector* barycentricOut) {
	dim3 grid;
	dim3 block;

	grid.x = numTriangles;
	grid.y = 1;
	grid.z = 1;

	block.x = 1;
	block.y = 1;
	block.z = 1;

	simToDisplayKernel<<<grid, block>>>(trianglesIn, numTriangles, verticesIn, verticesOut, normalsOut, barycentricOut);
}