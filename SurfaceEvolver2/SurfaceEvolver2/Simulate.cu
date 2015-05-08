#include "Simulate.cuh"

#define TINY_AMOUNT           0.000001
#define TEMP                  0.0001
#define UPDATE_ITERATIONS     20
#define SIGMA                 1.0f

// REMEMBER TO INITIALIZE THESE
__device__ float d_sum1;
__device__ float d_sum2;
__device__ float d_alpha;
__device__ float d_area;

__device__ uint d_numTriangles;
           uint h_numTriangles;
__device__ uint d_numVertices;
           uint h_numVertices;
__device__ float3* d_areaForce;
__device__ float3* d_volumeForce;
__device__ uint* d_triangleOffset; // Gives the offset to start at in the trianglesByVertexArray
__device__ uint* d_triangleCountPerVertex; // Gives the number of triangles that this vertex is a part of
__device__ uint2* d_trianglesByVertex;  // Given some vertex residing at position i in the vertices array,
								        // trianglesByVertex[triangleOffset[i]] to trianglesByVertex[triangleOffset[i]+triangleCountPerVertex-1]
								        // are the triangles that i is a part of.
__device__ uint3* d_triangles;
		   uint h_maxTrianglesPerVertex;



__global__ void initDeviceVariablesKernel(uint numTriangles,
										  uint numVertices,
										  float3* areaForce,
										  float3* volumeForce,
										  uint* triangleOffset,
										  uint* triangleCountPerVertex,
										  uint2* trianglesByVertex,
										  uint3* triangles) {
	d_numTriangles = numTriangles;
	d_numVertices = numVertices;
	d_areaForce = areaForce;
	d_volumeForce = volumeForce;
	d_triangleOffset = triangleOffset;
	d_triangleCountPerVertex = triangleCountPerVertex;
	d_trianglesByVertex = trianglesByVertex;
	d_triangles = triangles;
}

__host__ void initDeviceVariables(uint numTriangles,
	                              uint numVertices,
	                              float3* areaForce,
	                              float3* volumeForce,
	                              uint* triangleOffset,
	                              uint* triangleCountPerVertex,
	                              uint2* trianglesByVertex,
	                              uint3* triangles,
								  uint maxTrianglesByVertex) {
	dim3 grid = { 1, 1, 1 };
	dim3 block = { 1, 1, 1 };
	initDeviceVariablesKernel<<<grid, block>>>(numTriangles,
											   numVertices,
											   areaForce,
											   volumeForce,
											   triangleOffset,
											   triangleCountPerVertex,
											   trianglesByVertex,
											   triangles);
	h_numTriangles = numTriangles;
	h_maxTrianglesPerVertex = maxTrianglesByVertex;
	h_numVertices = numVertices;
	cudaDeviceSynchronize();
}

__global__ void cleanDeviceVariablesKernel() {
	d_sum1 = 0.0f;
	d_sum2 = 0.0f;
	d_area = 0.0f;
}

__host__ void cleanDeviceVariables() {
	dim3 grid = { 1, 1, 1 };
	dim3 block = { 1, 1, 1 };
	cleanDeviceVariablesKernel<<<grid, block>>>();
	cudaDeviceSynchronize();
}

// Note that only the addition of each component is guarenteed to be atomic
__device__ void atomicAddVecComponents(float3* destination, float3 toAdd) {
	atomicAdd(&(destination->x), toAdd.x);
	atomicAdd(&(destination->y), toAdd.y);
	atomicAdd(&(destination->z), toAdd.z);
}

// For now we assume that there is one block per vertex which has only one thread
/*__global__ void calculateForces(float3* vertices) {
	int thisVertex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	//int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
	if (thisVertex < d_numVertices) {
		uint offset = d_triangleOffset[thisVertex];
		uint numTriangles = d_triangleCountPerVertex[thisVertex];
		float3 areaForce = { 0.0f, 0.0f, 0.0f };
		float3 volumeForce = { 0.0f, 0.0f, 0.0f };
		for (uint i = offset; i < offset + numTriangles; i++) {
			uint2 tri = d_trianglesByVertex[i];
			float3 x1 = vertices[thisVertex];
			float3 x2 = vertices[tri.x];
			float3 x3 = vertices[tri.y];

			float3 s1 = x2 - x1;
			float3 s2 = x3 - x2;

			float3 c = cross(s1, s2);
			float3 a = (SIGMA / 2.0f) * cross(s2, c / length(c));
			areaForce += a;
			volumeForce += cross(x2, x3) / 6.0f;
		}
		d_areaForce[thisVertex] = areaForce;
		d_volumeForce[thisVertex] = volumeForce;
	}
}*/

__global__ void calculateForces(float3* vertices) {
	__shared__ float3 areaForce;
	__shared__ float3 volumeForce;

	uint thisVertex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	uint thisTriangle = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

	if (thisTriangle == 0) {
		areaForce = { 0.0f, 0.0f, 0.0f };
		volumeForce = { 0.0f, 0.0f, 0.0f };
	}

	__syncthreads();

	if (thisVertex < d_numVertices) {
		uint numTriangles = d_triangleCountPerVertex[thisVertex];
		if (thisTriangle < numTriangles) {
			uint offset = d_triangleOffset[thisVertex];
			uint i = offset + thisTriangle;
			uint2 tri = d_trianglesByVertex[i];
			float3 x1 = vertices[thisVertex];
			float3 x2 = vertices[tri.x];
			float3 x3 = vertices[tri.y];

			float3 s1 = x2 - x1;
			float3 s2 = x3 - x2;

			float3 c = cross(s1, s2);
			atomicAddVecComponents(&areaForce, (SIGMA / 2.0f) * cross(s2, c / length(c)));
			atomicAddVecComponents(&volumeForce, cross(x2, x3) / 6.0f);
			//volumeForce += cross(x2, x3) / 6.0f;
		}
	}

	__syncthreads();

	if (thisTriangle == 0) {
		d_areaForce[thisVertex] = areaForce;
		d_volumeForce[thisVertex] = volumeForce;
	}
}

__global__ void displaceVertices(float lambda,
					  float3* points1,
					  float3* points2)
{
	int vertexIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	if (vertexIndex < d_numVertices) {
		points2[vertexIndex] = points1[vertexIndex] + lambda*(d_areaForce[vertexIndex] - d_alpha*d_volumeForce[vertexIndex]);
	}
}

__global__ void calculateAreaKernel(float3* vertices) {
	//int thisTriangle = blockIdx.x + blockIdx.y * blockDim.x + blockIdx.z * (blockDim.x * blockDim.y);
	int thisTriangle = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	if (thisTriangle < d_numTriangles) {
		uint3 t = d_triangles[thisTriangle];
		float3 s1 = vertices[t.y] - vertices[t.x];
		float3 s2 = vertices[t.z] - vertices[t.y];
		float area = length(cross(s1, s2) / 2);
		atomicAdd(&d_area, area);
	}
}

__host__ float calculateArea(float3* vertices) {
	cleanDeviceVariables();
	dim3 grid = { h_numTriangles, 1, 1 };
	dim3 block = { 1, 1, 1 };
	calculateAreaKernel<<<grid, block>>>(vertices);
	cudaDeviceSynchronize();
	float area = d_area;
	cudaMemcpyFromSymbol(&area, d_area, sizeof(area), 0, cudaMemcpyDeviceToHost);
	return area;
}

__global__ void calculateAlphaKernelSum() {
	int thisVertex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
	float3 vf = d_volumeForce[thisVertex];
	atomicAdd(&d_sum1, dot(vf, d_areaForce[thisVertex]));
	atomicAdd(&d_sum2, dot(vf, vf));
}

__global__ void calculateAlphaKernelDivide() {
	d_alpha = d_sum1 / d_sum2;
}

// Synchronize between kernel calls!!!
__host__ float stepCudaSimulation(float lambda,
						    float3* sourceVertices,
							float3* destinationVertices) {
	cleanDeviceVariables();

	int gridX;
	cudaDeviceGetAttribute(&gridX, cudaDevAttrMaxGridDimX, 0);
	uint verticesYGrid = (h_numVertices / gridX) + 1;

	// 1. Calculate forces
	{
		dim3 grid = { h_numVertices, verticesYGrid, 1 };
		dim3 block = { h_maxTrianglesPerVertex, 1, 1 };
		calculateForces<<<grid, block>>>(sourceVertices);
	}

	cudaDeviceSynchronize();

	// 2. Calculate alpha sum
	{
		dim3 grid = { h_numVertices, verticesYGrid, 1 };
		dim3 block = { 1, 1, 1 };
		calculateAlphaKernelSum<<<grid, block>>>();
	}

	// 2.5 Calculate alpha division
	{
		dim3 grid = { 1, 1, 1 };
		dim3 block = { 1, 1, 1 };
		calculateAlphaKernelDivide<<<grid, block>>>();
	}

	cudaDeviceSynchronize();

	// 3. Displace the vertices
	{
		dim3 grid = { h_numVertices, verticesYGrid, 1 };
		dim3 block = { 1, 1, 1 };
		displaceVertices<<<grid, block>>>(lambda, sourceVertices, destinationVertices);
	}

	cudaDeviceSynchronize();

	// 4. Calculate the new area and return
	return calculateArea(destinationVertices);
}