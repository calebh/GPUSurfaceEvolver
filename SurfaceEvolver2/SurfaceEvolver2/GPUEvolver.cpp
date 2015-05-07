#include "GPUEvolver.h"
template <class T> T* allocate(uint count) {
	auto size = sizeof(T)*count;
	T* ret;
	cudaMalloc(&ret, size);
	return ret;
}

// Allocate and initialize memory on the GPU
template <class T> T* allocateInit(T* source, uint count) {
	T* ret = allocate<T>(count);
	auto size = sizeof(T)*count;
	cudaMemcpy(ret, source, size, cudaMemcpyHostToDevice);
	return ret;
}

GPUEvolver::GPUEvolver(Mesh* initMesh, int initItersUntilLambdaUpdate)
	: Evolver(initMesh, initItersUntilLambdaUpdate)
{
	std::vector<uint3>& triangles = initMesh->getTriangles();
	std::vector<float3>& vertices = initMesh->getVertices();

	uint numVertices = vertices.size();
	uint numTriangles = triangles.size();

	cudaVertices = allocateInit<float3>(&(vertices[0]), numVertices);
	cudaVertices2 = allocate<float3>(numVertices);
	cudaTriangles = allocateInit<uint3>(&(triangles[0]), numTriangles);
	cudaAreaForces = allocate<float3>(numVertices);
	cudaVolumeForces = allocate<float3>(numVertices);

	auto triangleOffset = new uint[numVertices];
	auto trianglesByVertex = new uint2[numTriangles * 3];
	auto triangleCountPerVertex = new uint[numVertices];
	uint offset = 0;
	for (uint i = 0; i < numVertices; i++){
		uint triCount = 0;
		triangleOffset[i] = offset;
		for (uint j = 0; j < numTriangles; j++){
			if (triangles[j].x == i || triangles[j].y == i || triangles[j].z == i) {
				trianglesByVertex[offset + triCount] = rearrangeTri(triangles[j], i);
				triCount++;
			}
		}
		triangleCountPerVertex[i] = triCount;
		offset += triCount;
	}

	cudaTriangleOffset = allocateInit<uint>(triangleOffset, numVertices);
	cudaTrianglesByVertex = allocateInit<uint2>(trianglesByVertex, numTriangles*3);
	cudaTriangleCountPerVertex = allocateInit<uint>(triangleCountPerVertex, numVertices);
	
	delete trianglesByVertex;
	delete triangleOffset;
	delete triangleCountPerVertex;

	initDeviceVariables(numTriangles,
						numVertices,
						cudaAreaForces,
						cudaVolumeForces,
						cudaTriangleOffset,
						cudaTriangleCountPerVertex,
						cudaTrianglesByVertex,
						cudaTriangles);
}

GPUEvolver::~GPUEvolver()
{
	cudaFree(cudaAreaForces);
	cudaFree(cudaVolumeForces);
	cudaFree(cudaTriangleOffset);
	cudaFree(cudaTriangleCountPerVertex);
	cudaFree(cudaTrianglesByVertex);
	cudaFree(cudaTriangles);

	cudaFree(cudaVertices);
	cudaFree(cudaVertices2);
}

uint2 GPUEvolver::rearrangeTri(uint3 tri, int pointIndex) {
	uint2 ret;
	if (tri.x == pointIndex) {
		ret.x = tri.y;
		ret.y = tri.z;
	} else if (tri.y == pointIndex) {
		ret.y = tri.y;
		ret.x = tri.x;
	} else {
		ret.x = tri.x;
		ret.y = tri.y;
	}
	return ret;
}

float GPUEvolver::stepSimulation(bool saveResults) {
	float3* destinationVertices;
	if (saveResults) {
		destinationVertices = cudaVertices;
	} else {
		destinationVertices = cudaVertices2;
	}

	float area = stepCudaSimulation(lambda, cudaVertices, destinationVertices);
	
	if (saveResults) {
		float3* temp = cudaVertices;
		cudaVertices = cudaVertices2;
		cudaVertices2 = temp;
	}

	return area;
}