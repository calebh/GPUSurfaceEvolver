#pragma once
#include "Evolver.h"
#include "Simulate.cuh"

class GPUEvolver :
	public Evolver
{
public:
	GPUEvolver(Mesh* initMesh, int initItersUntilLambdaUpdate);
	~GPUEvolver();
private:
	void synchronizeToMesh();
	float stepSimulation(bool saveResults);
	float getArea();
	float getMeanNetForce();
	float getMeanCurvature();
	float getVolume();

	uint2 rearrangeTri(uint3 tri, int pointIndex);

	uint3* cudaTriangles;
	float3* cudaVertices;
	float3* cudaVertices2;
	float3* cudaAreaForces;
	float3* cudaVolumeForces;
	uint2* cudaTrianglesByVertex;
	uint* cudaTriangleOffset;
	uint* cudaTriangleCountPerVertex;
};

