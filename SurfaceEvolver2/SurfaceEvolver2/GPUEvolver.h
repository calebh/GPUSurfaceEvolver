#pragma once
#include "Evolver.h"
#include "Simulate.cuh"

class GPUEvolver :
	public Evolver
{
public:
	GPUEvolver(Mesh* initMesh, int initItersUntilLambdaUpdate);
	~GPUEvolver();
	void synchronizeToMesh();
private:
	void stepSimulation();
	float getArea();
	float getMeanNetForce();
	float getMeanCurvature();
	float getVolume();

	virtual void outputPoints();
	virtual void outputVolumeForces();
	virtual void outputAreaForces();
	virtual void outputNetForces();

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

