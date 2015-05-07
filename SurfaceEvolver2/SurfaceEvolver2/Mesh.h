#pragma once

#include <GL/glew.h>
#include "ShaderProgram.h"
#include <iostream>
#include <vector>
#include <utility>

#include "vector_types.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "SimToDisplay.cuh"

class Mesh
{
public:
	Mesh();
	Mesh(int size);
	~Mesh();

	void draw(ShaderProgram* shader);
	void updateDisplayBuffers(float3* cudaVertices, uint3* cudaTriangles);

	std::vector<uint3>& getTriangles();
	std::vector<float3>& getVertices();

private:
	cudaGraphicsResource *resources[3];
	GLuint normalBuff;
	GLuint unindexedVertexBuff;
	GLuint barycentricBuff;

protected:
	std::vector<uint3> triangles;
	std::vector<float3> vertices;
	void initCudaBuffers();
	int numFaces;
};

