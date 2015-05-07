#pragma once

#include <GL/glew.h>
#include "ShaderProgram.h"
#include <iostream>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "vector_types.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "SimToDisplay.cuh"

class Mesh
{
public:
	Mesh(int size);
	Mesh(const std::string& filename);
	~Mesh();
	void draw(ShaderProgram* shader);
	void updateDisplayBuffers();
private:
	void initCudaBuffers();
	int numFaces;
	cudaGraphicsResource *resources[3];
	std::vector<uint3> indices;
	std::vector<float3> vertices;
	uint3* cudaIndices;
	float3* cudaVertices;
	GLuint normalBufferObject;
	GLuint unindexedPosBufferObject;
	GLuint barycentricBufferObject;
};

