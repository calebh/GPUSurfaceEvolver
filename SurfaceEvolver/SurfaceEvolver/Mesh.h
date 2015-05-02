#pragma once

#include <GL/glew.h>
#include "ShaderProgram.h"
#include <iostream>
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "CudaGeom.h"
#include "ComputeNormals.cuh"

class Mesh
{
public:
	Mesh(const std::string& filename);
	~Mesh();
	void draw(ShaderProgram* shader);
	void update();
private:
	int numFaces;
	GLuint normalBufferObject;
	GLuint unindexedPosBufferObject;
	GLuint barycentricBufferObject;
	cudaGraphicsResource *resources[3];
	cudaGraphicsResource* cudaUnindexedPosBufferRes;
	cudaGraphicsResource* cudaNormalBufferRes;
	std::vector<indexedTriangle> indices;
	std::vector<point> vertices;
	//std::vector<GLfloat> normals;
	indexedTriangle* cudaIndices;
	point* cudaVertices;
};

