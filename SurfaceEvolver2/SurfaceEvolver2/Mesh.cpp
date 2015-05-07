#include "Mesh.h"

Mesh::Mesh() {

}

Mesh::~Mesh()
{
	glDeleteBuffers(1, &normalBuff);
	glDeleteBuffers(1, &unindexedVertexBuff);
	glDeleteBuffers(1, &barycentricBuff);
}

template<class T> std::pair<GLuint, cudaGraphicsResource*> allocateGLBuff(int count) {
	std::pair<GLuint, cudaGraphicsResource*> ret;
	glGenBuffers(1, &ret.first);
	glBindBuffer(GL_ARRAY_BUFFER, ret.first);
	glBufferData(GL_ARRAY_BUFFER, sizeof(T) * count, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(ret.second), ret.first, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
	return ret;
}

void Mesh::initCudaBuffers() {
	{
		auto res = allocateGLBuff<float3>(3 * numFaces);
		unindexedVertexBuff = res.first;
		resources[0] = res.second;
	}

	{
		auto res = allocateGLBuff<float3>(3 * numFaces);
		normalBuff = res.first;
		resources[1] = res.second;
	}

	{
		auto res = allocateGLBuff<float3>(3 * numFaces);
		barycentricBuff = res.first;
		resources[2] = res.second;
	}
}

/*void Mesh::initCudaBuffers() {
	// This buffer is used to store vertice positions which are not indexed
	glGenBuffers(1, &unindexedVertexBuff);
	glBindBuffer(GL_ARRAY_BUFFER, unindexedVertexBuff);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* 3 * numFaces, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[0]), unindexedVertexBuff, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// This buffer is used to store normals per vertex
	glGenBuffers(1, &normalBuff);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuff);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* 3 * numFaces, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[1]), normalBuff, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// Used to draw outlines
	// See http://codeflow.org/entries/2012/aug/02/easy-wireframe-display-with-barycentric-coordinates/
	glGenBuffers(1, &barycentricBuff);
	glBindBuffer(GL_ARRAY_BUFFER, barycentricBuff);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* 3 * numFaces, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[2]), barycentricBuff, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}*/

void Mesh::updateDisplayBuffers(float3* cudaVertices, uint3* cudaTriangles) {
	// Lock the OpenGL buffers so that CUDA can use them
	cudaGraphicsMapResources(3, resources, NULL);

	// Get pointers to the buffers that CUDA can use
	float3* nonIndexedPos;
	size_t nonIndexedPosSize;
	cudaGraphicsResourceGetMappedPointer((void**)&nonIndexedPos, &nonIndexedPosSize, resources[0]);

	float3* normals;
	size_t normalsSize;
	cudaGraphicsResourceGetMappedPointer((void**)&normals, &normalsSize, resources[1]);

	float3* barycentric;
	size_t barycentricSize;
	cudaGraphicsResourceGetMappedPointer((void**)&barycentric, &barycentricSize, resources[2]);

	simToDisplay(cudaTriangles, numFaces, cudaVertices, nonIndexedPos, normals, barycentric);

	cudaGraphicsUnmapResources(3, resources, NULL);
}

void Mesh::draw(ShaderProgram* shader) {
	glBindBuffer(GL_ARRAY_BUFFER, unindexedVertexBuff);
	shader->vertexAttribPointer("position", 3, GL_FLOAT, 0, 0, GL_FALSE);

	glBindBuffer(GL_ARRAY_BUFFER, normalBuff);
	shader->vertexAttribPointer("normal", 3, GL_FLOAT, 0, 0, GL_FALSE);
	
	glBindBuffer(GL_ARRAY_BUFFER, barycentricBuff);
	shader->vertexAttribPointer("barycentricIn", 3, GL_FLOAT, 0, 0, GL_FALSE);

	glDrawArrays(GL_TRIANGLES, 0, numFaces*3);
}

std::vector<uint3>& Mesh::getTriangles() {
	return triangles;
}

std::vector<float3>& Mesh::getVertices() {
	return vertices;
}