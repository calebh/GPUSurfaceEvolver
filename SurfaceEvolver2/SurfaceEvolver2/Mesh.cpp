#include "Mesh.h"

int** new2D(int size) {
	int** ary = new int*[size];
	for (int i = 0; i < size; ++i) {
		ary[i] = new int[size];
	}
	return ary;
}

void delete2D(int** ary, int size) {
	for (int i = 0; i < size; ++i) {
		delete[] ary[i];
	}
	delete[] ary;
}


Mesh::Mesh(const int size) {
	int trianglesPerFace = size*size;
	numFaces = trianglesPerFace * 4;
	vertices.resize(4 + 6 * (size - 1) + 2 * (size - 2)*(size - 1));
	indices.resize(4 * trianglesPerFace);
	float3 initialPos = { size / 2.0, size / 2.0, size / 2.0 };
	float3 deltaX = { 0, 1, -1 };
	float3 deltaY = { -1, -1, 0 };
	int** indices1 = new2D(size + 1);
	int** indices2 = new2D(size + 1);
	int** indices3 = new2D(size + 1);
	int** indices4 = new2D(size + 1);
	int coordCount = 0, triangleCount = 0;

	// PLane 1:
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			indices1[x][y] = coordCount;
			vertices[coordCount++] = currentPos;
			if (x > 0) {
				indices[triangleCount++] = { indices1[x][y], indices1[x - 1][y], indices1[x - 1][y - 1] };
				if (x < y) {
					indices[triangleCount++] = { indices1[x][y], indices1[x - 1][y - 1], indices1[x][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}
	
	// Plane 2:
	deltaX = { 1, 0, -1 };
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			if (x == 0){
				indices2[x][y] = indices1[x][y];
			}
			else{
				indices2[x][y] = coordCount;
				vertices[coordCount++] = currentPos;
			}
			if (x > 0) {
				indices[triangleCount++] = { indices2[x][y], indices2[x - 1][y - 1], indices2[x - 1][y] };
				if (x < y) {
					indices[triangleCount++] = { indices2[x][y], indices2[x][y - 1], indices2[x - 1][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}

	// plane 3
	initialPos = { -size / 2.0, size / 2.0, -size / 2.0 };
	deltaY = { 1, -1, 0 };
	deltaX = { 0, 1, 1 };
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			if (y == size){
				indices3[x][y] = indices2[size - x][size - x];
			}
			else if (x == y){
				indices3[x][y] = indices1[size - x][size - x];
			}
			else{
				indices3[x][y] = coordCount;
				vertices[coordCount++] = currentPos;
			}
			if (x > 0) {
				indices[triangleCount++] = { indices3[x][y], indices3[x - 1][y], indices3[x - 1][y - 1] };
				if (x < y) {
					indices[triangleCount++] = { indices3[x][y], indices3[x - 1][y - 1], indices3[x][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}

	//plane 4
	deltaX = { -1, 0, 1 }; 
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			if (y == size){
				indices4[x][y] = indices2[size - x][size];
			}
			else if (x == y){
				indices4[x][y] = indices1[size - y][size];
			}
			else if (x == 0){
				indices4[x][y] = indices3[x][y];
			}
			else{
				indices4[x][y] = coordCount;
				vertices[coordCount++] = currentPos;
			}
			if (x > 0) {
				indices[triangleCount++] = { indices4[x][y], indices4[x - 1][y - 1], indices4[x - 1][y] };
				if (x < y) {
					indices[triangleCount++] = { indices4[x][y], indices4[x][y - 1], indices4[x - 1][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}
	initCudaBuffers();
	update();
}

Mesh::Mesh(const std::string& filename) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(filename,
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType);

	if (!scene) {
		std::cerr << "Scene is null";
	}
	
	if (scene->HasMeshes()) {
		aiMesh* m = scene->mMeshes[0];
		for (unsigned int i = 0; i < m->mNumVertices; i++) {
			aiVector3D* vertex = &(m->mVertices[i]);
			float3 p = { vertex->x, vertex->y, vertex->z };
			vertices.push_back(p);
		}
		numFaces = (int) m->mNumFaces;
		std::cout << "Num vertices: " << m->mNumVertices << std::endl;
		for (unsigned int i = 0; i < m->mNumFaces; i++) {
			aiFace* face = &(m->mFaces[i]);
			if (face->mNumIndices != 3) {
				std::cerr << "numIndices is not 3";
			}
			uint3 tri = {face->mIndices[0], face->mIndices[1], face->mIndices[2]};
			indices.push_back(tri);
		}
		std::cout << "Num faces: " << m->mNumFaces << std::endl;
	} else {
		std::cerr << "Scene has no meshes";
	}
	
	initCudaBuffers();
	update();
}

Mesh::~Mesh()
{
}

void Mesh::initCudaBuffers() {
	auto verticesSize = sizeof(float3)*vertices.size();
	cudaMalloc(&cudaVertices, verticesSize);
	cudaMemcpy(cudaVertices, &(vertices[0]), verticesSize, cudaMemcpyHostToDevice);

	auto indiciesSize = sizeof(uint3)*indices.size();
	cudaMalloc(&cudaIndices, indiciesSize);
	cudaMemcpy(cudaIndices, &(indices[0]), indiciesSize, cudaMemcpyHostToDevice);

	// This buffer is used to store vertice positions which are not indexed
	glGenBuffers(1, &unindexedPosBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, unindexedPosBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* 3 * numFaces, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[0]), unindexedPosBufferObject, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// This buffer is used to store normals per vertex
	glGenBuffers(1, &normalBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, normalBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* 3 * numFaces, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[1]), normalBufferObject, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// Used to draw outlines
	// See http://codeflow.org/entries/2012/aug/02/easy-wireframe-display-with-barycentric-coordinates/
	glGenBuffers(1, &barycentricBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, barycentricBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)* 3 * numFaces, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[2]), barycentricBufferObject, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

void Mesh::update() {
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

	simToDisplay(cudaIndices, numFaces, cudaVertices, nonIndexedPos, normals, barycentric);

	cudaGraphicsUnmapResources(3, resources, NULL);
}

void Mesh::draw(ShaderProgram* shader) {
	glBindBuffer(GL_ARRAY_BUFFER, unindexedPosBufferObject);
	shader->vertexAttribPointer("position", 3, GL_FLOAT, 0, 0, GL_FALSE);

	glBindBuffer(GL_ARRAY_BUFFER, normalBufferObject);
	shader->vertexAttribPointer("normal", 3, GL_FLOAT, 0, 0, GL_FALSE);
	
	glBindBuffer(GL_ARRAY_BUFFER, barycentricBufferObject);
	shader->vertexAttribPointer("barycentricIn", 3, GL_FLOAT, 0, 0, GL_FALSE);

	glDrawArrays(GL_TRIANGLES, 0, numFaces*3);
}