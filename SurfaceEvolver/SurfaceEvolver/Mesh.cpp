#include "Mesh.h"

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
			point p = { vertex->x, vertex->y, vertex->z };
			vertices.push_back(p);
		}
		numFaces = (int) m->mNumFaces;
		std::cout << "Num vertices: " << m->mNumVertices << std::endl;
		for (unsigned int i = 0; i < m->mNumFaces; i++) {
			aiFace* face = &(m->mFaces[i]);
			if (face->mNumIndices != 3) {
				std::cerr << "numIndices is not 3";
			}
			indexedTriangle tri = {face->mIndices[0], face->mIndices[1], face->mIndices[2]};
			indices.push_back(tri);
		}
		std::cout << "Num faces: " << m->mNumFaces << std::endl;
	} else {
		std::cerr << "Scene has no meshes";
	}
	
	/*glGenBuffers(1, &positionBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)* vertices.size(), &(vertices[0]), GL_STATIC_DRAW);

	glGenBuffers(1, &indexBufferObject);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferObject);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)* indices.size(), &(indices[0]), GL_STATIC_DRAW);*/

	auto verticesSize = sizeof(point)*vertices.size();
	cudaMalloc(&cudaVertices, verticesSize);
	cudaMemcpy(cudaVertices, &(vertices[0]), verticesSize, cudaMemcpyHostToDevice);

	std::cout << "First memcpy is away!" << std::endl;

	auto indiciesSize = sizeof(indexedTriangle)*indices.size();
	cudaMalloc(&cudaIndices, indiciesSize);
	cudaMemcpy(cudaIndices, &(indices[0]), indiciesSize, cudaMemcpyHostToDevice);
	
	glGenBuffers(1, &unindexedPosBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, unindexedPosBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)* numFaces * 9, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[0]), unindexedPosBufferObject, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	glGenBuffers(1, &normalBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, normalBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)* numFaces * 9, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[1]), normalBufferObject, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	// Used to draw outlines
	glGenBuffers(1, &barycentricBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, barycentricBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)* numFaces * 9, NULL, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&(resources[2]), barycentricBufferObject, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

	update();
}

Mesh::~Mesh()
{
}

void Mesh::update() {
	cudaGraphicsMapResources(3, resources, NULL);

	point* nonIndexedPos;
	size_t nonIndexedPosSize;
	cudaGraphicsResourceGetMappedPointer((void**)&nonIndexedPos, &nonIndexedPosSize, resources[0]);

	vector* normals;
	size_t normalsSize;
	cudaGraphicsResourceGetMappedPointer((void**)&normals, &normalsSize, resources[1]);

	vector* barycentric;
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