#include "ExternalMesh.h"


ExternalMesh::ExternalMesh(const std::string& filename) {
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
		numFaces = (int)m->mNumFaces;
		std::cout << "Num vertices: " << m->mNumVertices << std::endl;
		for (unsigned int i = 0; i < m->mNumFaces; i++) {
			aiFace* face = &(m->mFaces[i]);
			if (face->mNumIndices != 3) {
				std::cerr << "numIndices is not 3";
			}
			uint3 tri = { face->mIndices[0], face->mIndices[1], face->mIndices[2] };
			triangles.push_back(tri);
		}
		std::cout << "Num faces: " << m->mNumFaces << std::endl;
	}
	else {
		std::cerr << "Scene has no meshes";
	}

	initCudaBuffers();
}


ExternalMesh::~ExternalMesh()
{
}
