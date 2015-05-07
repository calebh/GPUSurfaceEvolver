#pragma once
#include "Mesh.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class ExternalMesh :
	public Mesh
{
public:
	ExternalMesh(const std::string& filename);
	~ExternalMesh();
};

