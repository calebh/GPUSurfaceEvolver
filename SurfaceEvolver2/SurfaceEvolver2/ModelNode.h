#pragma once
#include "SceneManager.h"
#include "CameraNode.h"
#include "SceneNode.h"
#include "Mesh.h"
#include <glm/mat4x4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/gtc/matrix_inverse.hpp>

class ModelNode : public SceneNode
{
public:
	ModelNode();
	ModelNode(Mesh* m);
	~ModelNode();
	void geometryPass(SceneManager* manager);
	Mesh* getMesh();
	void setMesh(Mesh* m);
private:
	Mesh* mesh;
};

