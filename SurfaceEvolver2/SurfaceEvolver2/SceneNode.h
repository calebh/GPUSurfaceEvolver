#pragma once

#include "Transform.h"
#include "Device.h"
#include "SceneManager.h"
#include "Mesh.h"

class SceneNode
{
public:
	SceneNode();
	~SceneNode();
	Transform& getTransform();

private:
	Transform transform;
	Mesh* mesh;
};