#pragma once

class CameraNode;
class ModelNode;
class LightNode;
class Device;
class ShaderProgram;

#include "Mesh.h"
#include <glm/gtc/type_ptr.hpp>
#include <vector>

class SceneManager
{
public:
	SceneManager(Device* initDevice);
	Device* getDevice();
	ShaderProgram* getGeometryProgram();
	void setGeometryProgram(ShaderProgram* prog);
	CameraNode* getCameraNode();
	~SceneManager();
	void addNode(CameraNode* node);
	void addNode(ModelNode* node);
	void drawAll();
private:
	std::vector<ModelNode*> modelNodes;
	Device* device;
	CameraNode* camera;
	ShaderProgram* geometryProgram;
	float diffuseConstant;
	float ambientConstant;
	float directionalIntensity;
	void geometryPass();
};
