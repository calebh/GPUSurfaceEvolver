#include "SceneManager.h"
#include "CameraNode.h"
#include "ModelNode.h"
#include "Device.h"
#include "ShaderProgram.h"

SceneManager::SceneManager(Device* initDevice) :
	device(initDevice),
	geometryProgram(NULL),
	camera(NULL),
	diffuseConstant(0.7f),
	ambientConstant(0.1f),
	directionalIntensity(0.2f)
{
	
}


SceneManager::~SceneManager()
{
}

Device* SceneManager::getDevice() { return device;}
void SceneManager::setGeometryProgram(ShaderProgram* prog) { geometryProgram = prog; }
ShaderProgram* SceneManager::getGeometryProgram() { return geometryProgram; }
CameraNode* SceneManager::getCameraNode() { return camera; }

void SceneManager::addNode(CameraNode* node) {
	camera = node;
}

void SceneManager::addNode(ModelNode* node) {
	modelNodes.push_back(node);
}

void SceneManager::geometryPass() {
	geometryProgram->use();
	
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	for (ModelNode* m : modelNodes) {
		m->geometryPass(this);
	}
}

void SceneManager::drawAll() {
	// Update the camera matrices
	camera->updateView();
	geometryPass();
}