#include "ModelNode.h"


ModelNode::ModelNode() :
	mesh(NULL)
{
}

ModelNode::ModelNode(Mesh* m) :
	mesh(m)
{
}

ModelNode::~ModelNode()
{
}

void ModelNode::setMesh(Mesh* m) {
	mesh = m;
}

Mesh* ModelNode::getMesh() {
	return mesh;
}

void ModelNode::geometryPass(SceneManager* manager) {
	if (mesh != NULL) {
		CameraNode* camera = manager->getCameraNode();
		ShaderProgram* shaderProgram = manager->getGeometryProgram();
		
		glm::mat4 mv = camera->getView() * getTransform().getTransformation();
		//glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(getTransform().getTransformation()));
		glm::mat4 mvp = camera->getProjection() * mv;
		
		shaderProgram->setUniformMatrix4fv("mvp", 1, false, glm::value_ptr(mvp));
		//shaderProgram->setUniformMatrix3fv("normalMatrix", 1, false, glm::value_ptr(normalMatrix));
		mesh->draw(shaderProgram);
	}
}