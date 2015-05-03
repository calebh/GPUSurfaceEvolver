#include "CameraNode.h"

CameraNode::CameraNode(int initialWidth, int initialHeight) :
	width(initialWidth),
	height(initialHeight),
	fov(0.785398163f),
	near(0.01f),
	far(5000.0f),
	up(0.0f, 1.0f, 0.0f)
{
	float ratio = width / ((float)height);
	projection = glm::perspective(fov, ratio, near, far);
}


CameraNode::~CameraNode()
{
}

float CameraNode::getFov() {
	return fov;
}

void CameraNode::setFov(float f) {
	fov = f;
}

void CameraNode::updateView() {
	view = glm::lookAt(getTransform().getTranslation(), lookAt, up);
}

void CameraNode::setLookAt(glm::vec3 newLookAt) {
	lookAt = newLookAt;
}

glm::vec3 CameraNode::getLookAt() {
	return lookAt;
}

glm::mat4& CameraNode::getProjection() {
	return projection;
}

glm::mat4& CameraNode::getView() {
	return view;
}

glm::vec3 CameraNode::getLookVector() {
	return glm::normalize(lookAt - getTransform().getTranslation());
}

glm::vec3& CameraNode::getUp() {
	return up;
}