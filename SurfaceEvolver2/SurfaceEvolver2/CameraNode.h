#pragma once
#define GLM_FORCE_RADIANS 1

#include "SceneNode.h"
#include "Device.h"
#include "SceneManager.h"
#include <glm/mat4x4.hpp>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

class CameraNode : public SceneNode
{
public:
	CameraNode(int initialWidth, int initialHeight);
	~CameraNode();
	void updateView();
	void setFov(float f);
	float getFov();
	void setLookAt(glm::vec3 newLookAt);
	glm::vec3 getLookAt();
	glm::mat4& getProjection();
	glm::mat4& getView();
	glm::vec3 getLookVector();
	glm::vec3& getUp();
private:
	float fov;
	float near;
	float far;
	int width;
	int height;
	glm::vec3 lookAt;
	glm::vec3 up;
	glm::mat4 projection;
	glm::mat4 view;
};

