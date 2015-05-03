#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <iostream>

class Transform
{
public:
	Transform();
	~Transform();
	void setTranslation();
	glm::mat4& getTransformation();
	glm::vec3 getTranslation();
	glm::vec3 getScale();
	glm::vec3 getRotation();
	void setTranslation(float x, float y, float z);
	void setTranslation(const glm::vec3& trans);
	void setRotation(float rotX, float rotY, float floatZ);
	void setScale(float sx, float sy, float sz);
	void print();
private:
	glm::vec3 translation;
	glm::vec3 scale;
	glm::vec3 rotation;
	glm::mat4 transformation;
	void computeTransformation();
};

