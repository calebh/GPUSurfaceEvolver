#include "Transform.h"


Transform::Transform() :
	translation(0.0f),
	scale(1.0f),
	rotation(0.0f)
{
}


Transform::~Transform()
{
}

void Transform::setTranslation(float x, float y, float z) {
	translation.x = x;
	translation.y = y;
	translation.z = z;
	computeTransformation();
}

void Transform::setTranslation(const glm::vec3& trans) {
	translation = trans;
	computeTransformation();
}

void Transform::setRotation(float rotX, float rotY, float rotZ) {
	rotation.x = rotX;
	rotation.y = rotY;
	rotation.z = rotZ;
	computeTransformation();
}

void Transform::setScale(float sx, float sy, float sz) {
	scale.x = sx;
	scale.y = sy;
	scale.z = sz;
	computeTransformation();
}

glm::vec3 Transform::getTranslation() {
	return translation;
}

glm::vec3 Transform::getRotation() {
	return rotation;
}

glm::vec3 Transform::getScale() {
	return scale;
}

void Transform::computeTransformation() {
	glm::mat4 t = glm::translate(glm::mat4(), translation);
	glm::mat4 r = glm::eulerAngleYXZ(rotation.y, rotation.x, rotation.z);
	glm::mat4 s = glm::scale(glm::mat4(), scale);
	transformation = t*r*s;
}

glm::mat4& Transform::getTransformation() {
	return transformation;
}

void Transform::print() {
	for (int i = 0; i < 4; i++) {
		glm::vec4 column = transformation[i];
		for (int j = 0; j < 4; j++) {
			std::cout << column[j];
			std::cout << ',';
		}
		std::cout << '\n';
	}
}