#version 330

smooth in vec3 diffuseColor;
smooth in vec3 barycentric;
layout (location = 0) out vec3 diffuseOut;

void main(void) {
	if (any(lessThan(barycentric, vec3(0.02)))){
		diffuseOut = vec3(1.0, 0.0, 0.0);
	} else{
		diffuseOut = diffuseColor;
		//discard;
	}
}