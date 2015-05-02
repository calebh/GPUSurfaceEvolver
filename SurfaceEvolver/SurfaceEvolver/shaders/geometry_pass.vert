#version 330

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 barycentricIn;

uniform mat4 mvp;
smooth out vec3 diffuseColor;
smooth out vec3 barycentric;

void main() {
	gl_Position = mvp * position;

	//float len = mod(length(normal), 256)/256;
	diffuseColor = vec3(0.6, 0.6, 0.6) * abs(dot(normalize(normal),normalize(vec3(-1,-1,-1))));
	//diffuseColor = vec3(0.6, 0.6, 0.6) * clamp(-dot(normalize(normal),normalize(vec3(-1,-1,-1))), 0.0, 1.0);
	barycentric = barycentricIn;
}