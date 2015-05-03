#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <fstream>
#include <iostream>

enum ShaderType {VERTEX, FRAGMENT, GEOMETRY, TESSELATION};

class Shader
{
public:
	Shader(std::string fileName, ShaderType stype);
	~Shader();
	GLuint getShaderHandle();
private:
	GLuint shaderHandle;
	void Shader::load(std::string& fileName);
};

