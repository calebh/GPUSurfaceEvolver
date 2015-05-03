#include "Shader.h"
#include <vector>

Shader::Shader(std::string fileName, ShaderType stype) {
	switch (stype) {
		case VERTEX:
			shaderHandle = glCreateShader(GL_VERTEX_SHADER_ARB);
			break;
		case FRAGMENT:
			shaderHandle = glCreateShader(GL_FRAGMENT_SHADER_ARB);
			break;
		case GEOMETRY:
			shaderHandle = glCreateShader(GL_GEOMETRY_SHADER_EXT);
			break;
	}

	std::ifstream shaderSource(fileName.c_str());
	if (!shaderSource.is_open()) {
		std::cerr << "File not found " << fileName.c_str() << "\n";
		exit(EXIT_FAILURE);
	}

	// now read in the data
	std::string source((std::istreambuf_iterator<char>(shaderSource)), std::istreambuf_iterator<char>());
	shaderSource.close();
	source += "\0";

	const char* data = source.c_str();
	glShaderSource(shaderHandle, 1, &data, NULL);
	glCompileShader(shaderHandle);

	GLint isCompiled = 0;
	glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE) {
		GLint maxLength = 0;
		glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &maxLength);

		//The maxLength includes the NULL character
		std::vector<GLchar> infoLog(maxLength);
		glGetShaderInfoLog(shaderHandle, maxLength, &maxLength, &infoLog[0]);

		std::cout << &infoLog[0] << std::endl;

		//We don't need the shader anymore.
		//glDeleteShader(vertexShader);

		//Use the infoLog as you see fit.

		//In this simple program, we'll just leave
		//return;
	}
}

Shader::~Shader() {}

GLuint Shader::getShaderHandle() {
	return shaderHandle;
}