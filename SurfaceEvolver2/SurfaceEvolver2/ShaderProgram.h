#pragma once

#include <GL/glew.h>
#include <vector>
#include "Shader.h"
#include <iostream>
#include <map>

class ShaderProgram
{
public:
	ShaderProgram();
	~ShaderProgram();
	void attachShader(Shader* sh);
	void bindAttrib(std::string name, GLuint index);
	void ShaderProgram::vertexAttribPointer(std::string name, GLint size, GLenum type, GLsizei stride, const GLvoid* data, bool normalize);
	void link();
	GLuint getProgramID();
	void use();

	void setUniform1f(const char* varname, GLfloat v0);
	void setUniform2f(const char* varname, GLfloat v0, GLfloat v1);
	void setUniform3f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2);
	void setUniform4f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
	void setUniform1i(const char* varname, GLint v0);
	void setUniform2i(const char* varname, GLint v0, GLint v1);
	void setUniform3i(const char* varname, GLint v0, GLint v1, GLint v2);
	void setUniform4i(const char* varname, GLint v0, GLint v1, GLint v2, GLint v3);
	void setUniform1ui(const char* varname, GLuint v0);
	void setUniform2ui(const char* varname, GLuint v0, GLuint v1);
	void setUniform3ui(const char* varname, GLuint v0, GLuint v1, GLuint v2);
	void setUniform4ui(const char* varname, GLuint v0, GLuint v1, GLuint v2, GLuint v3);
	void setUniform1fv(const char* varname, GLsizei count, const GLfloat *value);
	void setUniform2fv(const char* varname, GLsizei count, const GLfloat *value);
	void setUniform3fv(const char* varname, GLsizei count, const GLfloat *value);
	void setUniform4fv(const char* varname, GLsizei count, const GLfloat *value);
	void setUniform1iv(const char* varname, GLsizei count, const GLint *value);
	void setUniform2iv(const char* varname, GLsizei count, const GLint *value);
	void setUniform3iv(const char* varname, GLsizei count, const GLint *value);
	void setUniform4iv(const char* varname, GLsizei count, const GLint *value);
	void setUniform1uiv(const char* varname, GLsizei count, const GLuint *value);
	void setUniform2uiv(const char* varname, GLsizei count, const GLuint *value);
	void setUniform3uiv(const char* varname, GLsizei count, const GLuint *value);
	void setUniform4uiv(const char* varname, GLsizei count, const GLuint *value);
	void setUniformMatrix2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix2x3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix3x2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix2x4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix4x2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix3x4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
	void setUniformMatrix4x3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat *value);
private:
	GLuint programID;
	std::vector<Shader*> shaders;
	bool linked;
	//GLuint getUniformLocation(const char* name);
	GLuint getUniformLocation(std::string name);

	std::map<std::string, GLuint> uniformMap;
	std::map<std::string, GLuint> attribMap;
};

