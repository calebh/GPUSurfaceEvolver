#include "ShaderProgram.h"


ShaderProgram::ShaderProgram() :
	linked(false)
{
	programID = glCreateProgram();
}


ShaderProgram::~ShaderProgram()
{
}

void ShaderProgram::attachShader(Shader* sh) {
	shaders.push_back(sh);
	glAttachShader(programID, sh->getShaderHandle());
}

void ShaderProgram::link() {
	glLinkProgram(programID);
	std::cerr << "linking Shader\n";

	GLint infologLength = 0;

	glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infologLength);
	std::cerr << "Link Log Length " << infologLength << "\n";

	if (infologLength > 0) {
		char *infoLog = new char[infologLength];
		GLint charsWritten = 0;

		glGetProgramInfoLog(programID, infologLength, &charsWritten, infoLog);
		
		std::cerr << infoLog << std::endl;
		delete[] infoLog;
		glGetProgramiv(programID, GL_LINK_STATUS, &infologLength);
		if (infologLength == GL_FALSE) {
			std::cerr << "Program link failed exiting \n";
			exit(EXIT_FAILURE);
		}
	}
	linked = true;
}

GLuint ShaderProgram::getProgramID() {
	return programID;
}

void ShaderProgram::use() {
	glUseProgram(programID);
}

void ShaderProgram::vertexAttribPointer(std::string name, GLint size, GLenum type, GLsizei stride, const GLvoid* data, bool normalize) {
	GLint location;
	auto iter = attribMap.find(name);
	if (iter == attribMap.end()) {
		location = glGetAttribLocation(programID, name.c_str());
		attribMap[name] = location;
	} else {
		location = attribMap[name];
	}
	glEnableVertexAttribArray(location);
	glVertexAttribPointer(location, size, type, normalize, stride, data);
}

GLuint ShaderProgram::getUniformLocation(std::string name) {
	auto iter = uniformMap.find(name);
	if (iter == uniformMap.end()) {
		GLint loc = glGetUniformLocation(programID, name.c_str());
		uniformMap[name] = loc;
		return loc;
	} else {
		return iter->second;
	}
}

void ShaderProgram::setUniform1f(const char* varname, float v0) { glUniform1f(getUniformLocation(varname), v0); }
void ShaderProgram::setUniform2f(const char* varname, GLfloat v0, GLfloat v1) { glUniform2f(getUniformLocation(varname), v0, v1); }
void ShaderProgram::setUniform3f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2) { glUniform3f(getUniformLocation(varname), v0, v1, v2); }
void ShaderProgram::setUniform4f(const char* varname, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { glUniform4f(getUniformLocation(varname), v0, v1, v2, v3); }
void ShaderProgram::setUniform1i(const char* varname, GLint v0) { glUniform1i(getUniformLocation(varname), v0); }
void ShaderProgram::setUniform2i(const char* varname, GLint v0, GLint v1) { glUniform2i(getUniformLocation(varname), v0, v1); }
void ShaderProgram::setUniform3i(const char* varname, GLint v0, GLint v1, GLint v2) { glUniform3i(getUniformLocation(varname), v0, v1, v2); }
void ShaderProgram::setUniform4i(const char* varname, GLint v0, GLint v1, GLint v2, GLint v3) { glUniform4i(getUniformLocation(varname), v0, v1, v2, v3); }
void ShaderProgram::setUniform1ui(const char* varname, GLuint v0) { glUniform1ui(getUniformLocation(varname), v0); }
void ShaderProgram::setUniform2ui(const char* varname, GLuint v0, GLuint v1) { glUniform2ui(getUniformLocation(varname), v0, v1); }
void ShaderProgram::setUniform3ui(const char* varname, GLuint v0, GLuint v1, GLuint v2) { glUniform3ui(getUniformLocation(varname), v0, v1, v2); }
void ShaderProgram::setUniform4ui(const char* varname, GLuint v0, GLuint v1, GLuint v2, GLuint v3) { glUniform4ui(getUniformLocation(varname), v0, v1, v2, v3); }
void ShaderProgram::setUniform1fv(const char* varname, GLsizei count, const GLfloat* value) { glUniform1fv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform2fv(const char* varname, GLsizei count, const GLfloat* value) { glUniform2fv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform3fv(const char* varname, GLsizei count, const GLfloat* value) { glUniform3fv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform4fv(const char* varname, GLsizei count, const GLfloat* value) { glUniform4fv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform1iv(const char* varname, GLsizei count, const GLint* value) { glUniform1iv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform2iv(const char* varname, GLsizei count, const GLint* value) { glUniform2iv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform3iv(const char* varname, GLsizei count, const GLint* value) { glUniform3iv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform4iv(const char* varname, GLsizei count, const GLint* value) { glUniform4iv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform1uiv(const char* varname, GLsizei count, const GLuint* value) { glUniform1uiv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform2uiv(const char* varname, GLsizei count, const GLuint* value) { glUniform2uiv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform3uiv(const char* varname, GLsizei count, const GLuint* value) { glUniform3uiv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniform4uiv(const char* varname, GLsizei count, const GLuint* value) { glUniform4uiv(getUniformLocation(varname), count, value); }
void ShaderProgram::setUniformMatrix2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix2fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix3fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix4fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix2x3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix2x3fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix3x2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix3x2fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix2x4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix2x4fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix4x2fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix4x2fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix3x4fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix3x4fv(getUniformLocation(varname), count, transpose, value); }
void ShaderProgram::setUniformMatrix4x3fv(const char* varname, GLsizei count, GLboolean transpose, const GLfloat* value) { glUniformMatrix4x3fv(getUniformLocation(varname), count, transpose, value); }