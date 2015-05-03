#include "Device.h"

static void error_callback(int error, const char* description) {
	fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

Device::Device(int initialWidth, int initialHeight, bool fullscreen) :
	width(initialWidth),
	height(initialHeight)
{
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}
	GLFWmonitor* monitor;
	if (fullscreen) {
		monitor = glfwGetPrimaryMonitor();
	} else {
		monitor = NULL;
	}
	window = glfwCreateWindow(width, height, "Surface Evolver", monitor, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glewInit();
	glfwSetKeyCallback(window, key_callback);
	
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles with normals not towards camera
	//glEnable(GL_CULL_FACE);
}

Device::~Device() {
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool Device::run() {
	if (glfwWindowShouldClose(window)) {
		return false;
	}
	glViewport(0, 0, width, height);
	return true;
}

int Device::getWidth() {
	return width;
}

int Device::getHeight() {
	return height;
}

void Device::endScene() {
	glfwSwapBuffers(window);
	glfwPollEvents();
}

GLFWwindow* Device::getWindow() {
	return window;
}