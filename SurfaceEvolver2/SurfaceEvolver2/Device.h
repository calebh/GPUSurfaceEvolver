#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

class Device
{
public:
	Device(int initialWidth, int initialHeight, bool fullscreen);
	~Device();
	GLFWwindow* getWindow();
	bool run();
	int getWidth();
	int getHeight();
	void endScene();
private:
	GLFWwindow* window;
	int width;
	int height;
};

