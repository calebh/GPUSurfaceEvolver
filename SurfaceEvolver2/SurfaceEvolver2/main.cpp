#include "Mesh.h"
#include "Device.h"
#include "CameraNode.h"
#include "ModelNode.h"
#include "SceneManager.h"

int main(void) {
	int width = 800;
	int height = 600;
	bool fullscreen = false;

	// Device must be the very first thing created!
	Device device(width, height, fullscreen);
	
	Mesh jeep("models/dragon.ply");
	//Mesh jeep(127);
	
	SceneManager manager(&device);

	CameraNode camera(width, height);
	camera.getTransform().setTranslation(20.0f, 20.0f, 20.0f);
	manager.addNode(&camera);

	ModelNode mn;
	//mn.getTransform().setScale(0.025f, 0.025f, 0.025f);
	mn.getTransform().setScale(20.0f, 20.0f, 20.0f);
	mn.getTransform().setTranslation(0.0f, 0.0f, 0.0f);
	mn.setMesh(&jeep);
	manager.addNode(&mn);
	

	ShaderProgram geometryProgram;
	Shader geometryVertexShader("shaders/geometry_pass.vert", VERTEX);
	Shader geometryFragShader("shaders/geometry_pass.frag", FRAGMENT);
	geometryProgram.attachShader(&geometryVertexShader);
	geometryProgram.attachShader(&geometryFragShader);
	geometryProgram.link();
	manager.setGeometryProgram(&geometryProgram);

	while (device.run()) {
		if (glfwGetKey(device.getWindow(), GLFW_KEY_W)) {
			camera.getTransform().setTranslation(camera.getTransform().getTranslation() + camera.getLookVector());
		}
		else if (glfwGetKey(device.getWindow(), GLFW_KEY_S)) {
			camera.getTransform().setTranslation(camera.getTransform().getTranslation() + camera.getLookVector() * -1.0f);
		}

		glm::vec3 horizontal = glm::normalize(glm::cross(camera.getUp(), camera.getLookVector()));
		if (glfwGetKey(device.getWindow(), GLFW_KEY_A)) {
			camera.getTransform().setTranslation(camera.getTransform().getTranslation() + horizontal);
		}
		else if (glfwGetKey(device.getWindow(), GLFW_KEY_D)) {
			camera.getTransform().setTranslation(camera.getTransform().getTranslation() + horizontal * -1.0f);
		}

		manager.drawAll();
		device.endScene();
	}
}