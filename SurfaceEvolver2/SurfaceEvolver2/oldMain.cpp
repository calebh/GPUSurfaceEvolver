// A previous version of the program

#include "Mesh.h"
#include "ExternalMesh.h"
#include "Device.h"
#include "CameraNode.h"
#include "ModelNode.h"
#include "SceneManager.h"
#include "TetrahedronMesh.h"
#include "GPUEvolver.h"

int oldMain(void) {
	int width = 800;
	int height = 600;
	bool fullscreen = false;

	// Device must be the very first thing created!
	Device device(width, height, fullscreen);
	
	ExternalMesh tetra("models/icosa4.obj");
	//Mesh jeep(127);
	//TetrahedronMesh tetra(10);
	
	SceneManager manager(&device);

	CameraNode camera(&device, width, height);
	camera.getTransform().setTranslation(20.0f, 20.0f, 20.0f);
	manager.addNode(&camera);

	ModelNode mn;
	//mn.getTransform().setScale(0.025f, 0.025f, 0.025f);
	//mn.getTransform().setScale(20.0f, 20.0f, 20.0f);
	mn.getTransform().setTranslation(0.0f, 0.0f, 0.0f);
	mn.setMesh(&tetra);
	manager.addNode(&mn);

	ShaderProgram geometryProgram;
	Shader geometryVertexShader("shaders/geometry_pass.vert", VERTEX);
	Shader geometryFragShader("shaders/geometry_pass.frag", FRAGMENT);
	geometryProgram.attachShader(&geometryVertexShader);
	geometryProgram.attachShader(&geometryFragShader);
	geometryProgram.link();
	manager.setGeometryProgram(&geometryProgram);

	GPUEvolver evolver(&tetra, 20);

	while (device.run()) {
		for (int i = 0; i < 10; i++) {
			evolver.update();
		}
		manager.drawAll();
		device.endScene();
	}
	return 0;
}