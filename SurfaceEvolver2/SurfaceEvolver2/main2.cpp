#include "Mesh.h"
#include "ExternalMesh.h"
#include "Device.h"
#include "CameraNode.h"
#include "ModelNode.h"
#include "SceneManager.h"
#include "TetrahedronMesh.h"
#include "GPUEvolver.h"
#include "CPUEvolver.h"

#include <ctime>

enum MeshType { TETRAHEDRON, ICOSAHEDRON,
                MESH_FILE };


// Returns true if arg begins with match
bool argMatch(const char* arg, const char* match){
    //Wheee clean code! 
    for(;(*match)!='\0';match++)
        if((*match) != (*(arg++)))
            return false;
    return true;
}

//returns the last character of an argument
char lastChar(const char* s){
    int i = 0;
    while(s[i++] != '\0');
    return s[i-2];
}


int main(int argc, char** argv){
    
	int width = 800;
	int height = 600;
	bool fullscreen = false;

	// Device must be the very first thing created!
	Device device(width, height, fullscreen);

    //Default parameters
    
    MeshType meshType = TETRAHEDRON;
    // If meshType is MESH_FILE, load file from mesh stored in meshFile
    char* meshFile = NULL;
    
    // If true, run in visual mode. In this, all calculations are done on the GPU.
    // Otherwise, you get a pretty boring CSV outputted.
    bool visualization = false;
    
    // Number of times the shape is bisected (any number for tetrahedra, preset numbers for
    // icosahedra, and it means nothing for other meshes)
    int idealTriangleCount = 400;
    
    // Times Update is called if running in non visual mode
    int iterations = 10;
    
    // If true, use the GPU for calculations. Else, use the CPU
    bool gpu = true;
    
    //Type of stuff to output
    OutputType output[8];
    int outputLength = 0;
    
	int scale = 5;

	bool timing = false;

    // Parse Command Line Arguments:
    for(int i=1;i<argc;i++){
        if(argMatch(argv[i], "-v")){
            visualization = true;
        }else if(argMatch(argv[i], "-n=")){
            idealTriangleCount = atoi(argv[i]+3);
        }else if(argMatch(argv[i], "-i=")){
            iterations = atoi(argv[i]+3);
        }else if(argMatch(argv[i], "-iterations=")){
            iterations = atoi(argv[i]+12);
        }else if(argMatch(argv[i], "-m")){
            i++;
            if(argMatch(argv[i], "tetra")){
                meshType = TETRAHEDRON;
            }else if(argMatch(argv[i], "icosa")){
                meshType = ICOSAHEDRON;
            }else if(argMatch(argv[i], "jeep")){
                meshType = MESH_FILE;
                meshFile = "models/jeep.mtl";
            }else if(argMatch(argv[i], "dragon")){
                meshType = MESH_FILE;
                meshFile = "models/dragon.ply";
            }else{
                meshType = MESH_FILE;
                meshFile = argv[i];
            }
        }else if(argMatch(argv[i], "-cpu")){
            gpu = false;
		}
		else if (argMatch(argv[i], "-s=")){
			scale = atoi(argv[i] + 3);
		}
		else if (argMatch(argv[i], "-t")){
			timing = true;
		}
		else if (argMatch(argv[i], "-o")){
            do{
                i++;
                if(argMatch(argv[i], "SurfaceArea")){
                    output[outputLength++] = TOTAL_SURFACE_AREA;
                }else if(argMatch(argv[i], "Volume")){
                    output[outputLength++] = TOTAL_VOLUME;
                }else if(argMatch(argv[i], "MeanNetForce")){
                    output[outputLength++] = MEAN_NET_FORCE;
                }else if(argMatch(argv[i], "Curvature")){
                    output[outputLength++] = MEAN_CURVATURE;
                }else if(argMatch(argv[i], "AreaForces")){
                    output[outputLength++] = AREA_FORCES;
                }else if(argMatch(argv[i], "VolumeForces")){
                    output[outputLength++] = VOLUME_FORCES;
                }else if(argMatch(argv[i], "NetForces")){
                    output[outputLength++] = NET_FORCES;
                }else if(argMatch(argv[i], "Points")){
                    output[outputLength++] = POINTS;
                }
            } while(lastChar(argv[i]) == ',');
        }
    }
    
    Mesh* mesh;
    
    switch(meshType){
        case TETRAHEDRON:
            mesh = new TetrahedronMesh(ceil(sqrt(idealTriangleCount/4.0)));
            break;
        case ICOSAHEDRON:
            if(idealTriangleCount <= 100){
                meshFile = "models/icosa1.obj";
            }else if(idealTriangleCount <= 1000){
                meshFile = "models/icosa2.obj";
            }else if(idealTriangleCount <= 10000){
                meshFile = "models/icosa3.obj";
            }else if(idealTriangleCount <= 100000){
                meshFile = "models/icosa4.obj";
            }else{
                meshFile = "models/icosa5.obj";
            }
            mesh = new ExternalMesh(meshFile);
            break;
        default:
            mesh = new ExternalMesh(meshFile);
            break;
    }
    
    // visualization only applies to GPU calculation, so:
    if(visualization){
        
    
		SceneManager manager(&device);
        CameraNode camera(&device, width, height);
        camera.getTransform().setTranslation(20.0f, 20.0f, 20.0f);
        manager.addNode(&camera);

        ModelNode mn;
        //mn.getTransform().setScale(0.025f, 0.025f, 0.025f);
        mn.getTransform().setScale(scale, scale, scale);
        mn.getTransform().setTranslation(0.0f, 0.0f, 0.0f);
        mn.setMesh(mesh);
        manager.addNode(&mn);

        ShaderProgram geometryProgram;
        Shader geometryVertexShader("shaders/geometry_pass.vert", VERTEX);
        Shader geometryFragShader("shaders/geometry_pass.frag", FRAGMENT);
        geometryProgram.attachShader(&geometryVertexShader);
        geometryProgram.attachShader(&geometryFragShader);
        geometryProgram.link();
        manager.setGeometryProgram(&geometryProgram);

        GPUEvolver evolver(mesh, 10);

        while (device.run()) {
            evolver.update();
			evolver.synchronizeToMesh();
            manager.drawAll();
            device.endScene();
        }
    }else{
        if(gpu){
            GPUEvolver evolver(mesh, 10);
            evolver.setOutputFormat(output, outputLength);
			std::clock_t    start;

			start = std::clock();
			// your test
			
            for(int i=0; i < iterations; i++){
                evolver.update();
                evolver.outputData();
            }
			if (timing){
				std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
			}
        }else{
            CPUEvolver evolver(mesh, 10);
            evolver.setOutputFormat(output, outputLength);
			std::clock_t    start;

			start = std::clock();
            for(int i=0; i < iterations; i++){
                evolver.update();
                evolver.outputData();
            }
			if (timing){
				std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
			}
        }
    }
	return 0;
}
    
    
    
    
                