/*
 * Using arguments:
 *      ./a.out [-v] [-m (tetrahedron|icosahedron|jeep|face|file.toload)]
 *              [-cpu] [-n=#] [-i=#] [-o (Comma seperated list of output values)]
 * 
 *      -v   : activate visualization
 *      -m   : model type, one of tetrahedron, icosahedron, jeep, face, or a file name
 *      -n=# : quadricection count, only makes sense with tetrahedron?
 *      -i=# : iteration count, how many times update should be called
 *      -cpu : perform calculation on CPU
 *      -o   : output a JSON file that is a list of representations of a single iteration
 *              The single iteration will consist of all of the data specified
 *              Options: SurfaceArea, Volume, Force, Curvature, Points
 *              Example: ./a.out -o SurfaceArea, Volume
 *                      will output a list of 2 item lists, the first of which
 *                      will be the surface area at that iteration, the second
 *                      of which will be the volume.
 * 
 * I didn't actually implement any of the functionality, but the code in main will store the 
 * information in relevant variables.
 * 
 */

#include <iostream>
#include <cstdlib>

using namespace std;

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

// Some Mesh Types and OutputTypes, These should probable go somewhere else
enum MeshType { TETRAHEDRON, ICOSAHEDRON, JEEP, FACE, MESH_FILE };
enum OutputType { TOTAL_SURFACE_AREA, TOTAL_VOLUME, MEAN_NET_FORCE_MAG, MEAN_CURVATURE, POINTS, NONE };

int main(int argc, char** argv){
    
    MeshType meshType = TETRAHEDRON;
    bool visualization = false;
    int bisections = 100, iterations;
    char* meshFile = NULL;
    bool gpu = true;
    OutputType output[5] = { NONE };
    
    // Parsing!
    for(int i=1;i<argc;i++){
        if(argMatch(argv[i], "-v")){
            visualization = true;
        }else if(argMatch(argv[i], "-n=")){
            bisections = atoi(argv[i]+3);
        }else if(argMatch(argv[i], "-i=")){
            iterations = atoi(argv[i]+3);
        }else if(argMatch(argv[i], "-m")){
            i++;
            if(argMatch(argv[i], "tetrahedron")){
                meshType = TETRAHEDRON;
            }else if(argMatch(argv[i], "icosahedron")){
                meshType = ICOSAHEDRON;
            }else if(argMatch(argv[i], "jeep")){
                meshType = JEEP;
            }else if(argMatch(argv[i], "face")){
                meshType = FACE;
            }else{
                meshType = MESH_FILE;
                meshFile = argv[i];
            }
        }else if(argMatch(argv[i], "-cpu")){
            gpu = false;
        }else if(argMatch(argv[i], "-o")){
            int oIndex = 0;
            do{
                i++;
                if(argMatch(argv[i], "SurfaceArea")){
                    output[oIndex++] = TOTAL_SURFACE_AREA;
                }else if(argMatch(argv[i], "Volume")){
                    output[oIndex++] = TOTAL_VOLUME;
                }else if(argMatch(argv[i], "Force")){
                    output[oIndex++] = MEAN_NET_FORCE_MAG;
                }else if(argMatch(argv[i], "Curvature")){
                    output[oIndex++] = MEAN_CURVATURE;
                }else{
                    output[oIndex++] = POINTS;
                }
            } while(lastChar(argv[i]) == ',');
        }
    }
}