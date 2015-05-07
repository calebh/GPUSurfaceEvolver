//Compile with:
// g++ -g update.cpp math2.cpp test.cpp

#include "update.h"
#include "data.h"
#include <cstdlib>
#include <iomanip>

using namespace std;


#define UPDATES 10
#define BIG_DATA



uint3 rearrangeTri(uint3 tri, int pointIndex, int triIndex){
//     cout << pointIndex << ": " << tri << " -> ";
    if(tri.x == pointIndex){
        tri.x = tri.y;
        tri.y = tri.z;
    }else if(tri.y == pointIndex){
        tri.y = tri.x;
        tri.x = tri.z;
    }
//     cout << tri << endl;
    return tri;
}
    
void generateMeshData(MeshData* m, uint3* triangles, float3* points, 
                      int pointCount, int triangleCount
){
    m->triangles = triangles;
    m->points1 = points;
    m->points2 = new float3[pointCount];
    m->vertexCount = pointCount;
    m->triangleCount = triangleCount;
    m->triangleCountPerVertex = new int[pointCount];
    m->trianglesByVertex = new uint3[triangleCount * 3];
    m->triangleOffset = new int[pointCount];
    m->areaForce = new float3[pointCount];
    m->volumeForce = new float3[pointCount];
    m->areas = new float[triangleCount];
    int offset = 0;
    int triCount = 0;
    for(int i=0; i < pointCount; i++){
        int triCount = 0;
        m->triangleOffset[i] = offset;
        for(int j=0; j<triangleCount; j++){
            if(triangles[j].x == i ||
               triangles[j].y == i ||
               triangles[j].z == i){
                m->trianglesByVertex[offset + triCount] = rearrangeTri(triangles[j], i, j);
//                 cout << m->trianglesByVertex[offset + triCount] << endl;
                triCount++;
            }
        }
        m->triangleCountPerVertex[i] = triCount;
        offset += triCount;
    }
}
void deleteMeshData(MeshData* m){
    delete[] m->triangleCountPerVertex;
    delete[] m->triangleOffset;
    //delete[] m->points1;
    delete[] m->points2;
    delete[] m->areaForce;
    delete[] m->volumeForce;
    //delete[] m->triangles;
    delete[] m->areas;
}

float3 randomVector(){
    return vector(-1+(rand()%10)/5.0,-1+(rand()%10)/5.0,-1+(rand()%10)/5.0);
}
int main(){
    MeshData m;
    
// #ifdef BIG_DATA
//     generateMeshData(&m, triangles, points, pointCount, triangleCount);
// #else
//     generateMeshData(&m, triangles3, points3, pointCount3, triangleCount3);
// #endif
    generateMeshData(&m, icosaTris, icosaPoints, 42, 80);
    
//     for(int i=0;i<pointCount;i++){
//         points[i].z*=3;
//     }
    
    for(int i=0; i < UPDATES; i++)
        update(0, &m);
    /* // This bit outputs a thing that can be added to mathematica with relative ease
    cout << "points ={";
    for(int i=0;i<pointCount;i++){
        cout << points[i] << ", ";
    }
    cout << "};\nedges = {";
    for(int i=0; i<triangleCount; i++){
        cout << "UndirectedEdge[" << triangles[i].x + 1 << ", "
                                  << triangles[i].y + 1 << "], "
             << "UndirectedEdge[" << triangles[i].y + 1 << ", "
                                  << triangles[i].z + 1 << "], " 
            << "UndirectedEdge[" << triangles[i].x + 1 << ", "
                                  << triangles[i].z + 1 << "],\n";
    }
    */
    deleteMeshData(&m);
}