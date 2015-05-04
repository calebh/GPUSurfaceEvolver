//Compile with:
// g++ -g update.cpp math2.cpp test.cpp

#include "update.h"

#include <iostream>

float3 points[20] = {{1.5, 1.5, 1.5}, {0.5, 0.5, 1.5}, {0.5, 1.5, 0.5},
                     {-0.5, -0.5, 1.5}, {-0.5, 0.5, 0.5}, {-0.5, 1.5, -0.5},
                     {-1.5, -1.5, 1.5}, {-1.5, -0.5, 0.5}, {-1.5, 0.5, -0.5},
                     {-1.5, 1.5, -1.5}, {1.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, 
                     {1.5, -0.5, -0.5}, {-0.5, -1.5, 0.5}, {0.5, -1.5, -0.5}, 
                     {1.5, -1.5, -1.5}, {-0.5, 0.5, -1.5}, {0.5, -0.5, -1.5}, 
                     {0.5, 0.5, -0.5}, {-0.5, -0.5, -0.5} };
uint3 triangles[36] = {{2, 0,1}, {4, 1,3}, {4, 2,1}, {5, 2,4}, {7, 3,6},
                       {7, 4,3}, {8, 4,7}, {8, 5,4}, {9, 5,8}, {10, 0,1},
                       {11, 1,3}, {11, 10,1}, {12, 10,11}, {13, 3,6}, 
                       {13, 11,3}, {14, 11,13}, {14, 12,11}, {15, 12,14},
                       {5, 9,16}, {18, 16,17}, {18, 5,16}, {2, 5,18}, 
                       {12, 17,15}, {12, 18,17}, {10, 18,12}, {10, 2,18}, 
                       {0, 2,10}, {8, 9,16}, {19, 16,17}, {19, 8,16}, 
                       {7, 8,19}, {14, 17,15}, {14, 19,17}, {13, 19,14}, 
                       {13, 7,19}, {6, 7,13}};

uint3 makeWeirdTri(uint3 tri, int pointIndex, int triIndex){
    if(tri.x == pointIndex){
        tri.x = tri.y;
        tri.y = tri.z;
    }else if(tri.y == pointIndex){
        tri.x = tri.z;
        tri.y = tri.x;
    }
    tri.z = triIndex;
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
    int offset = 0;
    int triCount = 0;
    for(int i=0; i < pointCount; i++){
        int triCount = 0;
        m->triangleOffset[i] = offset;
        for(int j=0; j<triangleCount; j++){
            if(triangles[j].x == i ||
               triangles[j].y == i ||
               triangles[j].z == i){
                m->trianglesByVertex[offset + triCount] = makeWeirdTri(triangles[i], i, j);
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
    delete[] m->points1;
    delete[] m->points2;
    delete[] m->areaForce;
    delete[] m->volumeForce;
    delete[] m->triangles;
    delete[] m->areas;
}
    
int main(){
    int pointCount = 20;
    int triangleCount = 36;
    MeshData m;
    generateMeshData(&m, triangles, points, pointCount, triangleCount);
    update(.1, &m);
    deleteMeshData(&m);
}