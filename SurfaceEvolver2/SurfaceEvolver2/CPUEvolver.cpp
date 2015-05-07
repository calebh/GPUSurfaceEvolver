#include "CPUEvolver.h"

CPUEvolver::CPUEvolver(Mesh* m, int initItersUntilLambdaUpdate)
    : Evolver(m, initItersUntilLambdaUpdate)
{
    
    triangles = m->getTriangels();
    points1 = m->getVertices();
    vertexCount = vertex.size();
    triangleCount = triangles.size();
    
    points2 = new float3[pointCount];
    vertexCount = pointCount;
    triangleCount = triangleCount;
    triangleCountPerVertex = new int[pointCount];
    trianglesByVertex = new uint3[triangleCount * 3];
    triangleOffset = new int[pointCount];
    areaForce = new float3[pointCount];
    volumeForce = new float3[pointCount];
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
                triCount++;
            }
        }
        m->triangleCountPerVertex[i] = triCount;
        offset += triCount;
    }
}

CPUEvolver::~CPUEvolver();

uint3 CPUEvolver::rearrangeTri(uint3 tri, int pointIndex, int triIndex){
    if(tri.x == pointIndex){
        tri.x = tri.y;
        tri.y = tri.z;
    }else if(tri.y == pointIndex){
        tri.y = tri.x;
        tri.x = tri.z;
    }
    return tri;
}

void CPUEvolver::stepSimulation(){
    for(int i=0; i < m->vertexCount; i++){
        areaForce[i] = vector(0,0,0);
        volumeForce[i] = vector(0,0,0);
        for(int j=0, k = triangleCountPerVertex[i]; j < k; j++){
            calculateForces(i,j);
        }

    }
    float alpha = calculateAlpha();
    for(int i=0; i < vertexCount; i++){
        displaceVertices(i);
    }
}
    

float CPUEvolver::getArea();
float CPUEvolver::getMeanNetForce();
float CPUEvolver::getMeanCurvature();
float CPUEvolver::getVolume();

void CPUEvolver::outputPoints();
void CPUEvolver::outputVolumeForces();
void CPUEvolver::outputAreaForces();
void CPUEvolver::outputNetForces();

void CPUEvolver::calculateForces(int vertexIndex, int triangleOffset){
    int triangleIndexOffset = triangleOffset[vertexIndex];
    uint3 tri = trianglesByVertex[triangleIndexOffset + triangleIndex];
    
    float3 x1 = points1[vertexIndex];
    float3 x2 = points1[tri.x];
    float3 x3 = points1[tri.y];
    float3 s1 = x2 - x1;
    float3 s2 = x3 - x2;
    
    areaForce[vertexIndex] += SIGMA/2.0f * cross(s2, cross(s1, s2))/length(cross(s1,s2));
    volumeForce[vertexIndex] += cross(x3, x2)/6.0f;
}
    

float CPUEvolver::calculateAlpha(){
    float sum1 = 0, sum2 = 0;
    for(int i = 0; i < vertexCount; i++){
        sum1+=dot(volumeForce[i], areaForce[i]);
        sum2+=dot(volumeForce[i], volumeForce[i]);
    }
    return sum1 / sum2;
}

void CPUEvolver::displaceVertex(vertexIndex){
     points2[vertexIndex] = points1[vertexIndex] + lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
}
    