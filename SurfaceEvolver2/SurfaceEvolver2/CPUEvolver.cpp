#include "CPUEvolver.h"

#define TWO_PI 6.28318530718

CPUEvolver::CPUEvolver(Mesh* m, int initItersUntilLambdaUpdate)
    : Evolver(m, initItersUntilLambdaUpdate)
{
    
    triangles = m->getTriangels();
    points1 = m->getVertices();
    vertexCount = vertex.size();
    triangleCount = triangles.size();
    
    points2 = new float3[vertexCount];
    triangleCountPerVertex = new int[vertexCount];
    trianglesByVertex = new uint2[triangleCount * 3];
    triangleOffset = new int[vertexCount];
    areaForce = new float3[vertexCount];
    volumeForce = new float3[vertexCount];
    int offset = 0;
    for(int i=0; i < vertexCount; i++){
        int triCount = 0;
        triangleOffset[i] = offset;
        for(int j=0; j<triangleCount; j++){
            if(triangles[j].x == i ||
               triangles[j].y == i ||
               triangles[j].z == i){
                trianglesByVertex[offset + triCount] = rearrangeTri(triangles[j], i);
                triCount++;
            }
        }
        m->triangleCountPerVertex[i] = triCount;
        offset += triCount;
    }
}

CPUEvolver::~CPUEvolver(){
    delete[] triangleCountPerVertex;
    delete[] triangleOffset;
    delete[] points2;
    delete[] areaForce;
    delete[] volumeForce;
    delete[] areas;
}

uint3 CPUEvolver::rearrangeTri(uint3 tri, int pointIndex){
    uint2 simpleTri;
    if(tri.x == pointIndex){
        simpleTri.x = tri.y;
        simpleTri.y = tri.z;
    }else if(tri.y == pointIndex){
        simpleTri.y = tri.x;
        simpleTri.x = tri.z;
    }else{
        simpleTri.x = tri.x;
        simpleTri.y = tri.y;
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
    getArea();
}

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
    if(mutateMesh)
        points1[vertexIndex] = points1[vertexIndex] + lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
    else
        points2[vertexIndex] = points1[vertexIndex] + lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
}    

float CPUEvolver::getArea(){
    float3* outPoints = (mutateMesh) ? points1 : points2;

    float surfaceArea = 0;
    for(int i=0; i < triangleCount; i++){
        uint3 t = triangles[i];
        float3 s1 = outPoints[t.y] - outPoints[t.x];
        float3 s2 = outPoints[t.z] - outPoints[t.y];
        surfaceArea += length(cross(s1, s2))/2;
    }
    return surfaceArea;
}

float CPUEvolver::getMeanNetForce(){
    float total = 0;
    for(int i=0; i < vertexCount; i++){
        total += length(areaForce[i] + volumeForce[i]);
    }
    return total/vertexCount;
}

// approximates mean curvature by using angular difference
float CPUEvolver::getMeanCurvature(){
    float3* outPoints = (mutateMesh) ? points1 : points2;

    for(int i=0; i < vertexCount; i++){
        int triangleOffset = triangleOffset[i];
        float totalAngle = 0,
              totalArea  = 0;
        for(int j=0; j < triangleCountPerVertex[i]; j++){
            uint2 tri = trianglesByVertex[triangleOffset + j];
            float3 u = outPoints[tri.x] - outPoints[i],
                   v = outPoints[tri.y] - outPoints[i];
            totalAngle += acos(dot(u, v) / sqrt(dot(u, u) + dot(v, v));
            totalArea  += length(cross(u, v))/2;
        }
        totalCurvature += (TWO_PI - totalAngle) / totalArea;
    }
    return totalCurvature / vertexCount;
}
float CPUEvolver::getVolume(){
    float3* outPoints = (mutateMesh) ? points1 : points2;

    float v = 0;
    for(int i=0; i < triangleCount; i++){
        uint3 t = triangles[i];
        v += dot(points2[t.x], cross(outPoints[t.y],outPoints[t.z]))/6.0f;
    }
    return v;
};

void CPUEvolver::outputPoints(){
    float3* outPoints = (mutateMesh) ? points1 : points2;
    
    if(mutateMesh){
    for(int i=0; i<vertexCount; i++){
        if(i>0)
            cout << ", ";
        cout << "[ " << outPoints[i].x 
             << ", " << outPoints[i].y 
             << ", " << outPoints[i].z << "]";
    }
}
void CPUEvolver::outputVolumeForces(){
    for(int i=0; i<vertexCount; i++){
        if(i>0)
            cout << ", ";
        cout << "[ " << volumeForce[i].x 
             << ", " << volumeForce[i].y 
             << ", " << volumeForce[i].z << "]";
    }
}
    
void CPUEvolver::outputAreaForces(){
    for(int i=0; i<vertexCount; i++){
        if(i>0)
            cout << ", ";
        cout << "[ " << areaForce[i].x 
             << ", " << areaForce[i].y 
             << ", " << areaForce[i].z << "]";
    }
}
    
void CPUEvolver::outputNetForces(){
    for(int i=0; i<vertexCount; i++){
        if(i>0)
            cout << ", ";
        cout << "[ " << volumeForce[i].x + areaForce[i].x
             << ", " << volumeForce[i].y + areaForce[i].y
             << ", " << volumeForce[i].z + areaForce[i].z << "]";
    }
}
    


    