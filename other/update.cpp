#include "update.h"
#define TINY_AMOUNT 0.0001
#define MAX_ITERATIONS 500
#define TEMP 0.1
#define UPDATE_ITERATIONS 10

void calculateForces(int blockIndex, int threadIndex){
    if(threadIndex < trianglesPerVertex[blockIndex]){
        int count = trianglesPerVertex[blockIndex];
        int trianglesIndex = triangleOffset[blockIndex];
        triangle t = vertexTriangles[trianglesIndex + threadIndex];
        float3 x1 = points[blockIndex];
        float3 x2 = points[t.p1];
        float3 x3 = points[t.p2];
        float3 s1 = x2 - x1;
        float3 s2 = x3 - x2;
        
        // Remember to set to zero
        areaForce[blockIndex] += SIGMA/(4.0f*areas[t.p3]) * (crossProduct(s2, crossProduct(s1, s2))));
               
        volumeForce[blockIndex] += crossProduct(x2, x3)/6.0f;
        
    }
}

void calculateAlpha(){
    float sum1, sum2;
    for(int i = 0; i < verticesCount; i++){
        sum1+=dot(volumeForce[i], areaForce[i]);
        sum2+=dot(volumeForce[i], volumeForce[i]);
    }
    alpha = sum1 / sum2;
}

void displaceVertices(float alpha, float lambda, int blockIndex,int threadIndex){
    int vertexIndex = blockIndex * threadsPerBlock?? + threadIndex;
    points2[vertexIndex] += lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
}

void calculateAreas(int blockIndex, int threadIndex){
    int triangleIndex = blockIndex * threadsPerBlock?? + threadIndex;
    triangle t = triangles[triangleIndex];
    float3 s1 = points2[t.p2] - points[t.p1];
    float3 s2 = points[t.p3] - points[t.p2];
    
    areas[triangleIndex] = mag(crossProduct(s1, s2))/2;
}

void sumSurfaceArea(){
    surfaceArea = 0;
    for(int i=0; i < trianglesCount; i++){
        surfaceArea += areas[i];
    }
}
// Returns surface area
float moveVertices(float lambda){
    calculateForces();
    calculateAlpha();
    float alpha = retrieveAlpha??();
    displaceVertices(alpha, lambda);
    calculateAreas();
    sumSurfaceArea();
    return retrieveSurfaceArea();
} 

float abs(float x){
    return (x < 0) ? -x : x;
}

float findLambda(float lambda){
    float delta = 0;
    int i=0;
    do{
        lambda += delta;
        float a1 = moveVertices(lambda);
        float a2 = moveVertices(lambda + TINY_AMOUNT);
        float slope = (a2-a1) / TINY_AMOUNT;
        delta = -TEMP*slope;
        i++;
    } while(abs(delta) > 0.01 && i < MAX_LAMBDA_ITERATIONS);
    return lambda;
}

float update(float lambda){
    lambda = findLambda(lambda);
    for(int i=0; i < UPDATE_ITERATIONS; i++){
        moveVertices(lambda);
    }
    return lambda;
}