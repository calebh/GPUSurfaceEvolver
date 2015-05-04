/*
 * Data structures:
 *      points and points2: array of point coordinates indexed by vertex
 *      areaForce: array of forces indexed by vertex
 *      volumeForce: array of forces indexed by vertex
 *      triangleCountPerVertex: array of triangle indexes indexed by vertex
 *      trianglesByVertex: array of triangles indexed by index of first coordinate, 
 *              last component is an index into triangles
 *      triangles:array of triangles indexed by triangle
 *      areas: array of areas indexed by triangle
 *      d_alpha: single float
 *      d_surfaceArea: single float
 * 
 */

#include "math2.h"
#include <iostream>

#define TINY_AMOUNT 0.001
#define TEMP 0.001
#define MAX_LAMBDA_ITERATIONS 100
#define UPDATE_ITERATIONS 100
#define SIGMA 7

void calculateForces(int vertexIndex, 
                     int threadIndex,
                     int* triangleOffset,
                     uint3* trianglesByVertex,
                     float3* points,
                     float3* areaForce,
                     float3* volumeForce,
                     float3* areas
                    )
{
    int triangleIndex = triangleOffset[vertexIndex];
    uint3 tri = trianglesByVertex[triangleIndex + threadIndex];
    
    float3 x1 = points[vertexIndex];
    float3 x2 = points[tri.x];
    float3 x3 = points[tri.y];
    float3 s1 = x2 - x1;
    float3 s2 = x3 - x2;
    
    areaForce[vertexIndex] += SIGMA/(4.0f*areas[tri.z]) * cross(s2, cross(s1, s2));
            
    volumeForce[vertexIndex] += cross(x2, x3)/6.0f;
    
}

float calculateAlpha(int vertexCount,
                      float3* areaForce, float3* volumeForce)
{
    float sum1, sum2;
    for(int i = 0; i < vertexCount; i++){
        sum1+=dot(volumeForce[i], areaForce[i]);
        sum2+=dot(volumeForce[i], volumeForce[i]);
    }
    d_alpha = sum1 / sum2;
}


void displaceVertices(int vertexIndex, 
                      float alpha, float lambda, 
                      float3* points1, float3* points2, 
                      float3* areaForce,
                      float3* volumeForce)
{
    points2[vertexIndex] = points1[vertexIndex] + lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
}

void calculateAreas(uint3* triangles, float3* points2, float* areas){
    int triangleIndex = blockIdx.x*blockDim.x + threadIdx.x;
    triangle t = triangles[triangleIndex];
    float3 s1 = points2[t.y] - points2[t.x];
    float3 s2 = points2[t.z] - points2[t.y];
    
    areas[triangleIndex] = length(cross(s1, s2))/2;
}

void sumSurfaceArea(float* areas){
    d_surfaceArea = 0;
    for(int i=0; i < trianglesCount; i++){
        d_surfaceArea += areas[i];
    }
}


// Returns surface area
float moveVertices(float lambda, int vertexCount, int triangleCount,
                   int* triangleCountPerVertex, 
){
    for(int i=0; i < vertexCount; i++){
        for(int j=0, k = triangleCountPerVertex[i]; j < k; j++){
            calculateForces(i,j, triangleOffset, trianglesByVertex,
                            points1, areaForce, volumeForce, areas);
        }
    }
    
    float alpha = calculateAlpha(vertexCount, areaForce, volumeForce);
    for(int i=0; i < vertexCount; i++){
        displaceVertices(i,alpha, lambda, points1, points2,
                         areaForce, volumeForce);
    }
    for(int i=0; i < triangleCount; i++){
        calculateAreas(i);
    }
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
int main(){
    cout << "Hello world\n";
}