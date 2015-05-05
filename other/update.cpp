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

#define TINY_AMOUNT           0.0000001
#define TEMP                  0.00001
#define LAMBDA_THRESHOLD      0.001
#define MAX_LAMBDA_ITERATIONS 100
#define UPDATE_ITERATIONS     2
#define SIGMA                 7


#include "update.h"

using namespace std;

void calculateForces(int vertexIndex, 
                     int threadIndex,
                     MeshData* m)
{
    int triangleIndex = m->triangleOffset[vertexIndex];
    uint3 tri = m->trianglesByVertex[triangleIndex + threadIndex];
    
    float3 x1 = m->points1[vertexIndex];
    float3 x2 = m->points1[tri.x];
    float3 x3 = m->points1[tri.y];
    float3 s1 = x2 - x1;
    float3 s2 = x3 - x2;
    
    m->areaForce[vertexIndex] += SIGMA/2.0f * cross(s2, cross(s1, s2)/length(cross(s1,s2)));
            
    m->volumeForce[vertexIndex] += cross(x2, x3)/6.0f;
    
}

float calculateAlpha(int vertexCount,
                      float3* areaForce, float3* volumeForce)
{
    float sum1, sum2;
    for(int i = 0; i < vertexCount; i++){
        sum1+=dot(volumeForce[i], areaForce[i]);
        sum2+=dot(volumeForce[i], volumeForce[i]);
    }
    return sum1 / sum2;
}


void displaceVertices(int vertexIndex, 
                      float alpha, float lambda, 
                      float3* points1, float3* points2, 
                      float3* areaForce,
                      float3* volumeForce)
{
    points2[vertexIndex] = points1[vertexIndex] + lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
}

void calculateAreas(int triangleIndex, uint3* triangles, float3* points2, float* areas){
    uint3 t = triangles[triangleIndex];
    float3 s1 = points2[t.y] - points2[t.x];
    float3 s2 = points2[t.z] - points2[t.y];
    areas[triangleIndex] = length(cross(s1, s2)/2);
}

float sumSurfaceArea(int triangleCount, float* areas){
    float surfaceArea = 0;
    for(int i=0; i < triangleCount; i++){
        surfaceArea += areas[i];
    }
    return surfaceArea;
}




// Returns surface area
float iterate(float lambda, MeshData* m){
    for(int i=0; i < m->vertexCount; i++){
        m->areaForce[i] = vector(0,0,0);
        m->volumeForce[i] = vector(0,0,0);
        for(int j=0, k = m->triangleCountPerVertex[i]; j < k; j++){
            calculateForces(i,j, m);
        }
    }
    float alpha = calculateAlpha(m->vertexCount, m->areaForce, m->volumeForce);
    
    for(int i=0; i < m->vertexCount; i++){
        displaceVertices(i, alpha, lambda, m->points1, m->points2,
                         m->areaForce, m->volumeForce);
    }
    for(int i=0; i < m->triangleCount; i++){
        calculateAreas(i, m->triangles, m->points2, m->areas);
    }
    
    
    return sumSurfaceArea(m->triangleCount, m->areas);
} 


float findLambda(float lambda, MeshData* m){
    float delta = 0;
    int i=0;
    float temp = TEMP;
    do{
        lambda += delta;
        float a1 = iterate(lambda, m);
        float a2 = iterate(lambda + TINY_AMOUNT, m);
        float slope = (a2-a1) / TINY_AMOUNT;
        
        delta = -temp*slope;
        temp*=.9;
        i++;
    } while(abs(delta) > LAMBDA_THRESHOLD && i < MAX_LAMBDA_ITERATIONS);
   
    return lambda;
}

float update(float lambda, MeshData* m){
   lambda = findLambda(lambda, m);
    float3* tmpPoints = m->points2;
    
    // by doing this, m->points1 will be changed in each iteration
    m->points2 = m->points1;
    for(int i=0; i < UPDATE_ITERATIONS; i++){
//         iterate(lambda, m);
        cout << iterate(lambda, m) << ",\n";
    }
    m->points2 = tmpPoints;
    return lambda;
}

