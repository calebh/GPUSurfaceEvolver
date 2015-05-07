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

#define TINY_AMOUNT           0.000001
#define TEMP                  0.0001
#define LAMBDA_THRESHOLD      0.001
#define MAX_LAMBDA_ITERATIONS 100
#define UPDATE_ITERATIONS     10
#define SIGMA                 1

#define POINT_OUTPUT
#define VOLUMEF_OUTPUT
#define AREAF_OUTPUT
// #define NETF_OUTPUT

#include "update.h"

using namespace std;

//calculates the force on an individual vertex due to an individual triangle
void calculateForces(int vertexIndex, 
                     int triangleIndex,
                     MeshData* m)
{
    int triangleIndexOffset = m->triangleOffset[vertexIndex];
    uint3 tri = m->trianglesByVertex[triangleIndexOffset + triangleIndex];
    
    float3 x1 = m->points1[vertexIndex];
    float3 x2 = m->points1[tri.x];
    float3 x3 = m->points1[tri.y];
    float3 s1 = x2 - x1;
    float3 s2 = x3 - x2;
    
    m->areaForce[vertexIndex] += SIGMA/2.0f * cross(s2, cross(s1, s2))/length(cross(s1,s2));
    m->volumeForce[vertexIndex] += cross(x3, x2)/6.0f;
    
}


//calculates the alpha parameter from all of the area and volume forces in the mesh
float calculateAlpha(int vertexCount,
                      float3* areaForce, float3* volumeForce)
{
    float sum1 = 0, sum2 = 0;
    for(int i = 0; i < vertexCount; i++){
        sum1+=dot(volumeForce[i], areaForce[i]);
        sum2+=dot(volumeForce[i], volumeForce[i]);
    }
    return sum1 / sum2;
}

//displaces vertices according to previously calculated values
void displaceVertices(int vertexIndex, 
                      float alpha, float lambda, 
                      float3* points1, float3* points2, 
                      float3* areaForce,
                      float3* volumeForce)
{
    points2[vertexIndex] = points1[vertexIndex] + lambda*(areaForce[vertexIndex] - alpha*volumeForce[vertexIndex]);
}

// calculates the area of the triangle at triangles[triangleIndex]
void calculateAreas(int triangleIndex, uint3* triangles, float3* points2, float* areas){
    uint3 t = triangles[triangleIndex];
    float3 s1 = points2[t.y] - points2[t.x];
    float3 s2 = points2[t.z] - points2[t.y];
    areas[triangleIndex] = length(cross(s1, s2)/2);
}

// Sums the areas calculated in calculateAreas
float sumSurfaceArea(int triangleCount, float* areas){
    float surfaceArea = 0;
    for(int i=0; i < triangleCount; i++){
        surfaceArea += areas[i];
    }
    return surfaceArea;
}

// Calculates the total volume of a mesh defined by an array of points, an array of triangles
// connecting those points, and the number of triangles in the mesh
float calculateVolume(float3* points, uint3* triangles, int triangleCount){
    float v = 0;
    for(int i=0; i < triangleCount; i++){
        uint3 t = triangles[i];
        v += dot(points[t.x], cross(points[t.y],points[t.z]))/6.0f;
    }
    return v;
}
    




// Does one iteration of surface evolution and returns the total surface area
float evolveSurface(float lambda, MeshData* m){
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

// Finds a nice lambda value through gradient descent
float findLambda(float lambda, MeshData* m){
    float delta = 0;
    int i=0;
    float temp = TEMP;
    do{
        lambda += delta;
        float a1 = evolveSurface(lambda, m);
        float a2 = evolveSurface(lambda + TINY_AMOUNT, m);
        float slope = (a2-a1) / TINY_AMOUNT;
        
        delta = -temp*slope;
        temp*=.9;
        i++;
    } while(abs(delta) > LAMBDA_THRESHOLD && i < MAX_LAMBDA_ITERATIONS);
   
    return lambda;
}

// Finds a nice lambda and then calls evolveSurface UPDATE_ITERATIONS times, possible printing out more data
// as well
float update(float lambda, MeshData* m, Output){
    lambda = findLambda(lambda, m);
    float3* tmpPoints = m->points2;
    
    #ifdef STARS
        cout << "*******************************\n\n";
    #endif
    // by doing this, m->points1 will be changed in each iteration
    m->points2 = m->points1;
    for(int i=0; i < UPDATE_ITERATIONS; i++){
        iterate(lambda, m);
        
        
        #ifdef FORCES_OUTPUT
            for(int i=0; i < m->vertexCount;i++){
                cout << m->areaForce[i] << " | " << m->volumeForce[i] << endl;
            }
        #endif
        #ifdef POINT_OUTPUT
            for(int i=0;i<m->vertexCount;i++){
                cout << m->points1[i].x << ", " << m->points1[i].y << ", " << m->points1[i].z << ", ";
                #ifdef VOLUMEF_OUTPUT
                    cout << m->volumeForce[i].x << ", " << m->volumeForce[i].y << ", " << m->volumeForce[i].z << ", ";
                #endif
                #ifdef AREAF_OUTPUT
                    cout << m->areaForce[i].x << ", " << m->areaForce[i].y << ", " << m->areaForce[i].z << ", ";
                #endif
                #ifdef NETF_OUTPUT
                    cout << m->areaForce[i].x + m->volumeForce[i].x << ", " << m->areaForce[i].y + m->volumeForce[i].y << ", " << m->areaForce[i].z + m->volumeForce[i].z<< ", ";
                #endif
            }
            cout << endl;
        #endif
        
        #ifdef VOLUME_OUTPUT
            cout << calculateVolume(m->points1, m->triangles, m->triangleCount) << endl;
        #endif
    }
    m->points2 = tmpPoints;
    return lambda;
}

