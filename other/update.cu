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
 *      alpha: single float
 *      surfaceArea: single float
 * 
 */

__global__ void calculateForces(int* triangleCountPerVertex,
                           int* triangleOffset,
                           uint3* trianglesByVertex,
                           float3* points,
                           float3* areaForce,
                           float3* volumeForce)
{
    int vertexIndex = blockId.x;
    int threadIndex = threadId.x;
    if(threadIndex < trianglesPerVertex[vertexIndex]){
        int triangleIndex = triangleOffset[vertexIndex];
        uint3 tri = trianglesBy[triangleIndex + threadIndex];
        
        float3 x1 = points[blockIndex];
        float3 x2 = points[t.x];
        float3 x3 = points[t.y];
        float3 s1 = x2 - x1;
        float3 s2 = x3 - x2;
        
        // Remember to set to zero
        // atomic add?
        // UNSAFE!!!!
        areaForce[vertexIndex] += SIGMA/(4.0f*areas[t.z]) * (crossProduct(s2, crossProduct(s1, s2))));
               
        volumeForce[vertexIndex] += crossProduct(x2, x3)/6.0f;
        
    }
}

__global__ void calculateAlpha(int verticesCount, float* alpha,
                               float3* areaForce, float3* volumeForce)
{
    float sum1, sum2;
    for(int i = 0; i < verticesCount; i++){
        sum1+=dot(volumeForce[i], areaForce[i]);
        sum2+=dot(volumeForce[i], volumeForce[i]);
    }
    (*alpha) = sum1 / sum2;
}

__global__ void displaceVertices(float* alpha, 
                                 float lambda, 
                                 float3* points,
                                 float3* points2, 
                                 float3* areaForce,
                                 float3* volumeForce)
{
    int vertexIndex =  blockIdx.x*blockDim.x + threadIdx.x;
    points2[vertexIndex] = points[vertexIndex] + lambda*(areaForce[vertexIndex] - (*alpha)*volumeForce[vertexIndex]);
}

__global__ void calculateAreas(uint3* triangles, float3* points2, float* areas){
    int triangleIndex = blockIdx.x*blockDim.x + threadIdx.x;
    triangle t = triangles[triangleIndex];
    float3 s1 = points2[t.y] - points2[t.x];
    float3 s2 = points2[t.z] - points2[t.y];
    
    areas[triangleIndex] = length(crossProduct(s1, s2))/2;
}

__global__ void sumSurfaceArea(float* areas, float* surfaceArea){
    (*surfaceArea) = 0;
    for(int i=0; i < trianglesCount; i++){
        (*surfaceArea) += areas[i];
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