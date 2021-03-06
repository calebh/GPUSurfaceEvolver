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

__device__ float d_alpha, d_surfaceArea;

__device__ int *d_triangleCountPerVertex, *d_triangleOffset;

__device__ float *d_areas;

__device__ uint3 *d_trianglesByVertex;

__device__ float3 *d_areaForce, *d_volumeForce;

__device__ unsigned int d_triangleCount, d_vertexCount

__global__ void calculateForces(float3* d_points){
    int vertexIndex = blockId.x;
    
    if(vertexIndex >= d_vertexCount){
        return;
    }
    
    int threadIndex = threadId.x;
    if(threadIndex < d_trianglesPerVertex[vertexIndex]){
        int triangleIndex = d_triangleOffset[vertexIndex];
        uint3 tri = d_trianglesByVertex[triangleIndex + threadIndex];
        
        float3 x1 = d_points[blockIndex];
        float3 x2 = d_points[t.x];
        float3 x3 = d_points[t.y];
        float3 s1 = x2 - x1;
        float3 s2 = x3 - x2;
        
        // Remember to set to zero
        // atomic add?
        
        __syncthreads();
        atomicAdd(&(d_areaForce[vertexIndex]),SIGMA/(4.0f*d_areas[t.z]) *
                                              cross(s2, cross(s1, s2)));
               
        atomicAdd(&(d_volumeForce[vertexIndex]), cross(x2, x3)/6.0f);
    }
}

__global__ void calculateAlpha(){
    float sum1, sum2;
    for(int i = 0; i < d_vertexCount; i++){
        sum1+=dot(d_volumeForce[i], d_areaForce[i]);
        sum2+=dot(d_volumeForce[i], d_volumeForce[i]);
    }
    d_alpha = sum1 / sum2;
}

float retrieveAlpha(){
    typeof(d_alpha) alpha;
    cudaMemcpyFromSymbol(&alpha, "d_alpha", sizeof(alpha),
                         0, cudaMemcpyDeviceToHost);
    return (float)alpha;
}

__global__ void displaceVertices(float lambda, 
                                 float3* d_points1,
                                 float3* d_points2)
{
    int vertexIndex =  blockIdx.x*blockDim.x + threadIdx.x;
    if(vertexIndex < vertexCount){
        d_points2[vertexIndex] = d_points1[vertexIndex] + 
                                 lambda*(d_areaForce[vertexIndex] -
                                         d_alpha*d_volumeForce[vertexIndex]);
    }
}

// Need to calculate areas before calling calculateForces for the first time!!
__global__ void calculateAreas(float3* points2){
    int triangleIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if(triangleIndex < d_triangleCount){
        uint3 t = d_triangles[triangleIndex];
        float3 s1 = d_points2[t.y] - d_points2[t.x];
        float3 s2 = d_points2[t.z] - d_points2[t.y];
        
        d_areas[triangleIndex] = length(cross(s1, s2))/2;
    }
}

__global__ void sumSurfaceArea(){
    d_surfaceArea = 0;
    for(int i=0; i < trianglesCount; i++){
        d_surfaceArea += d_areas[i];
    }
}

float retrieveSurfaceArea(){
    typeof(d_surfaceArea) surfaceArea;
    cudaMemcpyFromSymbol(&surfaceArea, "d_surfaceArea", 
                         sizeof(answer), 0, cudaMemcpyDeviceToHost);
    return (float)surfaceArea;
}

// Returns surface area
float moveVertices(float lambda, float3* d_points1, float3* d_points2, 
                   int vertexCount, int triangleCount, 
                   int maxTrianglesPerVertex){
    
    int blockCount_a, threadsPerBlock_a // blockCount_a * threadsPerBlock_a >= vertexCount
        blockCount_b, threadsPerBlock_b;// blockCount_b * threadsPerBlock_b >= triangleCount
    
    calculateForces<<< vertexCount, maxTrianglesPerVertex >>>(d_points1);
    calculateAlpha<<< 1, 1 >>>();
    
    displaceVertices<<< blockCount_a, threadsPerBlock_a >>>(lambda, d_points1, d_points2);
   
    calculateAreas<<< blockCount_b, threadsPerBlock_b >>>(d_points2);
    sumSurfaceArea<<< 1, 1 >>>();
    
    return retrieveSurfaceArea();
} 