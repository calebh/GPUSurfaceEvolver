/*
 * CPUEvolver.h
 * 
 * The CPUEvolver contains the following data structures:
 * 
 * Each point is stored in a float3 vector representing the position
 * points1 and points2 are both arrays of every point in the mesh, and
 *      are of length vertexCount
 * Each triangle is stored in either a uint3 or uint2 (if one vertex is
 *      known), with each integer refering to a point by index within
 *      points1 or points2
 * vector<uint3> triangles is a list of every triangle in the mesh
 * int* triangleCountPerVertex is a list of length vertexCount that
 *      holds the number of triangles surrounding each vertex
 * int* triangleOffset is a list of length vertexCount that holds
 *      the offset of the first trianglearound each vertex within the
 *      trianglesByVertex array
 * uint2* trianglesByVertex is a list of length vertexCount that
 *      represents each triangle around each vertex. For a given vertex
 *      of index i, the triangles trianglesByVertex[triangleOffset[i]]
 *      through trianglesByVertex[triangleOffset[i] + 
 *                                triangleCountPerVertex[i]
 *                                - 1]
 *      hold the indices of the two other vertices that comprise all
 *      of the triangles surounding i. This is confusing, but it works.
 * float3* areaForce and float3* volumeForce are easier: they simply
 *      contain the area and volume force respectively of each point in
 *      the mesh.
 * 
 * This data types are echoed in GPUEvolver on the GPU.
 * 
 * Otherwise, this works more or less how you would expect the extension
 * of Evolver to work on the CPU, although it is designed to be similar
 * in manner to the GPU code so that the comparison between the two can 
 * be relatively cleanly made.
 * 
 */

#pragma once
#include "Evolver.h"
#include <vector>

#define SIGMA 1

class CPUEvolver :
    public Evolver
{
public:
    CPUEvolver(Mesh* m, int initItersUntilLambdaUpdate);
    ~CPUEvolver();

private:
    
    // given a triangle in the form of uint3 tri, return a uint2 made of
    // the vertices not equal to pointIndex that preserves the winding order
    // This is used for populating the trianglesByVertex array
    uint2 rearrangeTri(uint3 tri, int pointIndex);
    
    // Step simulation runs a single step of surface evolution

    void stepSimulation();
    
    float getArea();
    float getMeanNetForce();
    float getMeanCurvature();
    float getVolume();
    
    void outputPoints();
    void outputVolumeForces();
    void outputAreaForces();
    void outputNetForces();
    
    // calculates the forces due to volume and area of a specific
    // triangle on a specific vertex;
    void calculateForces(int vertexIndex, int triangleOffset);
    
    float calculateAlpha();
    
    // displaces an individual vertex
    void displaceVertex(int vertexIndex);
    
    
    int vertexCount, triangleCount;
    
    std::vector<uint3> triangles;
    int *triangleCountPerVertex, *triangleOffset;
    uint2 *trianglesByVertex;
    float3 *points1, *points2, *areaForce, *volumeForce;
    float alpha;
};
    