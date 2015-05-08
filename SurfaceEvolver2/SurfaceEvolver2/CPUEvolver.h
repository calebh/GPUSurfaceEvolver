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
    uint2 rearrangeTri(uint3 tri, int pointIndex);
    
    void stepSimulation();
    
    float getArea();
    float getMeanNetForce();
    float getMeanCurvature();
    float getVolume();
    
    void outputPoints();
    void outputVolumeForces();
    void outputAreaForces();
    void outputNetForces();
    
    void calculateForces(int vertexIndex, int triangleOffset);
    
    float calculateAlpha();
    
    void displaceVertex(int vertexIndex);
    
    
    int vertexCount, triangleCount;
    
	std::vector<uint3> triangles;
    int *triangleCountPerVertex, *triangleOffset;
    uint2 *trianglesByVertex;
    float3 *points1, *points2, *areaForce, *volumeForce;
    float alpha;
};
    