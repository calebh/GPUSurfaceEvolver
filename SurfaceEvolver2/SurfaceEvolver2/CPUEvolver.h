#pragma 
#include "Evolver.h"

class CPUEvolver :
    public Evolver
{
    CPUEvolver(Mesh* m, int initItersUntilLambdaUpdate);
    ~CPUEvolver();

private:
    void rearrangeTri(uint3 tri, int pointIndex);
    
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
    
    void displaceVertex(vertexIndex);
    
    
    int vertexCount, triangleCount;
    
    vector<float3>& points, triangles;
    int *triangleCountPerVertex, *triangleOffset;
    uint3 *trianglesByVertex, *triangles;
    float3 *points1, *points2, *areaForce, *volumeForce;
    float alpha;
};
    