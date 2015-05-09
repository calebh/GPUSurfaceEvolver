/*
 * Evolver.h
 * 
 * The Evolver class is extended into CPUEvolverand GPUEvolver, which are 
 * implementations of the same thing on the CPU and GPU respectively.
 * It takes a mesh, and upon calling update() will evolve that mesh
 * to minimize surface area while maintaining a constant volume. Evolution of the
 * surface also requires a parameter known as lambda, so every few iterations
 * it is necessary to use a gradient-descent like algorithm to find the best
 * lambda for minimizing the surface, so that is done in findLambda(), which is 
 * called every itersUntilLambdaUpdate iterations by update(). 
 * 
 * Additionally, the Evolver class can output data on the Mesh and it's evolution.
 * The specific things it can output are:
 * surface area, volume, average magnitude of the net force (called for the sake
 * of brevity MEAN_NET_FORCE), mean curvature estimateed with angle deficit, and
 * the following lists of vectors (with one vector per point in the mesh): 
 * points, area forces, volume forces, and net forces. See main.cpp for
 * instructions on how to configure the output with command line arguments.
 *
 */ 

#pragma once

#include "Mesh.h"

#include <iostream>

enum OutputType { TOTAL_SURFACE_AREA, TOTAL_VOLUME, MEAN_NET_FORCE,
                  MEAN_CURVATURE, POINTS, AREA_FORCES, VOLUME_FORCES,
                  NET_FORCES };

class Evolver
{
public:
        
    Evolver(Mesh* m, int initItersUntilLambdaUpdate);
    ~Evolver();
    
    // Calls stepSimulation and on occasional iterations, findLambda
    void update();
    
    // controls the output
    void setOutputFormat(OutputType* format, int formatLength );
    void outputData();
        
protected:
    // Step simulation runs a single step of surface evolution
    virtual void stepSimulation() = 0;
    virtual float getArea() = 0;
    
    // The following functions would be nice to implement,
    // but are not entirely necessary for the functioning of 
    // our program
    
    // Returns mean net force magnitude
    virtual float getMeanNetForce() = 0;
    
    // These are pretty self explanatory
    virtual float getMeanCurvature() = 0;
    virtual float getVolume() = 0;
    
    // These functions should output to stdout
    // They all print a 3-vector in the format "[x, y, z]"
    
    // outputs a list of points like
    virtual void outputPoints() = 0;
    // outputs a list of the volume force per vertex
    virtual void outputVolumeForces() = 0;
    // outputs a list of the area force per verex
    virtual void outputAreaForces() = 0;
    // outputs a list of the net force per vertex
    virtual void outputNetForces() = 0;
    
    Mesh* mesh;
    float lambda;
    // If this is true, then calls to stepSimulation will
    // overwrite the vertex data. Otherwise, stepSimulation
    // will write the output into a different buffer
    bool mutateMesh;
        
private:
    // finds a good lambda value
    void findLambda();
    
    // findLambda is called when updateCount % itersUntilLambdaUpdate == 0,
    // and update count is incremented in each call to update()
    int itersUntilLambdaUpdate;
    int updateCount;
    
    OutputType* format;
    int formatLength;
};

