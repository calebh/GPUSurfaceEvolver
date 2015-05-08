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
	void update();
	void setOutputFormat(OutputType* format, int formatLength );
	void outputData();
        
protected:
	// Step simulation returns the area
	// If saveResult is true then the evolver should override the current mesh state
	// with the new one
	virtual float stepSimulation() = 0;
	virtual float getArea() = 0;
	virtual float getMeanNetForce() = 0;
	virtual float getMeanCurvature() = 0;
	virtual float getVolume() = 0;
        
	
	virtual void outputPoints() = 0;
        virtual void outputVolumeForces() = 0;
        virtual void outputAreaForces() = 0;
        virtual void outputNetForces() = 0;
        
	Mesh* mesh;
	float lambda;
        bool mutateMesh;
        
private:
	void findLambda();
	int itersUntilLambdaUpdate;
	int updateCount;
	OutputType* format;
	int formatLength;
};

