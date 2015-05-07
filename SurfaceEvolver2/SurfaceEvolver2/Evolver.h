#include "Mesh.h"

#pragma once

enum OutputType { TOTAL_SURFACE_AREA, TOTAL_VOLUME, MEAN_NET_FORCE, MEAN_CURVATURE, POINTS, FORCES, NONE };

class Evolver
{
public:
	Evolver(Mesh* m, int initItersUntilLambdaUpdate);
	~Evolver();
	float findLambda();
	void update();
        
        
	void outputData();
        
private:
	virtual void stepSimulation() = 0;
	virtual float getArea() = 0;
	virtual float getMeanNetForce() = 0;
	virtual float getMeanCurvature() = 0;
	virtual float getVolume() = 0;
	Mesh* mesh;
	float lambda;
	int itersUntilLambdaUpdate;
	int updateCount;
};

