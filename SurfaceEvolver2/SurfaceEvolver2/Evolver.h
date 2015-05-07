#include "Mesh.h"

#pragma once
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
	virtual void getArea() = 0;
	virtual void getMeanNetForce() = 0;
	virtual void getMeanCurvature() = 0;
	virtual void getVolume() = 0;
	Mesh* mesh;
	float lambda;
	int itersUntilLambdaUpdate;
	int updateCount;
};

