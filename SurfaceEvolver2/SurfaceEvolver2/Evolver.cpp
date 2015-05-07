#include "Evolver.h"


Evolver::Evolver(Mesh* initMesh, int initItersUntilLambdaUpdate) :
	mesh(initMesh),
	lambda(0.0f),
	itersUntilLambdaUpdate(initItersUntilLambdaUpdate),
	updateCount(0)
{
}


Evolver::~Evolver()
{
}
