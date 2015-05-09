/*
 * Evolver.cpp
 * 
 * A certain amount of the evolver must be performed on the CPU; this file
 * does those bits,includeing: initializing values in the constructor, 
 * finding a new lambda and updating, and outputting data (these all use calls
 * to other functions to be implementedin CPUEvolver or GPUEvolver).
 */

#include "Evolver.h"

using namespace std;

// Some constants that can be fiddled with
#define TINY_AMOUNT      0.01f
#define LAMBDA_THRESHOLD      0.001f
#define MAX_LAMBDA_ITERATIONS 100


// Initializa Data:
Evolver::Evolver(Mesh* initMesh, int initItersUntilLambdaUpdate) :
	mesh(initMesh),
	lambda(0.1f),
	itersUntilLambdaUpdate(initItersUntilLambdaUpdate),
	updateCount(0),
	mutateMesh(false)
{
}

Evolver::~Evolver()
{
}


/* 
 * Find lambda by doing a gradient-descent like algorithm.
 * The basic algorithm is to start with a lambda value and find
 * the approximate slope of the relationship between lambda
 * and surface area of an evolved mesh. Then, change lambda
 * according to that slope and the amount of iterations that 
 * have passed; eventually this should lead to a reasonably good 
 * lambda for minimizing the surface area.
 */
void Evolver::findLambda() {
    float delta = 0;
    int i=0;
    float temp = 0.0001f;
    do {
        lambda += delta;
        stepSimulation();
        float a1 = getArea();
        
        lambda += TINY_AMOUNT;
        stepSimulation();
        float a2 = getArea();
        
        lambda -= TINY_AMOUNT;
        
        float slope = (a2-a1) / TINY_AMOUNT;
        
        delta = -temp*slope;
        temp*=.9;
        i++;
    } while(abs(delta) > LAMBDA_THRESHOLD && i < MAX_LAMBDA_ITERATIONS);
}

// The update function is relaticely straight forward:
// if a certain amount of iterations have passed, find a new 
// lambda, and always just step the simulation.
// Mutate mesh is worth paying attention to, as it dictates whether
// or not a call to stepSimulation() will overwrite the original point data
void Evolver::update() {
    if (updateCount % itersUntilLambdaUpdate == 0) {
        mutateMesh = false;
        findLambda();
    }
    mutateMesh = true;
    stepSimulation();
    updateCount++;
}

// When given a list of things to output (in an array of OutputType enums
// of length formatLength), these functions will tell the evolver to 
// output those things.
void Evolver::setOutputFormat(OutputType* format, int formatLength) {
    this->format       = format;
    this->formatLength = formatLength;
}

void Evolver::outputData(){
    for (int i=0; i < formatLength; i++) {
        if (i>0) {
            cout << ", ";
        }
        switch (format[i]) {
            case TOTAL_SURFACE_AREA:
                cout << getArea();
                break;
            case TOTAL_VOLUME:
                cout <<  getVolume();
                break;
            case MEAN_NET_FORCE:
                cout << getMeanNetForce();
                break;
            case MEAN_CURVATURE:
                cout << getMeanCurvature();
                break;
            case POINTS:
                outputPoints();
                break;
            case VOLUME_FORCES:
                outputVolumeForces();
                break;
            case AREA_FORCES:
                outputAreaForces();
                break;
            case NET_FORCES:
                outputNetForces();
                break;
            default:
                break;
        }
    }
    if (formatLength > 0) {
            cout << endl;
    }
}

    
    
