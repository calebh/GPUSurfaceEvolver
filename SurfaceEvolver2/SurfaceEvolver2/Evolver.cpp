#include "Evolver.h"

using namespace std;

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

void Evolver::findLambda(){
    float delta = 0;
    int i=0;
    float temp = 0.0001;
    do{
        lambda += delta;
        stepSimulation();
        float a1 = getArea();
        lambda += TINY_AMOUNT;
        stepSimulation();
        float a2 = getArea();
        float slope = (a2-a1) / TINY_AMOUNT;
        
        delta = -temp*slope;
        temp*=.9;
        i++;
    } while(abs(delta) > LAMBDA_THRESHOLD && i < MAX_LAMBDA_ITERATIONS);

}

void Evolver::update(){
    if(updateCount % itersUntilLambdaUpdate == 0)
        findLambda();
    stepSimulation();
    updateCount++;
}

void Evolver::setOutputFormat(OutputType* format, int formatLength){
    this->format       = format;
    this->formatLength = formatLength
}

void Evolver::outputData(){
    for(int i=0; i < formatLength; i++){
        if(i>0) {
            cout << ", ";
        }
        switch(format[i]){
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
                outputVolumeForces();
                break;
            case NET_FORCES:
                outputVolumeForces();
                break;
            default:
                break;
        }
    }
    if(formatLength > 0)
        cout << endl;
}
    
    
