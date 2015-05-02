void calculateForces(int blockIndex, int threadIndex);

void calculateAlpha();

void displaceVertices(float alpha, float lambda, int blockIndex,int threadIndex);

void calculateAreas(int blockIndex, int threadIndex);

void sumSurfaceArea();

float findSurfaceArea(float lambda);