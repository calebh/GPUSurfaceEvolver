#include "TetrahedronMesh.h"

int** new2D(int size) {
	int** ary = new int*[size];
	for (int i = 0; i < size; ++i) {
		ary[i] = new int[size];
	}
	return ary;
}

void delete2D(int** ary, int size) {
	for (int i = 0; i < size; ++i) {
		delete[] ary[i];
	}
	delete[] ary;
}

TetrahedronMesh::TetrahedronMesh(const int size)
{
	int trianglesPerFace = size*size;
	numFaces = trianglesPerFace * 4;
	vertices.resize(4 + 6 * (size - 1) + 2 * (size - 2)*(size - 1));
	triangles.resize(4 * trianglesPerFace);
	float3 initialPos = { size / 2.0, size / 2.0, size / 2.0 };
	float3 deltaX = { 0, 1, -1 };
	float3 deltaY = { -1, -1, 0 };
	int** indices1 = new2D(size + 1);
	int** indices2 = new2D(size + 1);
	int** indices3 = new2D(size + 1);
	int** indices4 = new2D(size + 1);
	int coordCount = 0, triangleCount = 0;

	// PLane 1:
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			indices1[x][y] = coordCount;
			vertices[coordCount++] = currentPos;
			if (x > 0) {
				triangles[triangleCount++] = { indices1[x][y], indices1[x - 1][y], indices1[x - 1][y - 1] };
				if (x < y) {
					triangles[triangleCount++] = { indices1[x][y], indices1[x - 1][y - 1], indices1[x][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}

	// Plane 2:
	deltaX = { 1, 0, -1 };
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			if (x == 0){
				indices2[x][y] = indices1[x][y];
			}
			else{
				indices2[x][y] = coordCount;
				vertices[coordCount++] = currentPos;
			}
			if (x > 0) {
				triangles[triangleCount++] = { indices2[x][y], indices2[x - 1][y - 1], indices2[x - 1][y] };
				if (x < y) {
					triangles[triangleCount++] = { indices2[x][y], indices2[x][y - 1], indices2[x - 1][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}

	// plane 3
	initialPos = { -size / 2.0, size / 2.0, -size / 2.0 };
	deltaY = { 1, -1, 0 };
	deltaX = { 0, 1, 1 };
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			if (y == size){
				indices3[x][y] = indices2[size - x][size - x];
			}
			else if (x == y){
				indices3[x][y] = indices1[size - x][size - x];
			}
			else{
				indices3[x][y] = coordCount;
				vertices[coordCount++] = currentPos;
			}
			if (x > 0) {
				triangles[triangleCount++] = { indices3[x][y], indices3[x - 1][y], indices3[x - 1][y - 1] };
				if (x < y) {
					triangles[triangleCount++] = { indices3[x][y], indices3[x - 1][y - 1], indices3[x][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}

	//plane 4
	deltaX = { -1, 0, 1 };
	for (int y = 0; y <= size; y++){
		float3 currentPos = initialPos + deltaY * y;
		for (int x = 0; x < y + 1; x++){
			if (y == size){
				indices4[x][y] = indices2[size - x][size];
			}
			else if (x == y){
				indices4[x][y] = indices1[size - y][size];
			}
			else if (x == 0){
				indices4[x][y] = indices3[x][y];
			}
			else{
				indices4[x][y] = coordCount;
				vertices[coordCount++] = currentPos;
			}
			if (x > 0) {
				triangles[triangleCount++] = { indices4[x][y], indices4[x - 1][y - 1], indices4[x - 1][y] };
				if (x < y) {
					triangles[triangleCount++] = { indices4[x][y], indices4[x][y - 1], indices4[x - 1][y - 1] };
				}
			}
			currentPos += deltaX;
		}
	}
	initCudaBuffers();
}


TetrahedronMesh::~TetrahedronMesh()
{
}
