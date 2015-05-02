#include "Math.cuh"

__device__ vector subtractPoints(point a, point b) {
	vector n = { a.x - b.x, a.y - b.y, a.z - b.z };
	return n;
}

__device__ vector crossProduct(vector u, vector v) {
	float nx = u.y*v.z - u.z*v.y;
	float ny = u.z*v.x - u.x*v.z;
	float nz = u.x*v.y - u.y*v.x;
	vector n = { nx, ny, nz };
	return n;
}