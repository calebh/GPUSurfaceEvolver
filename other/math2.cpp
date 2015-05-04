#include "math2.h"

float sq(float x){
    return x*x;
}


float length(float3 r){
    return sqrt(sq(r.x) + sq(r.y) + sq(r.z));
}

float3 cross(float3 a, float3 b){
    float3 toReturn = {a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z, 
                       a.x*b.y - a.y*b.x};
    return toReturn;
}

float dot(float3 a, float3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

float3 operator-(float3 a, float3 b){
    float3 toReturn = {a.x - b.x,
                       a.y - b.y,
                       a.z - b.z};
    return toReturn;
}
float3 operator+(float3 a, float3 b){
    float3 toReturn = {a.x + b.x,
                       a.y + b.y,
                       a.z + b.z};
    return toReturn;
}

float3 operator*(float a, float3 b){
    float3 toReturn = {a*b.x,
                       a*b.y,
                       a*b.z};
    return toReturn;
}

float3 operator/(float3 a, float b){
    float3 toReturn = {a.x/b,
                       a.y/b,
                       a.z/b};
    return toReturn;
}

void operator+=(float3& a, float3 b){
    a = a + b;
}