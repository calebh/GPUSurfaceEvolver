#ifndef MATHSTUFF
#define MATHSTUFF

#include <cmath>
#include <iostream>

float sq(float x);

typedef struct float3{
    float x, y, z;
} float3;

// typedef struct uint3{
//     unsigned x, y, z;
// } uint3;

typedef struct uint3{
    unsigned x, y;
    union{
        unsigned z;
        unsigned index;
    };
} uint3;

std::ostream& operator<<(std::ostream& out, const uint3& tri);

std::ostream& operator<<(std::ostream& out, const float3& point);


float length(float3 r);

float3 cross(float3 a, float3 b);

float dot(float3 a, float3 b);

float3 vector(float x, float y, float z);

float3 operator-(float3 a, float3 b);
float3 operator+(float3 a, float3 b);

float3 operator*(float a, float3 b);

float3 operator/(float3 a, float b);

void operator+=(float3& a, float3 b);

void operator-=(float3& a, float3 b);
    
    
#endif