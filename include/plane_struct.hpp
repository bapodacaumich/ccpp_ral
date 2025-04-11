#ifndef PLANE_STRUCT_HPP
#define PLANE_STRUCT_HPP

#include "vec3_struct.hpp"

struct Plane {
    vec3 normal;
    vec3 point;
    Plane() {
        normal = vec3();
        point = vec3();
    }
    Plane(vec3 n, vec3 p) {
        normal = n/n.norm();
        point = p;
    }
};

#endif