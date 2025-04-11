#ifndef TRIANGLE_STRUCT_HPP
#define TRIANGLE_STRUCT_HPP

#include "vec3_struct.hpp"

#include <iostream>
#include <string>

struct Triangle {
    vec3 a, b, c, n;
    size_t module_idx; // module index for module membership
    Triangle () : a(), b(), c(), n(), module_idx() {}

    Triangle (const Triangle& t) : a(t.a), b(t.b), c(t.c), n(t.n) {
        this->module_idx = t.module_idx;
    }

    Triangle (vec3 a, vec3 b, vec3 c) : a(a), b(b), c(c){
        this->n = (b - a).cross(c - a);
        (this->n).normalize();
        this->module_idx = 0;
    }

    Triangle (vec3 a, vec3 b, vec3 c, vec3 n): a(a), b(b), c(c), n(n) {
        this->module_idx = 0;
        (this->n).normalize();
    }

    Triangle (vec3 a, vec3 b, vec3 c, vec3 n, size_t module_idx): a(a), b(b), c(c), n(n){
        this->module_idx = module_idx;
        (this->n).normalize();
    }

    vec3 getCentroid() {
        return (this->a + this->b + this->c) / 3.0f;
    }

    float getArea() {
        vec3 ab = this->b - this->a;
        vec3 ac = this->c - this->a;
        return 0.5f * (ab.cross(ac)).norm();
    }

    std::string toString() {
        return "Triangle (v0, v1, v2, n): " + 
                this->a.toString() + ", " + 
                this->b.toString() + ", " + 
                this->c.toString() + ", " + 
                this->n.toString();
    }
};

#endif // TRIANGLE_STRUCT_HPP