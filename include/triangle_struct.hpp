/* space station inspection planning using optimal control problem
Copyright (C) 2026 Brandon Apodaca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
    */
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
