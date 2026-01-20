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
