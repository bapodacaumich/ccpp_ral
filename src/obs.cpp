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
#include "obs.hpp"
#include "node3d_struct.hpp"
#include "triangle_struct.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"

#include <vector>

OBS::OBS() {
    this->faces = std::vector<Triangle>();
    this->n_faces = 0;
}

OBS::OBS(std::vector<Triangle> faces) {
    this->faces = faces;
    this->n_faces = faces.size();
}

bool OBS::collision(vec3 point) {
    size_t num_collisions = 0;
    vec3 vec = vec3(1.0f, 0.0f, 0.0f);
    for (size_t i=0; i < this->n_faces; i++) {
        if (ray_int_triangle(point, vec, this->faces[i])) { num_collisions++; }
    }
    return num_collisions % 2 == 1;
}

bool OBS::collision(vec3 origin, vec3 end) {
    vec3 vec = end - origin;
    for (size_t i = 0; i < this->n_faces; i++) {
        if (ray_int_triangle(origin, vec, end, this->faces[i])) { return true; }
    }
    return false;
}
