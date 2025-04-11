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