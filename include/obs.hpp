#ifndef OBS_HPP
#define OBS_HPP

#include "node3d_struct.hpp"
#include "triangle_struct.hpp"
#include <vector>

class OBS { // OBS = Obstacle
    public:
        // public members
        std::vector<Triangle> faces;
        OBS();
        OBS(std::vector<Triangle> faces);
        // bool collision(Node3D node);
        bool collision(vec3 point);
        bool collision(vec3 origin, vec3 end);
        // size_t get_n_faces();
    private:
        size_t n_faces;
};

#endif // OBS_HPP