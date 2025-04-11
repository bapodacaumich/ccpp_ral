#ifndef VIEWPOINT_HPP
#define VIEWPOINT_HPP

#include "vec3_struct.hpp"

struct Viewpoint {
    vec3 pose;
    vec3 viewdir; // always unit vec
    size_t module_idx; // module index for module membership

    Viewpoint() : pose(0.0f, 0.0f, 0.0f), viewdir(1.0f, 0.0f, 0.0f) { 
        this->module_idx = 0;
        this->viewdir.normalize(); 
    }

    Viewpoint(vec3 p) : pose(p), viewdir(1.0f, 0.0f, 0.0f) { 
        this->module_idx = 0;
        this->viewdir.normalize(); 
    }

    Viewpoint(vec3 p, vec3 v) : pose(p), viewdir(v) { 
        this->module_idx = 0;
        this->viewdir.normalize(); 
    }

    Viewpoint(vec3 p, vec3 v, size_t module_idx) : pose(p), viewdir(v) { 
        this->module_idx = module_idx;
        this->viewdir.normalize(); 
    }
};

#endif