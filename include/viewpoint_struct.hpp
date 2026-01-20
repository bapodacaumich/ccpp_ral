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
