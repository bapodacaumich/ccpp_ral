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
