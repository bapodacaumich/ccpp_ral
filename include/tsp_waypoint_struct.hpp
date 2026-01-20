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

#ifndef TSP_WAYPOINT_STRUCT_HPP
#define TSP_WAYPOINT_STRUCT_HPP

#include <string>

struct TSPWaypoint {
    size_t vp_idx;
    size_t module_idx;
    std::string toString() {
        // return "vp_idx=" + std::to_string(vp_idx) + " module_idx=" + std::to_string(module_idx);
        return "(" + std::to_string(this->vp_idx) + ", " + std::to_string(this->module_idx) + ")";
    }
};

#endif // TSP_WAYPOINT_STRUCT_HPP
