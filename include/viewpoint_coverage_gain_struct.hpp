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


#ifndef VIEWPOINT_COVERAGE_GAIN_STRUCT_HPP
#define VIEWPOINT_COVERAGE_GAIN_STRUCT_HPP

#include "viewpoint_struct.hpp"
#include <vector>

struct VPCoverageGain {
    Viewpoint vp;
    float gain;
    std::vector<bool> coverage;
    size_t vp_map_idx;
    float inc_angle;
    bool redundant;
};

#endif // VIEWPOINT_COVERAGE_GAIN_STRUCT_HPP
