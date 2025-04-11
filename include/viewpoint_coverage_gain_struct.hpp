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