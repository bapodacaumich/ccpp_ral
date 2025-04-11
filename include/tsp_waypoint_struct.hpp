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