#include "limit_struct.hpp"
#include "obs.hpp"
#include "rrtz.hpp"
#include "station.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"

#include <iostream>

bool solveStation(vec3 start, vec3 goal, std::vector<vec3>& path, size_t max_nodes) {
    // print out the start, goal, and max_nodes
    std::cout << "Solving station: start=" << start.toString() << " goal=" << goal.toString() << " max_nodes=" << max_nodes << std::endl;

    Limit limits = { -5.0f, 10.0f, -5.0f, 15.0f, -5.0f, 10.0f };
    std::vector<OBS> obsVec;
    loadConvexStationOBS(obsVec, 4.0f);

    std::cout << "Planning rrtz" << std::endl;
    RRTZ rrtz = RRTZ(start, goal, obsVec, limits, max_nodes);
    if (!rrtz.run(path)) {
        std::cout << "Failed to find a path." << std::endl;
        return false;
    } else {
        std::cout << "Found path:" << std::endl;
        for (size_t i = 0; i < path.size(); i++) {
            std::cout << path[i].toString() << std::endl;
        }
        return true;
    }
}