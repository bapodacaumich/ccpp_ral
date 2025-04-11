#include "limit_struct.hpp"
#include "rrtz.hpp"
#include "station.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    // argc is the number of arguments passed to the program
    // argv is an array of strings containing the arguments

    // check for correct number of arguments
    if (argc != 1 && argc != 2) {
        std::cout << "Usage: ./rrtz or ./rrtz max_nodes" << std::endl;
        return 1;
    }

    size_t max_nodes = 500;
    if (argc == 2) {
        std::cout << "Running with max nodes=" << argv[1] << std::endl;
        max_nodes = std::stoi(argv[1]);
    }

    // create start and goal
    vec3 start = vec3(-2.0f, 0.0f, 0.0f);
    vec3 goal = vec3(2.0f, 0.0f, 0.0f);
    std::vector<vec3> path;
    // print out the start, goal, and max_nodes
    std::cout << "Solving cube: start=" << start.toString() << " goal=" << goal.toString() << " max_nodes=" << max_nodes << std::endl;

    Limit limits = { -6.0f, 6.0f, -6.0f, 6.0f, -6.0f, 6.0f };
    std::vector<OBS> obsVec;
    loadCubeOBS(obsVec);

    // set up RRTZ
    RRTZ rrtz = RRTZ(start, goal, obsVec, limits, max_nodes);

    // run RRTZ:
    // if rrtz a solution, print it
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
    return 0;
}