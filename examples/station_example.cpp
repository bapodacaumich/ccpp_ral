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
    // vec3 start = vec3(-0.5f, -2.0f, -0.5f);
    vec3 start = vec3(0.574084f,6.236693f,6.799296f);
    // vec3 goal = vec3(1.0f, 4.0f, 0.0f);
    vec3 goal = vec3(-1.221587f,7.040967f,4.904668f);
    // vec3 offset = vec3(2.529f, 4.821, 2.591);
    // start += offset;
    // goal += offset;
    std::vector<vec3> path;
    solveStation(start, goal, path, max_nodes);
    return 0;
}