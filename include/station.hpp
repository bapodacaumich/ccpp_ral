#ifndef STATION_HPP
#define STATION_HPP

#include "vec3_struct.hpp"
#include <vector>

bool solveStation(vec3 start, vec3 goal, std::vector<vec3>& path, size_t max_nodes);

#endif // STATION_HPP