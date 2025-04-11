#include "utils.hpp"
#include "obs.hpp"

#include <iostream>
#include <vector>

int main() {
    // first load the obs file
    std::vector<OBS> obsVec;
    loadStationOBS(obsVec, 4);

    // save obs vector to obj file
    saveObjFile(obsVec, "../data/model_remeshed/station_remeshed.obj");

    // first load the obs file
    std::vector<OBS> obsVec_convex;
    loadConvexStationOBS(obsVec_convex, 4);

    // save obs vector to obj file
    saveObjFile(obsVec, "../data/model_convex/station_convex.obj");
    return 0;
}