
#include "cuda_kernels.h"
#include "obs.hpp"
#include "utils.hpp"
#include "viewpoint_struct.hpp"

int main() {
    // test the raytracing functions in cuda_kernels.h
    // test viewpoints:
    std::vector<Viewpoint> viewpoints;
    Viewpoint vp0 = Viewpoint(vec3(0.574084f,6.236693f,6.799296f), vec3(0.489475f,0.033094f,-0.871389f));
    viewpoints.push_back(vp0);

    // load station
    std::vector<OBS> obsVec;
    loadStationOBS(obsVec);

    // load triangles from station
    std::vector<Triangle*> all_faces;
    for (size_t obs_idx = 0; obs_idx < obsVec.size(); obs_idx++) {
        for (size_t face_idx = 0; face_idx < obsVec[obs_idx].faces.size(); face_idx++) {
            all_faces.push_back(&(obsVec[obs_idx].faces[face_idx]));
        }
    }

    // compute coverage map
    std::vector<std::vector<bool>> coverage_map;
    getCoverage(viewpoints, all_faces, coverage_map);
    // std::vector<std::vector<float>> inc_angle_map;
    // cuda_kernel_inc_angle(viewpoints, all_faces, inc_angle_map);

    // print coverage map
    std::cout << "Coverage Map:" << std::endl;
    for (size_t i = 0; i < coverage_map.size(); i++) {
        std::cout << "Viewpoint " << i << ": ";
        for (size_t j = 0; j < coverage_map[i].size(); j++) {
            std::cout << coverage_map[i][j] << ",";
        }
        std::cout << std::endl;
    }
}