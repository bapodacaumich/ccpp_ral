#include "cost_matrix.hpp"
#include "cuda_kernels.h"
#include "triangle_struct.hpp"
#include "tsp.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"
#include "viewpoint_generator.hpp"
#include "viewpoint_struct.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    // argc is the number of arguments passed to the program
    // argv is an array of strings containing the arguments

    // check for correct number of arguments
    if (argc != 1 && argc != 2 && argc != 3) {
        std::cout << "Usage: ./run_ccpp or ./run_ccpp vgd or ./run_ccpp vgd -l" << std::endl;
        return 1;
    }

    float vgd = 4.0f;
    if (argc >= 2) {
        std::cout << "Running with VGD=" << argv[1] << std::endl;
        vgd = std::stof(argv[1]);
    }

    bool local = false;
    if (argc == 3 && std::string(argv[2]) == "-l") {
        local = true;
        std::cout << "Running with local" << std::endl;
    } else {
        std::cout << "Running with global" << std::endl;
    }

    std::string locality_str;
    if (local) {
        locality_str = "local";
    } else {
        locality_str = "global";
    }

    // bool compute_coverage = false;
    bool vx = false;
    std::string vx_str = "";
    if (vx) {vx_str = "_vx";}

    std::vector<OBS> obsVec;
    if (vx) {
        loadVxStationOBS(obsVec, 4);
    } else {
        loadStationOBS(obsVec, 4);
    }

    std::vector<OBS> convex_obsVec;


    ViewpointGenerator vg(obsVec, convex_obsVec, vgd);
    // std::string unfiltered_vp_file = "unfiltered_viewpoints_" + std::to_string(static_cast<int>(vgd)) + "m.csv";
    // vg.saveUnfilteredViewpoints(unfiltered_vp_file);
    std::string coverage_save_file = "station_remeshed_coverage_" + std::to_string(static_cast<int>(vgd)) + "m.csv";

    // // ***************** Assigning Module membership to existing coverage viewpoints *****************
    // std::vector<std::vector<float>> coverage_viewpoints_data;
    // loadCSV("../data/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_vp_set.csv", coverage_viewpoints_data, 7, ',');
    // std::vector<Viewpoint> coverage_viewpoints;
    // for (size_t i = 0; i < coverage_viewpoints.size(); i++) {
    //     coverage_viewpoints.push_back(Viewpoint(
    //         vec3(coverage_viewpoints_data[i][0], coverage_viewpoints_data[i][1], coverage_viewpoints_data[i][2]),
    //         vec3(coverage_viewpoints_data[i][3], coverage_viewpoints_data[i][4], coverage_viewpoints_data[i][5]),
    //         coverage_viewpoints_data[i][6]
    //     ));
    // }
    // std::cout << "coverage viewpoints length: " << coverage_viewpoints_data.size() << std::endl; 
    // std::cout << "assigning module membership" << std::endl;
    // vg.assignModuleMembership(coverage_viewpoints);
    // coverage_viewpoints_data.clear();
    // for (size_t i = 0; i < coverage_viewpoints.size(); i++) {
    //     std::vector<float> row;
    //     row.push_back(coverage_viewpoints[i].pose.x);
    //     row.push_back(coverage_viewpoints[i].pose.y);
    //     row.push_back(coverage_viewpoints[i].pose.z);
    //     row.push_back(coverage_viewpoints[i].viewdir.x);
    //     row.push_back(coverage_viewpoints[i].viewdir.y);
    //     row.push_back(coverage_viewpoints[i].viewdir.z);
    //     row.push_back(coverage_viewpoints[i].module_idx);
    //     coverage_viewpoints_data.push_back(row);
    // }

    // for (size_t i = 0; i < coverage_viewpoints_data.size(); i++) {
    //     for (size_t j = 0; j < coverage_viewpoints_data[i].size(); j++) {
    //         std::cout << coverage_viewpoints_data[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // ***************** Assigning Module Membership to Coverage Viewpoints End *****************

    std::cout << "Running Greedy.." << std::endl;
    std::string save_file;
    if (local) {
        save_file = "../data" + vx_str + "/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_local_vp_set.csv";
        std::cout << "Saving viewpoint data to: " << save_file << std::endl;
        // saveCSV(save_file, viewpoint_data_save);
        // saveCSV("../data/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_local_vp_set.csv", viewpoint_data_save);
    } else {
        save_file = "../data" + vx_str + "/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_global_vp_set.csv";
        std::cout << "Saving viewpoint data to: " << save_file << std::endl;
        // saveCSV(save_file, viewpoint_data_save);
        // saveCSV("../data/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_global_vp_set.csv", viewpoint_data_save);
    }
    // vg.getCoverageViewpoints(local, coverage_save_file, compute_coverage, vx, save_file);
    vg.getCoverageViewpoints(local, coverage_save_file, true, vx, "");
    // std::vector<std::vector<float>> viewpoint_data_save;
    // for (size_t i = 0; i < coverage_viewpoints.size(); i++) {
    //     std::vector<float> vp_data;
    //     vp_data.push_back(coverage_viewpoints[i].pose.x);
    //     vp_data.push_back(coverage_viewpoints[i].pose.y);
    //     vp_data.push_back(coverage_viewpoints[i].pose.z);
    //     vp_data.push_back(coverage_viewpoints[i].viewdir.x);
    //     vp_data.push_back(coverage_viewpoints[i].viewdir.y);
    //     vp_data.push_back(coverage_viewpoints[i].viewdir.z);
    //     vp_data.push_back(coverage_viewpoints[i].module_idx);
    //     viewpoint_data_save.push_back(vp_data);
    // }

    // save bool vector of final coverage of mesh faces
    std::vector<bool> final_coverage;
    vg.getFilteredCoverage(final_coverage);
    vg.missedCoverage();
    std::vector<std::vector<float>> final_coverage_data;
    for (size_t i = 0; i < final_coverage.size(); i++) {
        std::vector<float> coverage_data;
        coverage_data.push_back(final_coverage[i]);
        final_coverage_data.push_back(coverage_data);
    }
    if (local) {
        std::string save_file = "../data" + vx_str + "/coverage_viewpoint_sets/" + std::to_string(static_cast<int>(vgd)) + "m_local_coverage.csv";
        std::cout << "Saving final coverage data to: " << save_file << std::endl;
        saveCSV(save_file, final_coverage_data);
        // saveCSV("../data/coverage_viewpoint_sets/" + std::to_string(static_cast<int>(vgd)) + "m_local_coverage.csv", final_coverage_data);
    } else {
        std::string save_file = "../data" + vx_str + "/coverage_viewpoint_sets/" + std::to_string(static_cast<int>(vgd)) + "m_global_coverage.csv";
        std::cout << "Saving final coverage data to: " << save_file << std::endl;
        saveCSV(save_file, final_coverage_data);
        // saveCSV("../data/coverage_viewpoint_sets/" + std::to_string(static_cast<int>(vgd)) + "m_global_coverage.csv", final_coverage_data);
    }

    // Compute cost matrix for travelling salesman problem for all viewpoints
    // Viewpoint start = Viewpoint( vec3(1.8f, 4.7f, 2.7f), vec3(0.0f, 0.0f, -1.0f), 2); # for unscaled station

    // new start for scaled up station
    Viewpoint start = Viewpoint( vec3(-5.0f, -2.0f, 3.0f), vec3(1.0f, 0.0f, 0.0f), 2);
    size_t rrtz_iter = 2000;
    CostMatrix cm(rrtz_iter);
    std::cout << "loading viewpoints" << std::endl;
    std::string viewpoint_file;

    cm.loadViewpoints(
        "../data" + vx_str + "/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_vp_set.csv",
        start
    );

    // // LOADING
    // std::cout << "loading path matrix" << std::endl;
    // cm.loadPathMatrix("../data/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_path_matrix.csv");
    // std::cout << "loading simple cost matrix" << std::endl;
    // cm.loadSimpleCostMatrix("../data/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_simple_cost_matrix.csv");
    // std::cout << "loading cost matrix" << std::endl;
    // cm.loadCostMatrix("../data/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_cost_matrix.csv");

    // GENERATING
    std::cout << "generating paths" << std::endl;
    // cm.generatePathMatrixParallel();
    cm.generatePathMatrix();
    std::cout << "saving path matrix" << std::endl;
    cm.savePathMatrix("../data" + vx_str + "/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_path_matrix.csv");
    std::cout << "saving simple cost matrix" << std::endl;
    cm.saveSimpleCostMatrix("../data" + vx_str + "/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_simple_cost_matrix.csv");
    std::cout << "generating cost matrix" << std::endl;
    cm.generateCostMatrix();
    std::cout << "saving cost matrix" << std::endl;
    cm.saveCostMatrix("../data" + vx_str + "/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_cost_matrix.csv");

    // TSP tsp(cm);
    std::cout << "creating TSP object" << std::endl;
    TSP tsp(cm);
    // if (local) {
    //     tsp.reassignModuleMembership();
    // } else {
    if (!local) {
        tsp.globalOpt();
    }
    tsp.greedyInit();
    tsp.twoOpt();
    
    // get path
    std::vector<std::vector<float>> path;
    tsp.getPath(path);

    // view path
    std::cout << "Path:" << std::endl;
    for (size_t i = 0; i < path.size(); i++) {
        for (size_t j = 0; j < path[i].size(); j++) {
            std::cout << std::to_string(path[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    // save path
    if (local) {
        std::string save_file = "../data" + vx_str + "/ordered_viewpoints/" + std::to_string(static_cast<int>(vgd)) + "m_local.csv";
        std::cout << "Saving path to: " << save_file << std::endl;
        saveCSV(save_file, path);
    } else {
        std::string save_file = "../data" + vx_str + "/ordered_viewpoints/" + std::to_string(static_cast<int>(vgd)) + "m_global.csv";
        std::cout << "Saving path to: " << save_file << std::endl;
        saveCSV(save_file, path);
    }

    // return 0;
}