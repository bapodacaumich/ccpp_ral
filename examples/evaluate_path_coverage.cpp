#include "cuda_kernels.h"
#include "obs.hpp"
#include "triangle_struct.hpp"
#include "utils.hpp"
#include "viewpoint_struct.hpp"

#include <iomanip>
#include <iostream>
#include <cmath>
#include <string>

/*
compute coverage for all paths in packaged_paths folder
save coverage data to pareto_front folder
*/

int main(int argc, char** argv) {
    std::vector<std::string> folders = {
        "pf_2m_global",
        "pf_2m_local",
        "pf_4m_global",
        "pf_4m_local",
        "pf_8m_global",
        "pf_8m_local",
        "pf_16m_global",
        "pf_16m_local"
    };

    std::vector<std::vector<float>> valid_weights = {
        {0.001,	0.1},
        {0.1,	100}, 
        {0.1,	100}, 
        {0.01,	10}, 
        {0.1,	1000}, 
        {0.001,	1}, 
        {0.001,	10},
        {0.001,	10}
    };

    // std::string orientation_folder = "knotpoint_mapped_orientations/";
    std::string orientation_folder = "face_oriented/";
    std::string packaged_folder = "packaged_paths_fo/";

    std::cout << "valid_weights size = " << valid_weights.size() << std::endl;

    bool compute_coverage = true;
    bool compute_fuelcost = false;
    bool compute_pathtime = false;
    bool pf_coverage = true;


    std::string dir = "../knot_ocp/";
    size_t idx = 0;
    for (auto folder : folders) {
        // std::cout << folder << ",";
        std::vector<std::vector<float>> path_cost;
        std::vector<std::vector<float>> path_time;
        folder += "/";

        std::vector<std::vector<float>> weights;

        loadCSV(dir + "pareto_front/" + orientation_folder + folder + "wcomb.csv", weights, 2, ' ');

        // std::vector<std::vector<bool>> coverages;
        std::vector<std::vector<float>> pf_coverages;
        for (size_t i = 0; i < weights.size(); i++) {
            float kw = static_cast<float>(weights[i][0]);
            float fw = static_cast<float>(weights[i][1]);
            if (pf_coverage) {
                std::vector<bool> coverage_per_face;
                std::string xfile = dir + packaged_folder + folder + "k_" + getnum(kw) + "_f_" + getnum(fw) + ".csv";
                // std::cout << "Computing coverage for " << xfile << " ";
                std::vector<float> coverage = {compute_coverage_file(xfile, coverage_per_face)};
                // std::cout << " coverage=" << std::to_string(coverage[0]) << std::endl;
                // std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << coverage[0] << std::endl;
                // coverages.push_back(coverage_per_face);
                pf_coverages.push_back(coverage);
                // std::vector<std::vector<bool>> coverage_data = {coverage_per_face};
                // std::string savefile = "../visualization_python/coverage/" + folder.substr(3,folder.length()-4) + "_cov.csv";
                // std::cout << "saving to " << savefile << std::endl;
                // saveCSVbool(savefile, coverage_data);

            } else if (kw == valid_weights[idx][0] && fw == valid_weights[idx][1]) {
                // std::cout << "Valid Weights: k=" << kw << " f=" << fw << std::endl;
                if (compute_coverage) {
                    std::vector<bool> coverage_per_face;
                    std::string xfile = dir + "packaged_paths_so/" + folder + "k_" + getnum(kw) + "_f_" + getnum(fw) + ".csv";
                    // std::cout << "Computing coverage for " << xfile << " ";
                    std::vector<float> coverage = {compute_coverage_file(xfile, coverage_per_face)};
                    // std::cout << " coverage=" << std::to_string(coverage[0]) << std::endl;
                    std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << coverage[0] << std::endl;
                    // coverages.push_back(coverage_per_face);
                    // path_coverage.push_back(coverage);
                    // std::vector<std::vector<bool>> coverage_data = {coverage_per_face};
                    // std::string savefile = "../visualization_python/coverage/" + folder.substr(3,folder.length()-4) + "_cov.csv";
                    // std::cout << "saving to " << savefile << std::endl;
                    // saveCSVbool(savefile, coverage_data);
                }

                if (compute_fuelcost) {
                    std::string ufile = folder + "k_" + getnum(kw) + "_f_" + getnum(fw) + "_U.csv";
                    std::string tfile = folder + "k_" + getnum(kw) + "_f_" + getnum(fw) + "_t.csv";
                    std::vector<float> cost = {compute_fuel_cost_path(ufile, tfile)};
                    path_cost.push_back(cost);
                }

                if (compute_pathtime) {
                    std::string xfile = folder + "k_" + getnum(kw) + "_f_" + getnum(fw) + "_t.csv";
                    std::vector<float> pathtime = {compute_pathtime_path(xfile)};
                    path_time.push_back(pathtime);
                }
            }
        }
        // std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << path_time[0][0] << std::endl;
        idx++;

        // std::vector<size_t> common_uncovered;
        // for (size_t i = 0; i < coverages[0].size(); i++) {
        //     bool is_common = true;
        //     for (size_t j = 0; j < coverages.size(); j++) {
        //         if (coverages[j][i]) {
        //             is_common = false;
        //             break;
        //         }
        //     }
        //     if (is_common) {
        //         common_uncovered.push_back(i);
        //     }
        // }
        // std::cout << "common uncovered idxs: ";
        // for (size_t i = 0; i < common_uncovered.size(); i++) {
        //     std::cout << common_uncovered[i] << "\n";
        // }

        if (compute_coverage) {
            std::string savefile = "../knot_ocp/pareto_front/" + orientation_folder + folder + "cov.csv";
            std::cout << "Saving to " << savefile << std::endl;
            saveCSV(savefile, pf_coverages);
        }

        // if (compute_fuelcost) {
        //     std::string savefile_cost = "../knot_ocp/pareto_front/" + folder + "cost.csv";
        //     std::cout << "Saving to " << savefile_cost << std::endl;
        //     saveCSV(savefile_cost, path_cost);
        // }

        // if (compute_pathtime) {
        //     std::string savefile_time = "../knot_ocp/pareto_front/" + folder + "time.csv";
        //     // std::cout << "Saving to " << savefile_time << std::endl;
        //     saveCSV(savefile_time, path_time);
        // }
    }
}