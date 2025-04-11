#include "cuda_kernels.h"

#include "utils.hpp"
#include "vec3_struct.hpp"
#include "viewpoint_struct.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

/* 
    * This file is used to evaluate the path saturation of the different pathfinding algorithms.
    * path saturation is the measure of 'how many frames' each face gets given a camera specification such as framerate
    * path saturation also measures the 'quality' of each frame by evaluating the incidence angle between the camera and the face
*/

int main(int argc, char** argv) {
    // build path data based on framerate

    if (argc < 2) {
        std::cout << "Usage: ./saturation <fps>" << std::endl;
        return 1;
    }

    float fps = std::stof(argv[1]);
    float dt = 1.0f / fps;

    std::vector<std::string> folders = {
        "cw_opt_packaged_10",
        "cw_opt_packaged_50",
        "cw_opt_packaged_var",
        "packaged_paths_ko/pf_final",
        "packaged_paths_ko_slerp/pf_final",
        "packaged_paths_ma/pf_final",
        "packaged_paths_ma_slerp/pf_final",
        "packaged_paths_so/pf_final",
        "packaged_paths_fo/pf_final"
        // "packaged_paths_ko_slerp/study_paths"
        // "packaged_paths_fo/study_paths"
    };

    std::vector<std::string> savefolders = {
        "ivt_10",
        "ivt_50",
        "ivt_var",
        "ocp_ko",
        "ocp_ko_slerp",
        "ocp_ma",
        "ocp_ma_slerp",
        "ocp_so",
        "ocp_fo"
        // "study_paths_ko_slerp"
        // "study_paths_fo"
    };

    std::vector<std::string> conditions = {
        "2m_global",
        "2m_local",
        "4m_global",
        "4m_local",
        "8m_global",
        "8m_local",
        "16m_global",
        "16m_local"
    };

    std::string dir = "../knot_ocp/";

    std::string savedir = "../visualization_python/saturation/";

    // iterate through each folder and each condition to evaluate path saturation
    for (size_t f = 0; f < folders.size(); f++) {
        std::string folder = folders[f];
        std::string savefolder = savefolders[f];
        for (std::string condition : conditions) {
            std::cout << "Evaluating path saturation for " << folder << " " << condition << std::endl;

            // first load path
            std::string pathfile = dir + folder + "/" + condition + ".csv";
            std::vector<std::vector<float>> path_data;
            loadCSV(pathfile, path_data, 7);

            // TODO: interpolate path to desired framerate
            std::vector<std::vector<float>> path_data_fps;
            float t = 0;
            for (size_t after_idx = 1; after_idx < path_data.size(); after_idx++) {
                size_t prev_idx = after_idx - 1;
                while (t < path_data[after_idx][6]) {
                    float alpha = (t - path_data[prev_idx][6]) / (path_data[after_idx][6] - path_data[prev_idx][6]);
                    vec3 viewdir = slerp(
                        vec3(path_data[prev_idx][3], path_data[prev_idx][4], path_data[prev_idx][5]),
                        vec3(path_data[after_idx][3], path_data[after_idx][4], path_data[after_idx][5]),
                        alpha
                    );
                    std::vector<float> path_step = {
                        std::lerp(path_data[prev_idx][0], path_data[after_idx][0], alpha),
                        std::lerp(path_data[prev_idx][1], path_data[after_idx][1], alpha),
                        std::lerp(path_data[prev_idx][2], path_data[after_idx][2], alpha),
                        viewdir.x,
                        viewdir.y,
                        viewdir.z,
                        t
                    };
                    path_data_fps.push_back(path_step);
                    t += dt;
                }
            }

            std::cout << path_data_fps.size() << " frames at " << fps << " fps" << std::endl;

            // compute saturation
            std::vector<std::vector<float>> saturation_map;
            std::vector<std::vector<size_t>> saturation_bins;
            size_t n_bins = 64;
            compute_saturation_path(path_data_fps, saturation_map, saturation_bins, n_bins);

            // save saturation map
            std::string savefile_sat = savedir + savefolder + "/" + condition + "_sat.csv";
            std::cout << "\nSaving saturation stats to " << savefile_sat << " ..." << std::endl;
            saveCSV(savefile_sat, saturation_map); // saturation = {viewing_time, avg_incidence_angle, min_incidence_angle}

            // save saturation binning
            std::string savefile_bin = savedir + savefolder + "/" + condition + "_sat_" + std::to_string(n_bins) + "_bins.csv";
            std::cout << "Saving saturation histogram binning to " << savefile_bin << " ... " << std::endl;
            saveCSVsizet(savefile_bin, saturation_bins); // saturation = {viewing_time, avg_incidence_angle, min_incidence_angle}
        }
        std::cout << "Done evaluating path saturation for " << folder << std::endl;
    }
    std::cout << "Done evaluating path saturation for all folders" << std::endl;
    return 0;
}