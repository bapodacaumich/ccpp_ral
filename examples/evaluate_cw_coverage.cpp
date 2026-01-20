/* space station inspection planning using optimal control problem
Copyright (C) 2026 Brandon Apodaca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
    */

#include "cuda_kernels.h"
#include "obs.hpp"
#include "triangle_struct.hpp"
#include "utils.hpp"
#include "viewpoint_struct.hpp"

#include <iomanip>
#include <iostream>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    std::vector<std::string> files = {
        "2m_global",
        "2m_local",
        "4m_global",
        "4m_local",
        "8m_global",
        "8m_local",
        "16m_global",
        "16m_local"
    };

    std::string dir = "../knot_ocp/cw_opt_packaged_";

    std::string folder = "var/";
    if (argc > 1) {
        folder = argv[1];
        folder += "/";
    }

    dir += folder;

    std::cout << "Computing coverage for files in " << dir << std::endl;

    std::cout << "condition, coverage" << std::endl;
    for (auto file : files) {
        std::vector<bool> coverage_per_face;
        std::string xfile = dir + file + ".csv";
        // std::cout << xfile << std::endl; // Debugging line

        try {
            float coverage = compute_coverage_file(xfile, coverage_per_face);
            std::cout << file << ", ";
            std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << coverage << std::endl;
        } catch (...) {
            std::cerr << "Error processing file: " << xfile << std::endl;
        }
        std::string savefile = "../visualization_python/coverage/ivt_" + folder + file + "_cov.csv";
        std::vector<std::vector<bool>> coverage_per_face_2d(1, coverage_per_face);
        saveCSVbool(savefile, coverage_per_face_2d);
    }
}
