#include "cuda_kernels.h"
#include "obs.hpp"
#include "triangle_coverage_struct.hpp"
#include "utils.hpp"
#include "viewpoint_coverage_gain_struct.hpp"
#include "viewpoint_generator.hpp"
#include "viewpoint_struct.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

ViewpointGenerator::ViewpointGenerator() {
    // default constructor -- useless to do this
    this->vgd = 2.0f;
    this->structure = std::vector<OBS>();
    this->convex_structure = std::vector<OBS>();
    this->unfiltered_viewpoints = std::vector<Viewpoint>();
    this->coverage_viewpoints = std::vector<Viewpoint>();
    this->coverage_map = std::vector<std::vector<bool>>();

    std::cout << "Warning: default constructor for ViewpointGenerator assumes no inspectable object and is useless" << std::endl;
}
// ViewpointGenerator::ViewpointGenerator(
//     std::vector<OBS> structure
// ) {

//     // initialize members
//     this->structure = structure;
//     this->vgd = 2.0f;
//     this->unfiltered_viewpoints = std::vector<Viewpoint>();
//     this->coverage_viewpoints = std::vector<Viewpoint>();
//     this->coverage_map = std::vector<std::vector<bool>>();

//     // initialize viewpoint generator
//     this->initialize();
// }

// ViewpointGenerator::ViewpointGenerator(
//     std::vector<OBS> structure,
//     float vgd
//     ) {

//     // initialize members
//     this->structure = structure;
//     this->vgd = vgd;
//     this->unfiltered_viewpoints = std::vector<Viewpoint>();
//     this->coverage_viewpoints = std::vector<Viewpoint>();
//     this->coverage_map = std::vector<std::vector<bool>>();

//     // initialize viewpoint generator
//     this->initialize();
// }

ViewpointGenerator::ViewpointGenerator(
    std::vector<OBS> structure,
    std::vector<OBS> convex_structure,
    float vgd,
    float inc_angle_max,
    float inc_improvement_minimum,
    float inc_improvement_threshold
    ) {

    // initialize members
    this->structure = structure;
    this->convex_structure = convex_structure;
    this->vgd = vgd;
    this->inc_angle_max = inc_angle_max;
    this->inc_improvement_threshold = inc_improvement_threshold;
    this->unfiltered_viewpoints = std::vector<Viewpoint>();
    this->coverage_viewpoints = std::vector<Viewpoint>();
    this->coverage_map = std::vector<std::vector<bool>>();

    // initialize viewpoint generator
    this->initialize();
}

void ViewpointGenerator::initialize() {
    // print incidence angle threshold
    std::cout << "Incidence Angle Threshold=" << this->inc_angle_max << std::endl;

    // count number of mesh faces
    std::cout << "Initializing Viewpoint Generator..." << std::endl;
    this->countMeshFaces();
    std::cout << "Number of mesh faces=" << this->num_mesh_faces << std::endl;

}

void ViewpointGenerator::printIncidenceAngles() {
    // print viewpoints and faces with corresponding indices, then print incidence angles organized wrt indices
    for (size_t i=0; i < this->unfiltered_viewpoints.size(); i++) {
        std::cout << "Viewpoint " << i << ": " << this->unfiltered_viewpoints[i].pose.toString() << std::endl;
    }

    for (size_t i=0; i < this->num_mesh_faces; i++) {
        std::cout << "Face " << i << ": " << this->all_faces[i]->toString() << std::endl;
    }

    // print inc_angles
    std::cout << "Incidence Angles:" << std::endl;
    for (size_t vp_idx = 0; vp_idx < this->unfiltered_viewpoints.size(); vp_idx++) {
        std::cout << "Viewpoint " << vp_idx << ": "; 
        for (size_t face_idx = 0; face_idx < this->num_mesh_faces; face_idx++) {
            std::cout << this->inc_angle_map[vp_idx][face_idx] << ", ";
        }
        std::cout << std::endl;
    }
}

void ViewpointGenerator::printCoverageMap() {
    std::cout << "Coverage Map:" << std::endl;
    for (size_t i = 0; i < this->coverage_map.size(); i++) {
        std::cout << "Viewpoint " << i << ": ";
        for (size_t j = 0; j < this->coverage_map[i].size(); j++) {
            std::cout << this->coverage_map[i][j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void ViewpointGenerator::saveCoverageMap(const std::string& filename, bool vx) {
    // save coverage map to file
    // std::vector<std::vector<float>> data;
    // for (size_t i = 0; i < this->coverage_map.size(); i++) {
    //     std::vector<float> row;
    //     for (size_t j = 0; j < this->coverage_map[i].size(); j++) {
    //         row.push_back(this->coverage_map[i][j] ? 1.0f : 0.0f);
    //     }
    //     data.push_back(row);
    // }
    if (vx){
        saveCSVbool("../data_vx/coverage_maps/" + filename, this->coverage_map);
    } else {
        saveCSVbool("../data/coverage_maps/" + filename, this->coverage_map);
    }
}

void ViewpointGenerator::loadCoverageMap(const std::string& filename, bool vx) {
    // assume right coverage_map - viewpoints - faces are loaded
    // load coverage map from file
    std::vector<std::vector<float>> data;
    if (vx) {
        loadCSVbool("../data_vx/coverage_maps/" + filename, this->coverage_map);
        // loadCSV("../data_vx/coverage_maps/" + filename, data, this->num_mesh_faces);
    } else {
        loadCSV("../data/coverage_maps/" + filename, data, this->num_mesh_faces);
        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[i].size(); j++) {
                row.push_back(data[i][j] == 1.0f);
            }
            this->coverage_map.push_back(row);
        }
    }
}

void ViewpointGenerator::saveUnfilteredViewpoints(std::string& filename, bool vx) {
    // save unfiltered viewpoints with updated module memberships
    this->reassignModuleMembership();
    std::vector<std::vector<float>> data;
    for (size_t i = 0; i < this->unfiltered_viewpoints.size(); i++) {
        std::vector<float> row;
        row.push_back(this->unfiltered_viewpoints[i].pose.x);
        row.push_back(this->unfiltered_viewpoints[i].pose.y);
        row.push_back(this->unfiltered_viewpoints[i].pose.z);
        row.push_back(this->unfiltered_viewpoints[i].viewdir.x);
        row.push_back(this->unfiltered_viewpoints[i].viewdir.y);
        row.push_back(this->unfiltered_viewpoints[i].viewdir.z);
        row.push_back(this->unfiltered_viewpoints[i].module_idx);
        data.push_back(row);
    }
    if (vx) {
        saveCSV("../data_vx/unfiltered_viewpoints/" + filename, data);
    } else {
        saveCSV("../data/unfiltered_viewpoints/" + filename, data);
    }
}

void ViewpointGenerator::reassignModuleMembership() {
    // reassign viewpoint module membership to four modules
    std::vector<size_t> module_membership = {0,3,2,3,2,2,2,1,1,3};
    for (size_t i = 0; i < this->unfiltered_viewpoints.size(); i++) {
        if (this->unfiltered_viewpoints[i].module_idx == 2) {
            if (this->unfiltered_viewpoints[i].pose.y < -9.0f) {
                this->unfiltered_viewpoints[i].module_idx = 3;
            } else {
                this->unfiltered_viewpoints[i].module_idx = 2;
            }
        } else {
            this->unfiltered_viewpoints[i].module_idx = module_membership[this->unfiltered_viewpoints[i].module_idx];
        }
    }
}

void ViewpointGenerator::getCoverageViewpoints(bool local, const std::string& coverage_file, bool compute_coverage, bool vx, const std::string& save_file) {
    // run greedy algorithm to select viewpoints and put in coverage_viewpoints
    if (local) {
        // remap module membership for this->all_faces
        this->remapModuleMembership();

        // then populate unfiltered_viewpoints
        this->populateViewpoints();

        // compute incidence angle between each viewpoint and each face
        std::cout << "Computing Incidence Angles..." << std::endl;
        batch_incidence_angle(this->unfiltered_viewpoints, this->all_faces, this->inc_angle_map);
        std::cout << "Incidence Angles Computed" << std::endl;

        if (!compute_coverage) {
            // load coverage map
            this->loadCoverageMap(coverage_file, vx);
        } else {
            // compute coverage map
            std::cout << "Computing Coverage Map..." << std::endl;
            this->populateCoverage();
            std::cout << "Saving Coverage Map..." << std::endl;
            this->saveCoverageMap(coverage_file, vx);
        }


        std::cout << "Running Greedy Algorithm..." << std::endl;
        for (size_t i = 0; i < 4; i++) {
            this->greedyModule(i);
        }
    } else {
        // then populate unfiltered_viewpoints
        std::cout << "Populating Viewpoints..." << std::endl;
        this->populateViewpoints();

        // compute incidence angle between each viewpoint and each face
        std::cout << "Computing Incidence Angles..." << std::endl;
        batch_incidence_angle(this->unfiltered_viewpoints, this->all_faces, this->inc_angle_map);

        if (!compute_coverage) {
            // load coverage map
            std::cout << "Loading Coverage Map..." << std::endl;
            this->loadCoverageMap(coverage_file, vx);
        } else {
            // compute coverage map
            std::cout << "Computing Coverage Map..." << std::endl;
            this->populateCoverage();
            std::cout << "Saving Coverage Map..." << std::endl;
            this->saveCoverageMap(coverage_file, vx);
        }

        std::cout << "Running Greedy Algorithm..." << std::endl;
        this->greedy();
    }

    if (save_file != "") {
        std::vector<std::vector<float>> viewpoint_data_save;
        for (size_t i = 0; i < coverage_viewpoints.size(); i++) {
            std::vector<float> vp_data;
            vp_data.push_back(this->coverage_viewpoints[i].pose.x);
            vp_data.push_back(this->coverage_viewpoints[i].pose.y);
            vp_data.push_back(this->coverage_viewpoints[i].pose.z);
            vp_data.push_back(this->coverage_viewpoints[i].viewdir.x);
            vp_data.push_back(this->coverage_viewpoints[i].viewdir.y);
            vp_data.push_back(this->coverage_viewpoints[i].viewdir.z);
            vp_data.push_back(this->coverage_viewpoints[i].module_idx);
            viewpoint_data_save.push_back(vp_data);
        }

        saveCSV(save_file, viewpoint_data_save);

    }

    std::cout << "NUM VIEWPOINTS: " << this->coverage_viewpoints.size() << std::endl;

    std::cout << "\n-----------------Coverage Viewpoints-----------------\n";
    for (size_t i = 0; i < this->coverage_viewpoints.size(); i++) {
        std::cout << "Viewpoint " << i << ": pose=" << this->coverage_viewpoints[i].pose.toString() << " viewdir=" << coverage_viewpoints[i].viewdir.toString() << std::endl;
    }

    // return this->coverage_viewpoints;
}

void ViewpointGenerator::getFilteredCoverage(std::vector<bool>& filtered_coverage_data) {
    // get filtered coverage map
    // filtered_coverage_data = this->filtered_coverage;
    filtered_coverage_data.clear();
    for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
        filtered_coverage_data.push_back(this->triangle_coverage[i].covered);
    }
}

void ViewpointGenerator::missedCoverage() {
    // DO NOT CALL THIS if greedy has not been run!!
    // debug coverage of unfiltered coverage after greedy algorithm interms of marginal gain
    // update marginal gain as we search for the maximal element
    for (size_t i = 0; i < this->vpcg_unfiltered.size(); i++) {
        float gain = 0;
        for (size_t face_idx = 0; face_idx < this->num_mesh_faces; face_idx++) {
            // if the viewpoint covers the face and the face is not already covered, increment gain
            if (
            // !(this->filtered_coverage[face_idx]) // face is not covered yet (according to coverage function)
            !(this->triangle_coverage[face_idx].covered) // face is not covered yet (according to coverage function)
            && this->coverage_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx] // viewpoint covers face
            && this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx] < this->inc_angle_max // viewpoint - face incidence angle is within threshold
            // && this->filtered_inc_angles[face_idx] > this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx] // improve incidence angle with this viewpoint
            && this->triangle_coverage[face_idx].best_inc_angle > this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx] // improve incidence angle with this viewpoint
            ) { 
                // std::cout << "Incidence Angle: " << this->inc_angle_map[vpcg_unfiltered[i].vp_map_idx][face_idx] << std::endl;
                // define gain function here
                // 1 is added to avoid division by zero and normalize best incidence angle to gain = 1
                gain += 1/(this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx] + 1);
                // gain += 1;
            }
        }
        // update the marginal gain array (marginal_gain)
        this->vpcg_unfiltered[i].gain = gain;
    }
    // all gain should be zero after running algorithm
    float leftover_gain = 0;
    for (size_t i = 0; i < this->vpcg_unfiltered.size(); i++) {
        leftover_gain += this->vpcg_unfiltered[i].gain;
    }
    std::cout << "Leftover gain=" << leftover_gain << std::endl;

    // Examine how many faces are left uncovered:
    size_t num_covered = 0;
    for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
        if (this->triangle_coverage[i].covered) { num_covered++; }
    }
    std::cout << "Number of faces covered with inc angle threshold=" << num_covered << std::endl;

    // distribution of incidence angles after rerunning coverage with max inc angle threshold (look at all faces that can be seen)
    num_covered = 0;
    std::cout << "Best Inc Angles: ";
    for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
        if (this->triangle_coverage[i].best_inc_angle < this->inc_angle_max) {
            num_covered++;
        }
    }
    std::cout << "Number of faces with incidence angle < max threshold=" << num_covered << std::endl;
}

void ViewpointGenerator::assignModuleMembership(std::vector<Viewpoint>& viewpoints) {
    // assign module membership to viewpoints
    // for now, just assign all viewpoints to module 0
    for (size_t i = 0; i < viewpoints.size(); i++) {
        float closest_err = std::numeric_limits<float>::max();
        size_t uidx = -1;
        std::cout << "Viewpoint " << i << ": " << viewpoints[i].pose.toString() << std::endl;
        for (size_t j = 0; j < this->unfiltered_viewpoints.size(); j++) {
            vec3 vec_err = viewpoints[i].pose - this->unfiltered_viewpoints[j].pose;
            float err = vec_err.norm();
            if (err < closest_err) {
                closest_err = err;
                uidx = j;
            }
        }
        std::cout << "Viewpoint " << i << " assigned to module " << this->unfiltered_viewpoints[uidx].module_idx << " with err=" << std::to_string(closest_err) << std::endl;
        viewpoints[i].module_idx = this->unfiltered_viewpoints[uidx].module_idx;
    }
}

void ViewpointGenerator::remapModuleMembership() {
    // remap module membership to four modules
    // reassign triangle module membership to four modules
    std::vector<size_t> module_membership = {0,3,2,3,2,2,2,1,1,3};
    for (size_t i = 0; i < this->all_faces.size(); i++) {
        if (this->all_faces[i]->module_idx == 2) {
            if (this->all_faces[i]->getCentroid().y < -9.0f){
                this->all_faces[i]->module_idx = 3;
            } else {
                this->all_faces[i]->module_idx = 2;
            }
        } else {
            this->all_faces[i]->module_idx = module_membership[this->all_faces[i]->module_idx];
        }
        if (this->all_faces[i]->module_idx > 3) {
            std::cout << "Error: Module Index out of bounds" << std::endl;
        }
    }
}

void ViewpointGenerator::greedyModule(size_t module_idx) {
    /*
    * greedy algorithm to select viewpoints from unfiltered_viewpoints
    * and put into filtered_viewpoints based on coverage
    */
    // initialize marginal gain and vector of ordered pointers for each viewpoint

    this->setUpCoverageGain();

    // this->filtered_coverage.clear();
    // this->filtered_coverage = std::vector<bool>(this->num_mesh_faces, false);
    // this->filtered_inc_angles.clear();
    // this->filtered_inc_angles = std::vector<float>(this->num_mesh_faces, std::numeric_limits<float>::max());

    // std::vector<TriangleCoverage> tri_coverage;

    // iterate over number of viewpoints (most time we can add viewpoints to this->coverage_viewpoints)
    std::cout << "Number of unfiltered viewpoints=" << this->unfiltered_viewpoints.size() << std::endl;
    for (size_t i = 0; i < this->unfiltered_viewpoints.size(); i++) {
        // skip viewpoints that don't match module_idx
        if (this->unfiltered_viewpoints[i].module_idx != module_idx) { continue; }

        // each iteration sort the viewpoint-gain objects
        this->sortUpdateMarginalGain(module_idx);

        // get viewpoint with maximal gain
        this->coverage_viewpoints.push_back(this->vpcg_unfiltered.begin()->vp);
        this->vpcg_filtered.push_back(*(this->vpcg_unfiltered.begin()));

        // remove first element from vpcoveragegains
        this->vpcg_unfiltered.erase(this->vpcg_unfiltered.begin());

        // update filtered coverage map
        this->updateCoverage(this->inc_angle_max);
        this->updateBestIncAngles();

        // check if we have covered all faces
        if (allTrue(this->triangle_coverage, module_idx) || allZeroGain(this->vpcg_unfiltered)) { break; }
    }
    this->sortUpdateMarginalGain(module_idx);
    size_t num_covered = 0;
    size_t num_faces = 0;
    for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
        if (this->triangle_coverage[i].module_idx == module_idx) {
            if (this->triangle_coverage[i].covered) { num_covered++; }
            num_faces++;
        }
    }
    std::cout << "Number of faces covered=" << num_covered << "/" << num_faces << " for Module " << module_idx << std::endl;
}

void ViewpointGenerator::greedy() {
    /*
    * greedy algorithm to select viewpoints from unfiltered_viewpoints
    * and put into filtered_viewpoints based on coverage
    */
    // initialize marginal gain and vector of ordered pointers for each viewpoint

    this->setUpCoverageGain();

    // this->filtered_coverage.clear();
    // this->filtered_coverage = std::vector<bool>(this->num_mesh_faces, false);
    // this->filtered_inc_angles.clear();
    // this->filtered_inc_angles = std::vector<float>(this->num_mesh_faces, std::numeric_limits<float>::max());

    // iterate over number of viewpoints (most time we can add viewpoints to this->coverage_viewpoints)
    std::cout << "Number of unfiltered viewpoints=" << this->unfiltered_viewpoints.size() << std::endl;
    // std::cout << "adding viewpoints: ";
    for (size_t i = 0; i < this->unfiltered_viewpoints.size(); i++) {
        // each iteration sort the viewpoint-gain objects
        this->sortUpdateMarginalGain();

        // // get viewpoint with maximal gain
        // this->coverage_viewpoints.push_back(this->vpcg_unfiltered.begin()->vp); // do this after we prune
        // std::cout << this->vpcg_unfiltered.begin()->vp.pose.toString() << " ";
        this->vpcg_filtered.push_back(*(this->vpcg_unfiltered.begin()));

        // remove first element from vpcoveragegains
        this->vpcg_unfiltered.erase(this->vpcg_unfiltered.begin());

        // update filtered coverage map
        this->updateCoverage(this->inc_angle_max);
        this->updateBestIncAngles();
        size_t num_covered = 0;
        for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
            if (this->triangle_coverage[i].covered) { num_covered++; }
        }

        // float coverage = static_cast<float>(num_covered) / this->num_mesh_faces;
        // std::cout << "\rCoverage=" << std::to_string(coverage) << std::flush;
        // if (coverage > 0.98) {
        //     std::cout << "breaking at 98 percent coverage" << std::endl;
        //     break;
        // }

        // check if we have covered all faces
        if (allTrue(this->triangle_coverage)) { 
            std::cout << "all true" << std::endl;
            break; 
        }
        if (allZeroGain(this->vpcg_unfiltered)) { 
            std::cout << "all zero gain" << std::endl;
            break;
        }

        // print histogram of filtered gains
        if (this->vpcg_filtered.size() > 20 && i % 20 == 0) {
            this->pruneFilteredViewpoints(true); // boolean is for visualization
        }

    }
    this->pruneFilteredViewpoints();
    this->sortUpdateMarginalGain();
    size_t num_covered = 0;
    for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
        if (this->triangle_coverage[i].covered) { num_covered++; }
    }

    this->coverage_viewpoints.clear();
    for (size_t i = 0; i < this->vpcg_filtered.size(); i++) {
        this->coverage_viewpoints.push_back(this->vpcg_filtered[i].vp);
    }
    std::cout << "Number of faces covered=" << num_covered << "/" << this->num_mesh_faces << std::endl;
    std::cout << "Number of viewpoints selected=" << this->coverage_viewpoints.size() << std::endl;
}

float ViewpointGenerator::computeMarginalGain(size_t module_idx, VPCoverageGain& vpcg) {
    float gain = 0;
    if (vpcg.vp.module_idx == module_idx
        || module_idx == SIZE_MAX) {
        for (size_t face_idx = 0; face_idx < this->num_mesh_faces; face_idx++) {

            // check if the face applies to this module
            if ( this->all_faces[face_idx]->module_idx == module_idx
                || module_idx == SIZE_MAX) {
                // if the viewpoint covers the face and the face is not already covered, increment gain
                if (
                // !(this->filtered_coverage[face_idx]) // face is not covered yet (according to coverage function)
                !(this->triangle_coverage[face_idx].covered) // face is not covered yet (according to coverage function)
                && this->coverage_map[vpcg.vp_map_idx][face_idx] // viewpoint covers face
                ) {
                    if ( this->inc_angle_map[vpcg.vp_map_idx][face_idx] < this->inc_angle_max // viewpoint - face incidence angle is within threshold
                    ) {
                        // float inc_unfiltered = this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx];
                        // bool covered = this->coverage_map[vpcg_unfiltered[i].vp_map_idx][face_idx];
                        // std::cout << "Incidence Angle: " << inc_unfiltered << " Covered: " << covered << std::endl;
                        // && this->inc_angle_map[vpcg_unfiltered[i].vp_map_idx][face_idx] < this->filtered_inc_angles[face_idx] // improve incidence angle with this viewpoint
                        // ) { 
                        if ( this->inc_angle_map[vpcg.vp_map_idx][face_idx] < this->triangle_coverage[face_idx].best_inc_angle ) { // improve incidence angle with this viewpoint
                            // std::cout << "Incidence Angle: " << this->inc_angle_map[vpcg_unfiltered[i].vp_map_idx][face_idx] << std::endl;
                            // define gain function here
                            // 1 is added to avoid division by zero and normalize best incidence angle to gain = 1
                            float inc_best = this->triangle_coverage[face_idx].best_inc_angle;
                            // float inc_best = this->filtered_inc_angles[face_idx];
                            float inc_unfiltered = this->inc_angle_map[vpcg.vp_map_idx][face_idx];
                            float inc_improve = inc_best - inc_unfiltered;
                            float inc_improve_thresh = 0.0f;
                            if (inc_best < this->inc_improvement_minimum) {
                                inc_improve_thresh = this->inc_improvement_threshold;
                            }

                            if (inc_improve > inc_improve_thresh) {
                                // gain += 1/(inc_improve + 1);
                                gain += inc_improve;
                            }
                            // gain += 1/(this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx] + 1);
                            // gain++;
                        }
                    } else {
                        // float inc_unfiltered = this->inc_angle_map[this->vpcg_unfiltered[i].vp_map_idx][face_idx];
                        // bool covered = this->coverage_map[vpcg_unfiltered[i].vp_map_idx][face_idx];
                        // std::cout << "Incidence Angle: " << inc_unfiltered << " Covered: " << covered << std::endl;
                    }
                }
            }
        }
    }
    return gain;
}

void ViewpointGenerator::pruneFilteredViewpoints(bool visualize) {
    // update gains for filtered viewpoints
    this->updateFilteredGain();

    // prune the filtered viewpoints with zero gain
    size_t num_pruned(0);
    if (visualize) {std::cout << "Viewpoint Count before=" << this->vpcg_filtered.size();}
    for (size_t i = 0; i < this->vpcg_filtered.size(); i++) {
        if (this->vpcg_filtered[i].gain < 1e-5) {
            this->vpcg_filtered.erase(this->vpcg_filtered.begin() + i);
            num_pruned++;
            i--;
        }   
    }
    if (visualize) {
        std::vector<float> filtered_gain_data;
        for (size_t i = 0; i < this->vpcg_filtered.size(); i++) {
            filtered_gain_data.push_back(std::log(this->vpcg_filtered[i].gain + 1e-9));
        }
        std::cout << ", after=" << this->vpcg_filtered.size();
        std::cout << " | num pruned = " << num_pruned;
        size_t num_covered = 0;
        for (size_t i = 0; i < this->triangle_coverage.size(); i++) {
            if (this->triangle_coverage[i].covered) { num_covered++; }
        }
        std::cout << " | coverage=" << static_cast<float>(num_covered) / this->num_mesh_faces * 100 << "%" << std::endl;
        printHistogram(filtered_gain_data);
    }
}

void ViewpointGenerator::sortUpdateMarginalGain(size_t module_idx) {

    // update marginal gain as we search for the maximal element
    auto it = this->vpcg_unfiltered.begin();
    for (size_t i = 0; i < this->vpcg_unfiltered.size(); i++) {
        // end iterator to sort up to (this element and after don't need to be sorted)
        it++;
        // update the marginal gain array (marginal_gain)
        this->vpcg_unfiltered[i].gain = this->computeMarginalGain(module_idx, this->vpcg_unfiltered[i]);

        // check if this is the largest element by comparing it to the next element (except if on last element)
        if (i < this->vpcg_unfiltered.size() - 1 && this->vpcg_unfiltered[i].gain >= this->vpcg_unfiltered[i+1].gain) {
            break;
        }
    }

    // sort the subset of the marginal gain array that still needs to be sorted
    std::sort(
        this->vpcg_unfiltered.begin(), 
        // this->vpcg_unfiltered.end(),
        it,
        [](VPCoverageGain a, VPCoverageGain b) -> bool { 
            return a.gain > b.gain;
        }
    );
}

void ViewpointGenerator::updateFilteredGain() {
    // updated gains for filtered viewpoints wrt to filtered coverage of all other filtered viewpoints

    // iterate over each filtered viewpoint
    for (size_t vp_idx = 0; vp_idx < this->vpcg_filtered.size(); vp_idx++) {

        // accumulate gain across all covered triangles
        float gain = 0;

        // iterate over each face
        for (size_t tri_idx = 0; tri_idx < this->num_mesh_faces; tri_idx++) {

            // for each face, check if viewpoint covers it
            if (this->coverage_map[this->vpcg_filtered[vp_idx].vp_map_idx][tri_idx]
            && this->inc_angle_map[this->vpcg_filtered[vp_idx].vp_map_idx][tri_idx] < this->inc_angle_max) {

                // if viewpoint covers face, check if face is covered by any other viewpoint in vpcg_filtered
                // float best_inc_angle = std::numeric_limits<float>::max();
                float best_inc_angle = M_PI;
                float this_inc_angle = this->inc_angle_map[this->vpcg_filtered[vp_idx].vp_map_idx][tri_idx];

                // iterate over each other viewpoint in vpcg_filtered
                for (size_t vp_idx2 = 0; vp_idx2 < this->vpcg_filtered.size(); vp_idx2++) {

                    // if face is covered by this second viewpoint and isn't the same viewpoint as the first
                    if (this->coverage_map[this->vpcg_filtered[vp_idx2].vp_map_idx][tri_idx] && vp_idx != vp_idx2) {

                        // get the incidence angle of the viewpoint with the best incidence angle
                        float inc_angle = this->inc_angle_map[this->vpcg_filtered[vp_idx2].vp_map_idx][tri_idx];

                        // if this incidence angle is better than the current best, update the best
                        if (inc_angle < best_inc_angle) {
                            best_inc_angle = inc_angle;
                        }
                    }
                }

                // if this_inc_angle is better than all the others, add to gain
                if (this_inc_angle < best_inc_angle) {
                    // gain += 1/(best_inc_angle - this_inc_angle + 1);
                    gain += best_inc_angle - this_inc_angle;
                }
            }
        }

        // update the gain for this viewpoint
        this->vpcg_filtered[vp_idx].gain = gain;
    }
}

void ViewpointGenerator::updateBestIncAngles() {
    // for each mesh face, find the incidence angle of the viewpoint with the 
    // best incidence angle in vpcg_filtered and save to filtered_inc_angles
    for (size_t i = 0; i < this->num_mesh_faces; i++) {
        float best_inc_angle = M_PI;
        for (size_t j = 0; j < this->vpcg_filtered.size(); j++) {
            // if this mesh face is covered by this viewpoint, save the incidence angle
            float inc_angle = this->inc_angle_map[this->vpcg_filtered[j].vp_map_idx][i];
            if ( this->coverage_map[this->vpcg_filtered[j].vp_map_idx][i] && inc_angle < best_inc_angle) {
                best_inc_angle = inc_angle;
            }
        }
        this->triangle_coverage[i].best_inc_angle = best_inc_angle;
        // this->filtered_inc_angles[i] = best_inc_angle;
    }
}

void ViewpointGenerator::updateCoverage(float inc_angle_threshold) {
    // update the triangle coverage array based on this->vpcg_filtered viewpoints
    for (size_t i = 0; i < this->num_mesh_faces; i++) {
        for (size_t j = 0; j < this->vpcg_filtered.size(); j++) {
            if (this->coverage_map[this->vpcg_filtered[j].vp_map_idx][i]) {
                this->triangle_coverage[i].covered_any = true;
                if (this->inc_angle_map[this->vpcg_filtered[j].vp_map_idx][i] < inc_angle_threshold) {
                    this->triangle_coverage[i].covered = true;
                    break;
                }
            }
        }
    }
}

void ViewpointGenerator::populateCoverage() {
    std::cout << "Populating Coverage Map..." << std::endl;
    // populate the filtered or unfiltered coverage map
    getCoverage(this->unfiltered_viewpoints, this->all_faces, this->coverage_map);
    // std::cout << "Coverage Map Computed. Filtering by inv_angle" << std::endl;
    // for (size_t vp_idx=0; vp_idx < this->unfiltered_viewpoints.size(); vp_idx++) {
    //     for (size_t face_idx=0; face_idx < this->num_mesh_faces; face_idx++) {
    //         if (this->coverage_map[vp_idx][face_idx] && this->inc_angle_map[vp_idx][face_idx] > M_PI / 2.0f) {
    //             this->coverage_map[vp_idx][face_idx] = false;
    //         }
    //     }
    // }
    std::cout << std::endl;
}


void ViewpointGenerator::setUpCoverageGain() {
    this->vpcg_filtered = std::vector<VPCoverageGain>();
    this->vpcg_unfiltered = std::vector<VPCoverageGain>();
    for (size_t i = 0; i < this->coverage_map.size(); i++) {
        VPCoverageGain vpcg;
        vpcg.vp = this->unfiltered_viewpoints[i];
        vpcg.gain = std::numeric_limits<float>::max();
        vpcg.coverage = this->coverage_map[i];
        vpcg.vp_map_idx = i;
        vpcg.redundant = false;

        // incidence angle to corresponding triangle generation in radians:
        getIncidenceAngle(this->unfiltered_viewpoints[i].viewdir, *(this->all_faces[i]), vpcg.inc_angle);
        this->vpcg_unfiltered.push_back(vpcg);
    }
    this->triangle_coverage = std::vector<TriangleCoverage>(this->num_mesh_faces);
    for (size_t i = 0; i < this->num_mesh_faces; i++) {
        this->triangle_coverage[i].covered = false;
        this->triangle_coverage[i].best_inc_angle = M_PI;
        this->triangle_coverage[i].module_idx = this->all_faces[i]->module_idx;
    }
}

void ViewpointGenerator::countMeshFaces() {
    // count the mesh faces and populate all_faces (vector of pointers to each face)
    this->num_mesh_faces = 0;
    for (size_t obs_idx = 0; obs_idx < this->structure.size(); obs_idx++) {
        this->num_mesh_faces += this->structure[obs_idx].faces.size();
        for (size_t face_idx = 0; face_idx < this->structure[obs_idx].faces.size(); face_idx++) {
            this->all_faces.push_back(&(this->structure[obs_idx].faces[face_idx]));
        }
    }
    for (size_t obs_idx = 0; obs_idx < this->convex_structure.size(); obs_idx++) {
        this->num_convex_mesh_faces += this->convex_structure[obs_idx].faces.size();
        for (size_t face_idx = 0; face_idx < this->convex_structure[obs_idx].faces.size(); face_idx++) {
            this->all_convex_faces.push_back(&(this->convex_structure[obs_idx].faces[face_idx]));
        }
    }
}

bool ViewpointGenerator::populateViewpoints() {

    // float norm_rotate_angle = 30.0f * M_PI / 180.0f;
    // float viewdir_rotate_angle = 30.0f * M_PI / 180.0f;

    size_t num_successful_viewpoints = 0;
    size_t num_failed_viewpoints = 0;
    std::vector<Viewpoint> sampled_viewpoints = std::vector<Viewpoint>();
    std::vector<size_t> sampled_face_indices = std::vector<size_t>();


    for (size_t face_idx = 0; face_idx < this->num_mesh_faces; face_idx++) {
    // for (size_t face_idx = 0; face_idx < 10; face_idx++) {
        vec3 centroid = this->all_faces[face_idx]->getCentroid();
        vec3 normal = this->all_faces[face_idx]->n;

        // ********* For Generating extra viewpoints for every face primitive ************
        // std::vector<vec3> view_dir_centroid;
        // // view_dir_centroid.push_back(normal);

        // // get rotational axes
        // vec3 v_hat = vec3(
        //     normal.z * cosf( atan2f(normal.y, normal.x) ),
        //     normal.y * sinf( atan2f(normal.y, normal.x) ),
        //     sqrtf(normal.x * normal.x + normal.y * normal.y) // 0 to 1
        // );
        // vec3 u_hat = v_hat.cross(normal);

        // view_dir_centroid.push_back(normal.rotate(norm_rotate_angle, u_hat));
        // view_dir_centroid.push_back(normal.rotate(-norm_rotate_angle, u_hat));
        // view_dir_centroid.push_back(normal.rotate(norm_rotate_angle, v_hat));
        // view_dir_centroid.push_back(normal.rotate(-norm_rotate_angle, v_hat));

        // for (size_t i = 0; i < view_dir_centroid.size(); i++) {
        //     Viewpoint vp = Viewpoint(
        //         centroid + view_dir_centroid[i] * this->vgd, // TODO: dynamic vgd for obstacle avoidance
        //         view_dir_centroid[i] * -1.0f
        //     );
        //     sampled_viewpoints.push_back(vp);
        //     sampled_face_indices.push_back(face_idx);

        //     // rotate viewpoint up, down, left and right by angle to get four new viewpoints with different viewdirs
        //     this->rotateViewpoint(vp, sampled_viewpoints, viewdir_rotate_angle); // rotate by one radian
        //     sampled_face_indices.push_back(face_idx);
        //     sampled_face_indices.push_back(face_idx);
        //     sampled_face_indices.push_back(face_idx);
        //     sampled_face_indices.push_back(face_idx);
        // }
        // ************************************************************

        Viewpoint vp = Viewpoint(
            centroid + normal * this->vgd, // TODO: dynamic vgd for obstacle avoidance
            normal * -1.0f,
            this->all_faces[face_idx]->module_idx
            );

        // std::cout << " (pose=" << vp.pose.toString() << ", viewdir=" << vp.viewdir.toString() << ") ";
        sampled_viewpoints.push_back(vp);
        sampled_face_indices.push_back(face_idx);

        // // rotate viewpoint up, down, left and right by angle to get four new viewpoints with different viewdirs
        // this->rotateViewpoint(vp, sampled_viewpoints, 0.5f); // rotate by one radian
        // sampled_face_indices.push_back(face_idx);
        // sampled_face_indices.push_back(face_idx);
        // sampled_face_indices.push_back(face_idx);
        // sampled_face_indices.push_back(face_idx);
    }

    // // debug int points
    // vec3 **int_points = new vec3*[sampled_viewpoints.size()];
    // for (size_t i = 0; i < sampled_viewpoints.size(); i++) {
    //     int_points[i] = new vec3[this->num_mesh_faces * 3];
    //     for (size_t j = 0; j < this->num_mesh_faces * 3; j++) {
    //         int_points[i][j].set(0.0f, 0.0f, 0.0f);
    //     }
    // }

    // need to test viewpoints against 
    // check each viewpoint-face combination for ray casting collisions
    std::vector<bool> in_collision_convex;
    cuda_kernel_collision_points(
        sampled_viewpoints,
        this->all_convex_faces,
        vec3(1e9f, 1e9f, 1e9f), // needs to be collision free
        in_collision_convex
    );


    // check each viewpoint-face combination for ray casting collisions
    std::vector<bool> in_collision;
    cuda_kernel_collision_points(
        sampled_viewpoints,
        this->all_faces,
        vec3(1e9f, 1e9f, 1e9f), // needs to be collision free
        in_collision
    );

    for (size_t i = 0; i < sampled_viewpoints.size(); i++) {
        if (!in_collision[i] && !in_collision_convex[i]) {
            // std::cout << "Viewpoint " << sampled_viewpoints[i].pose.toString() << " is not in collision" << std::endl;
            this->unfiltered_viewpoints.push_back(sampled_viewpoints[i]);
            num_successful_viewpoints++;
        } else {
            // std::cout << "Viewpoint " << i << " : "<< sampled_viewpoints[i].pose.toString() << " is in collision" << std::endl;
            num_failed_viewpoints++;
        }
    }

    // debug
    // std::cout << "Collision Matrix:" << std::endl;
    // for (size_t i = 0; i < sampled_viewpoints.size(); i++) {
    //     std::cout << "Viewpoint " << sampled_viewpoints[i].pose.toString();
    //     std::cout << " paired with " << this->all_faces[sampled_face_indices[i]]->toString() << ": ";
    //     if (collisions[i]) {
    //         std::cout << "collisions: ";
    //         for (size_t j = 0; j < this->num_mesh_faces * 3; j++) {
    //             if (int_points[i][j].x != std::numeric_limits<float>::infinity()) {
    //                 std::cout << this->all_faces[size_t(j/3)]->toString() << ", ";
    //                 std::cout << ", Intersection=" << int_points[i][j].toString() << ", ";
    //             }
    //         }
    //         std::cout << std::endl;
    //     } else {
    //         std::cout << "no collision" << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    // for (size_t i = 0; i < sampled_viewpoints.size(); i++) {
    //     delete[] int_points[i];
    // }
    // delete[] int_points;

    // delete[] collisions;


    std::cout << "Successfully added " << num_successful_viewpoints << "/" << num_failed_viewpoints + num_successful_viewpoints << " viewpoints" << std::endl;
    return num_successful_viewpoints > 0;
}

void ViewpointGenerator::rotateViewpoint(const Viewpoint& vp, std::vector<Viewpoint>& rotated_viewpoints, float angle) {
    // rotate viewpoint up, down, left and right by angle to get four new viewpoints with different viewdirs

    // get rotational axes
    vec3 v_hat = vec3(
        vp.viewdir.z * cosf( atan2f(vp.viewdir.y, vp.viewdir.x) ),
        vp.viewdir.y * sinf( atan2f(vp.viewdir.y, vp.viewdir.x) ),
        sqrtf(vp.viewdir.x * vp.viewdir.x + vp.viewdir.y * vp.viewdir.y) // 0 to 1
    );
    vec3 u_hat = v_hat.cross(vp.viewdir);

    // normalize axes so we rotate correctly
    v_hat = v_hat / v_hat.norm();
    u_hat = u_hat / u_hat.norm();

    // rotate positive around v_hat
    rotated_viewpoints.push_back(Viewpoint(
        vp.pose,
        vp.viewdir.rotate(angle, v_hat)
    ));

    // rotate negative around v_hat
    rotated_viewpoints.push_back(Viewpoint(
        vp.pose,
        vp.viewdir.rotate(-angle, v_hat)
    ));

    // rotate positive around u_hat
    rotated_viewpoints.push_back(Viewpoint(
        vp.pose,
        vp.viewdir.rotate(angle, u_hat)
    ));

    // rotate negative around u_hat
    rotated_viewpoints.push_back(Viewpoint(
        vp.pose,
        vp.viewdir.rotate(-angle, u_hat)
    ));
}

bool ViewpointGenerator::collision(Viewpoint vp) {
    for (size_t obs_idx = 0; obs_idx < this->structure.size(); obs_idx++) {
        if (this->structure[obs_idx].collision(vp.pose)) { return true; }
    }
    return false;
}

bool ViewpointGenerator::collision(vec3 pose, const std::vector<vec3*>& points) {
    // must displace centroid in direction of normal by small amount to avoid self-collision
    for (size_t obs_idx = 0; obs_idx < this->structure.size(); obs_idx++) {
        for (size_t i=0; i < points.size(); i++) {
            if (this->structure[obs_idx].collision(pose, *points[i] + (pose - *points[i]) * 1e-9)) { return true; }
        }
    }
    return false;
}