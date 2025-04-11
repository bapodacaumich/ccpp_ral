#include "cuda_kernels.h"
#include "model.hpp"
#include "obs.hpp"
#include "utils.hpp"

#include <cmath>

// default model constructor
Model::Model() {
    std::vector<OBS> obsVec;
    loadStationOBS(obsVec, 4.0f);

    for (size_t i = 0; i < obsVec.size(); i++) {
        for (size_t j = 0; j < obsVec[i].faces.size(); j++) {
            this->normals.push_back(obsVec[i].faces[j].n);
        }
    }

    this->obsVec = obsVec;
}

// initialize model with scale and choose voxelized station or normal (bool)
Model::Model(bool vx, float scale) {
    std::vector<OBS> obsVec;

    if (vx) {
        loadVxStationOBS(obsVec, scale);
    } else {
        loadStationOBS(obsVec, scale);
    }

    this->loadOBSVec(obsVec);
}

// load obsvec into private obsvec variable
void Model::loadOBSVec(std::vector<OBS> obsVec) {

    for (size_t i = 0; i < obsVec.size(); i++) {
        for (size_t j = 0; j < obsVec[i].faces.size(); j++) {
            this->normals.push_back(obsVec[i].faces[j].n);
        }
    }

    this->obsVec = obsVec;
}

void Model::loadTriangles() {
    for (size_t i = 0; i < this->obsVec.size(); i++) {
        for (size_t j = 0; j < this->obsVec[i].faces.size(); j++) {
            this->all_faces.push_back(&(this->obsVec[i].faces[j]));
        }
    }
}

/**
 * @brief Get binary coverage for a single viewpoint and populate coverage_out with bools
 * @param coverage_out std::vector<bool>&, output data
 * @param pose Viewpoint&, input data
 * @return void
 */

void Model::getBinaryCoverage(std::vector<bool> &coverage_out, Viewpoint &pose) {
    cuda_kernel_coverage(pose, this->all_faces, coverage_out);
}

/**
 * @brief Get summed binary coverage for a single viewpoint and populate coverage_out with size_t
 * @param coverage_out size_t&, output data
 * @param pose Viewpoint&, input data
 * @return void
 */

void Model::getBinaryCoverage(size_t &coverage_out, Viewpoint &pose) {
    std::vector<bool> coverage;
    cuda_kernel_coverage(pose, this->all_faces, coverage);
    coverage_out = 0;
    for (size_t i = 0; i < coverage.size(); i++) {
        if (coverage[i]) {
            coverage_out++;
        }
    }
}

/**
 * @brief get the marginal coverage (not quality) for a single viewpoint given a coverage prior
 * @param coverage_out std::vector<bool>&, output marginal coverage vector
 * @param pose Viewpoint&, input viewpoint
 * @param coverage_prior std::vector<bool>&, input coverage prior
 * @return void
 */

void Model::getMarginalBinaryCoverage(std::vector<bool> &coverage_out, Viewpoint &pose, std::vector<bool> &coverage_prior) {
    std::vector<bool> coverage;
    cuda_kernel_coverage(pose, this->all_faces, coverage);
    for (size_t i = 0; i < coverage.size(); i++) {
        coverage_out.push_back(coverage[i] && !coverage_prior[i]);
    }
}

/**
 * @brief get the summed marginal coverage (not quality) for a single viewpoint given a coverage prior
 * @param coverage_out size_t&, output marginal coverage
 * @param pose Viewpoint&, input viewpoint
 * @param coverage_prior std::vector<bool>&, input coverage prior
 * @return void
 */

void Model::getMarginalBinaryCoverage(size_t &coverage_out, Viewpoint &pose, std::vector<bool> &coverage_prior) {
    std::vector<bool> coverage;
    cuda_kernel_coverage(pose, this->all_faces, coverage);
    coverage_out = 0;
    for (size_t i = 0; i < coverage.size(); i++) {
        if (coverage[i] && !coverage_prior[i]) {
            coverage_out++;
        }
    }
}

/**
 * @brief get the coverage quality for a single viewpoint per triangle
 * @param coverage_quality std::vector<float>&, output coverage quality
 * @param pose Viewpoint&, input viewpoint
 * @return void
 */

void Model::getCoverageQuality(std::vector<float> &coverage_quality, Viewpoint &pose) {
    // get coverage quality for each viewpoint-triangle pair
    std::vector<std::vector<float>> inc_angles;
    std::vector<Viewpoint> vp = {pose};
    cuda_kernel_inc_angle(vp, this->all_faces, inc_angles);

    // get coverage
    std::vector<bool> coverage_out;
    this->getBinaryCoverage(coverage_out, pose);

    // conglomerate coverage quality
    for (size_t i = 0; i < inc_angles[0].size(); i++) {
        if (coverage_out[i]) {
            float coverage_quality_i = this->coverageQuality(pose.pose, i, inc_angles[0][i]);
            coverage_quality.push_back(coverage_quality_i);
        }
    }
}

/**
 * @brief get the summed coverage quality for a single viewpoint
 * @param coverage_quality float&, output coverage quality
 * @param pose Viewpoint&, input viewpoint
 * @return void
 */

void Model::getCoverageQuality(float &coverage_quality, Viewpoint &pose) {
    std::vector<float> coverage_quality_map;
    this->getCoverageQuality(coverage_quality_map, pose);
    coverage_quality = 0.0f;
    for (size_t i = 0; i < coverage_quality_map.size(); i++) {
        coverage_quality += coverage_quality_map[i];
    }
}

/**
 * @brief get the marginal coverage quality per triangle for a single viewpoint given coverage quality prior
 * @param marginal_coverage_quality std::vector<float>&, output marginal coverage quality
 * @param pose Viewpoint&, input viewpoint
 * @param coverage_prior std::vector<float>&, input coverage quality prior (higher is better -- area)
 * @return void
 */

void Model::getMarginalCoverageQuality(std::vector<float> &marginal_coverage_quality, Viewpoint &pose, std::vector<float> &coverage_prior) {
    // get coverage quality for each viewpoint-triangle pair
    std::vector<std::vector<float>> inc_angles;
    std::vector<Viewpoint> vp = {pose};
    cuda_kernel_inc_angle(vp, this->all_faces, inc_angles);

    // get coverage
    std::vector<bool> coverage_out;
    this->getBinaryCoverage(coverage_out, pose);

    // conglomerate coverage quality
    for (size_t i = 0; i < inc_angles[0].size(); i++) {
        float coverage_quality = this->coverageQuality(pose.pose, i, inc_angles[0][i]);
        if (coverage_out[i] && coverage_quality > coverage_prior[i]) {
            marginal_coverage_quality.push_back(coverage_quality - coverage_prior[i]);
        }
    }
}

/**
 * @brief get the summed marginal coverage quality per triangle for a single viewpoint given coverage quality prior
 * @param marginal_coverage_quality float&, output marginal coverage quality
 * @param pose Viewpoint&, input viewpoint
 * @param coverage_prior std::vector<float>&, input coverage quality prior (higher is better -- area)
 * @return void
 */

void Model::getMarginalCoverageQuality(
    float &marginal_coverage_quality, 
    Viewpoint &pose,
    std::vector<float> &coverage_prior
) {
    // get coverage quality for each viewpoint-triangle pair
    std::vector<std::vector<float>> inc_angles;
    std::vector<Viewpoint> vp = {pose};
    cuda_kernel_inc_angle(vp, this->all_faces, inc_angles);

    // get coverage
    std::vector<bool> coverage_out;
    this->getBinaryCoverage(coverage_out, pose);

    // conglomerate coverage quality
    for (size_t i = 0; i < inc_angles[0].size(); i++) {
        float coverage_quality = this->coverageQuality(pose.pose, i, inc_angles[0][i]);
        if (coverage_out[i] && coverage_quality > coverage_prior[i]) {
            marginal_coverage_quality += (coverage_quality - coverage_prior[i]);
        }
    }
}

/**
 * @brief get coverage quality for a single viewpoint per triangle
 * @param pose vec3, input viewpoint pose
 * @param tri_idx size_t, input triangle index
 * @param inc_angle float, input incidence angle between triangle and pose
 * @return float, output coverage quality
 */

float Model::coverageQuality(vec3 pose, size_t tri_idx, float inc_angle) {
    // get coverage quality for a single viewpoint per triangle
    return std::cos(inc_angle) * this->all_faces[tri_idx]->getArea() / ((pose - this->all_faces[tri_idx]->c).norm() * (pose - this->all_faces[tri_idx]->c).norm());
}