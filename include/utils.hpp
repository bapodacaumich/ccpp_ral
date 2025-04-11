#ifndef UTILS_HPP
#define UTILS_HPP

#include "node3d_struct.hpp"
#include "obs.hpp"
#include "triangle_coverage_struct.hpp"
#include "triangle_struct.hpp"
#include "vec3_struct.hpp"
#include "viewpoint_struct.hpp"
#include "viewpoint_coverage_gain_struct.hpp"

#include <atomic>
#include <string>
#include <vector>

static vec3* TEMP = new vec3();

bool ray_int_plane(Node3D node, Plane plane, float eps=1e-5f, vec3& intPoint= * TEMP);
bool ray_int_triangle(vec3 origin, vec3 vector, vec3 end, Triangle tri, vec3& intPoint= * TEMP, float eps=1e-5f);
bool ray_int_triangle(vec3 origin, vec3 vector, Triangle tri, vec3& intPoint= * TEMP, float eps=1e-5f);
bool ray_int_triangle(Node3D node, Triangle tri, vec3& intPoint= * TEMP, float eps=1e-5f);
float heading_change(Node3D node, Node3D next_node);
float heading_change(Node3D node, vec3 vector);
float heading_change(vec3 v0, vec3 v1);

// loading functions
bool loadCSV(const std::string& filename, std::vector<std::vector<float>>& data, int rowlen, char delimiter=',', bool raw=false);
bool loadCSVbool(const std::string& filename, std::vector<std::vector<bool>>& data);

// saving functions
void saveCSVbool(const std::string& filename, const std::vector<std::vector<bool>>& data);
void saveCSV(const std::string& filename, const std::vector<std::vector<float>>& data);
void saveCSVsizet(const std::string& filename, const std::vector<std::vector<size_t>>& data);
void loadCube(std::vector<std::vector<std::vector<float>>>& data, float xs=-1, float xf=1);
void convertFlatToTriangle(const std::vector<std::vector<float>>& flatData, std::vector<Triangle>& tris, size_t module_idx=0);
void loadCubeOBS(std::vector<OBS>& obsVec);
void loadConvexStationOBS(std::vector<OBS>& obsVec, float scale);
void loadStationOBS(std::vector<OBS>& obsVec, float scale);
void printHistogram(std::vector<float>& data);
void loadVxStationOBS(std::vector<OBS>& obsVec, float scale);

void vecToTri(const std::vector<std::vector<std::vector<float>>>& data, std::vector<Triangle>& tris);
bool allTrue(const std::vector<TriangleCoverage>& arr, size_t module_idx);
bool allTrue(const std::vector<TriangleCoverage>& arr);
bool allTrue(const std::vector<bool>& arr);
void numTrue(const std::vector<bool>& arr, size_t& num_true);
bool allZeroGain(const std::vector<VPCoverageGain>& arr);
void getCoverage(const std::vector<Viewpoint>& viewpoints, const std::vector<Triangle*>& triangles, std::vector<std::vector<bool>>& coverage_map);
void displayProgressBar(double progress, int width, std::ostringstream& message);
void getIncidenceAngle(vec3 viewdir, Triangle tri, float& angle);
void pinhole_camera_test( bool& visible, vec3 pose, vec3 viewdir, vec3 point, float hfov, float vfov);
void cw_acceleration(vec3& acceleration, vec3 pose, vec3 velocity);

// compute fuel cost of a path segment
float cw_cost(vec3 start, vec3 end, float speed, size_t N);

// compute the fuel cost of a change in heading maneuver
float fuel_cost(vec3 pose, vec3 v0, vec3 v1, float speed, float dt);

void batch_incidence_angle(const std::vector<Viewpoint>& viewpoints, const std::vector<Triangle*>& triangles, std::vector<std::vector<float>>& inc_angle_map);
float compute_fuel_cost_path(const std::string& ufile, const std::string& tfile, float Isp=80, float m=5);
float compute_pathtime_path(const std::string& tfile);
void get_uncoverable_faces(std::vector<OBS>& obsVec, std::vector<size_t>& uncoverable);
void orientation_moving_average(const std::vector<std::vector<float>> &path_data, std::vector<std::vector<float>> & smoothed_path, size_t window_size=5);

// compute number of views each face gets, minimum and average aoi for each face, and binning counts for aoi
void compute_saturation_path(
    const std::vector<std::vector<float>> &path_data, 
    std::vector<std::vector<float>> &saturation_map, 
    std::vector<std::vector<size_t>> &saturation_bins,
    size_t n_bins=16 // n_bins between 0 and pi/2. one extra will be added for outliers
);
float compute_coverage_path(const std::vector<std::vector<float>> &path_data, std::vector<bool>& coverage);
float compute_coverage_file(const std::string& file, std::vector<bool>& coverage);
float compute_smoothed_coverage(const std::vector<std::vector<float>> &path_data, std::vector<bool>& coverage);
std::string getnum(float num);
std::string removeTrailingZeros(const std::string& str);

vec3 slerp(vec3 v0, vec3 v1, float t);

void writeTriToFile(const Triangle &tri, std::ofstream &file);
void saveObjFile(const std::vector<OBS>& obsVec, const std::string &filename);

#endif // UTILS_HPP