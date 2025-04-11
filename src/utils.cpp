#include "cuda_kernels.h"
#include "limit_struct.hpp"
#include "node3d_struct.hpp"
#include "obs.hpp"
#include "rrtz.hpp"
#include "triangle_coverage_struct.hpp"
#include "triangle_struct.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"
#include "viewpoint_struct.hpp"
#include "viewpoint_coverage_gain_struct.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

bool ray_int_plane(Node3D node, Plane plane, float eps, vec3& intPoint) {
    vec3 origin_to_point = plane.point - node.origin;
    // vec3 end_to_point = node.end - node.origin;
    vec3 end_to_point = node.end - plane.point;
    float origin_dot = origin_to_point.dot(plane.normal);
    float end_dot = end_to_point.dot(plane.normal);
    float abs_origin_dot = fabsf(origin_dot);
    float abs_end_dot = fabsf(end_dot);
    if (abs_origin_dot > eps && abs_end_dot > eps && (origin_dot > 0 ) ^ (end_dot > 0)) {
        float fac = abs_origin_dot / (abs_origin_dot + abs_end_dot);
        intPoint = node.end * fac + node.origin * (1 - fac);
        return true;
    }
    return false;
}

bool ray_int_triangle(vec3 origin, vec3 vector, vec3 end, Triangle tri, vec3& intPoint, float eps) {
    // look for any intersections between the ray and triangle (before end-point)
    vec3 e1 = tri.b - tri.a;
    vec3 e2 = tri.c - tri.a;
    vec3 h = vector.cross(e2);
    float a = e1.dot(h);

    // if ray is parallel to triangle
    if (fabsf(a) < eps) { return false; }

    float f = 1 / a;
    vec3 s = origin - tri.a;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) { return false; }
    vec3 q = s.cross(e1);
    float v = f * vector.dot(q);
    if (v < 0.0f || u + v > 1.0f) { return false; }

    // now find intersection point
    float t = f * e2.dot(q);
    intPoint = origin + vector * t; // will save garbage answer to intpoint if t <= eps

    // check if intersection point is between start and endpoint
    vec3 origin_end = end - origin;
    vec3 origin_int = intPoint - origin;
    float origin_end_dot = origin_end.dot(origin_int/origin_int.norm());
    float origin_int_norm = origin_int.norm();


    return origin_int_norm > 0 && origin_int_norm < origin_end_dot;
}

bool ray_int_triangle(vec3 origin, vec3 vector, Triangle tri, vec3& intPoint, float eps) {
    // look for any intersections between the ray and triangle (no end point)
    vec3 e1 = tri.b - tri.a;
    vec3 e2 = tri.c - tri.a;
    vec3 h = vector.cross(e2);
    float a = e1.dot(h);

    // if ray is parallel to triangle
    if (fabsf(a) < eps) { return false; }

    float f = 1 / a;
    vec3 s = origin - tri.a;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) { return false; }
    vec3 q = s.cross(e1);
    float v = f * vector.dot(q);
    if (v < 0.0f || u + v > 1.0f) { return false; }

    // now find intersection point
    float t = f * e2.dot(q);
    intPoint = origin + vector * t; // will save garbage answer to intpoint if t <= eps
    // std::cout << " inFunction: " << intPoint.x << " " << intPoint.y << " " << intPoint.z << std::endl;
    return t > eps;
}

bool ray_int_triangle(Node3D node, Triangle tri, vec3& intPoint, float eps) {
    vec3 e1 = tri.b - tri.a;
    vec3 e2 = tri.c - tri.a;
    vec3 h = node.vector.cross(e2);
    float a = e1.dot(h);

    // if ray is parallel to triangle
    if (fabsf(a) < eps) { return false; }

    float f = 1 / a;
    vec3 s = node.origin - tri.a;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) { return false; }
    vec3 q = s.cross(e1);
    float v = f * node.vector.dot(q);
    if (v < 0.0f || u + v > 1.0f) { return false; }

    // now find intersection point
    float t = f * e2.dot(q);
    intPoint = node.origin + node.vector * t; // will save garbage answer to intpoint if t < eps
    // std::cout << " inFunction: " << intPoint.x << " " << intPoint.y << " " << intPoint.z << std::endl;
    return t > eps;
}

float heading_change(Node3D node, Node3D next_node) {
    if (node.vector.norm() < 1e-9f || next_node.vector.norm() < 1e-9f) { return 0.0f; }
    float heading_change = acosf(node.vector.dot(next_node.vector) / (node.vector.norm() * next_node.vector.norm() + 1e-9f));
    return heading_change;
}

float heading_change(Node3D node, vec3 vector) {
    return acosf(node.vector.dot(vector) / (node.vector.norm() * vector.norm() + 1e-9f));
}

float heading_change(vec3 v0, vec3 v1) {
    return acosf(v0.dot(v1) / (v0.norm() * v1.norm() + 1e-9f));
}

bool loadCSV(const std::string& filename, std::vector<std::vector<float>>& data, int rowlen, char delimiter, bool raw){
    /*
    load a csv file into a vector of vectors
    args:
    - filename: std::string, path to csv file
    - data: std::vector<std::vector<float>>, output data
    */
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        int num_cells = 0;
        while (std::getline(ss, cell, delimiter) && num_cells < rowlen)
        {
            try {
                row.push_back(std::stof(cell)); // Convert string to float and add to row
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number format: " << cell << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "Number out of range: " << cell << std::endl;
                return false;
            }
            if (!raw) {
                ++num_cells;
            }
            // ++num_cells;
        }
        data.push_back(row);
    }

    file.close();
    return true;
}

bool loadCSVbool(const std::string& filename, std::vector<std::vector<bool>>& data) {
    // assume no delimiter
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return false;
    }

    while (std::getline(file, line)) {
        std::vector<bool> row;
        for (const auto& c : line) {
            try {
                row.push_back(c == '1');
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number format: " << c << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "Number out of range: " << c << std::endl;
                return false;
            }
        }
        data.push_back(row);
    }
    return true;
}

void saveCSVbool(const std::string& filename, const std::vector<std::vector<bool>>& data) {
    /*
    * Save 2d std::vector float to a csv file
    * @param filename: std::string, path to save file
    * @param data: std::vector<std::vector<float>>, data to save
    */


    // Open the file in output mode
    std::ofstream file(filename);
    
    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (const auto& value : row) {
            file << (value ? "1" : "0"); // Write value
        }
        file << "\n"; // End of row
    }
    file.close();
}

void saveCSV(const std::string& filename, const std::vector<std::vector<float>>& data) {
    /*
    * Save 2d std::vector float to a csv file
    * @param filename: std::string, path to save file
    * @param data: std::vector<std::vector<float>>, data to save
    */


    // Open the file in output mode
    std::ofstream file(filename);
    
    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    // Iterate over the rows
    size_t n_rows = data.size();
    for (size_t i = 0; i < n_rows; ++i) {
        // Iterate over the columns
        size_t n_cols = data[i].size();
        for (size_t j = 0; j < n_cols; ++j) {
            // file << std::fixed << std::setprecision(6) << data[i][j]; // Write value
            file << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << data[i][j];
            if (j < n_cols - 1) {
                file << ","; // Separate values with a comma
            }
        }
        file << "\n"; // End of row
    }

    // Close the file
    file.close();
}

void saveCSVsizet(const std::string& filename, const std::vector<std::vector<size_t>>& data) {
    /*
    * Save 2d std::vector float to a csv file
    * @param filename: std::string, path to save file
    * @param data: std::vector<std::vector<float>>, data to save
    */


    // Open the file in output mode
    std::ofstream file(filename);
    
    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    // Iterate over the rows
    size_t n_rows = data.size();
    for (size_t i = 0; i < n_rows; ++i) {
        // Iterate over the columns
        size_t n_cols = data[i].size();
        for (size_t j = 0; j < n_cols; ++j) {
            file << data[i][j]; // uints have no decimals - just save them
            if (j < n_cols - 1) {
                file << ","; // Separate values with a comma
            }
        }
        file << "\n"; // End of row
    }

    // Close the file
    file.close();
}

void loadCube(std::vector<std::vector<std::vector<float>>>& data, float xs, float xf) {
    /*
    * instantiate cube triangles into data
    * @param data: std::vector<std::vector<std::vector<float>>>, output data
    */
    data = {
        {{xf, 1,-1}, {xf,-1,-1}, {xs, 1,-1}, { 0, 0,-1}},  {{xf,-1,-1},  {xs, 1,-1},  {xs,-1,-1}, {0, 0, -1}}, // base
        {{xs, 1, 1}, {xs,-1, 1}, {xs, 1,-1}, {-1, 0, 0}},  {{xs,-1, 1},  {xs, 1,-1},  {xs,-1,-1}, {-1, 0, 0}}, // front
        {{xf, 1,-1}, {xs, 1,-1}, {xs, 1, 1}, { 0, 1, 0}},  {{xf, 1,-1},  {xs, 1, 1},  {xf, 1, 1}, { 0, 1, 0}}, // left
        {{xf, 1, 1}, {xf,-1, 1}, {xs, 1, 1}, { 0, 0, 1}},  {{xf,-1, 1},  {xs, 1, 1},  {xs,-1, 1}, { 0, 0, 1}}, // top
        {{xf,-1, 1}, {xs,-1, 1}, {xf,-1,-1}, { 0,-1, 0}},  {{xf,-1,-1},  {xs,-1,-1},  {xs,-1, 1}, { 0,-1, 0}}, // right
        {{xf, 1,-1}, {xf,-1,-1}, {xf,-1, 1}, { 1, 0, 0}},  {{xf, 1,-1},  {xf, 1, 1},  {xf,-1, 1}, { 1, 0, 0}}  // back
    };
}

void convertFlatToTriangle(const std::vector<std::vector<float>>& flatData, std::vector<Triangle>& triangles, size_t module_idx) {
    /* convert flat 2d object to triangles assuming each triangle is flattened into x1,y1,z1,x2,y2,z2,x3,y3,z3
    * each triangle also has normal -- [3 points, 9 numbers] [1 normal, 3 numbers]
    */
    for (size_t i = 0; i < flatData.size(); i++) {
        triangles.push_back(Triangle(
            vec3(flatData[i][0], flatData[i][1],  flatData[i][2]),  // v0
            vec3(flatData[i][3], flatData[i][4],  flatData[i][5]),  // v1
            vec3(flatData[i][6], flatData[i][7],  flatData[i][8]),  // v2
            vec3(flatData[i][9], flatData[i][10], flatData[i][11]), // normal
            module_idx                                              // module index
        ));
    }
}

void loadCubeOBS(std::vector<OBS>& obsVec) {
    /*
    * instantiate station into obstacle objects
    * @param obsVec: std::vector<OBS>, output data
    */
    // load in cube data
    std::vector<std::vector<std::vector<float>>> cubeData;
    loadCube(cubeData); // -1 to 1 cube

    // load cube data into triangles
    std::vector<Triangle> triCubeFaces;
    vecToTri(cubeData, triCubeFaces);

    // load cube data into a single obstacle
    obsVec.push_back(OBS(triCubeFaces));
}

void loadConvexStationOBS(std::vector<OBS>& obsVec, float scale) {
    /*
    * instantiate station into obstacle objects
    * @param obsVec: std::vector<OBS>, output data
    */
    size_t num_obstacles = 15;
    std::string model_dir = "../data/model_convex/";

    for (size_t i=0; i < num_obstacles; ++i) {
        // load triangle mesh data
        std::string filename = model_dir + std::to_string(i) + "_faces_normals.csv"; // 9 length vector
        std::vector<std::vector<float>> tri_data;

        // each row is a triangle (3 points = 9 numbers) and normals (3 numbers)
        loadCSV(filename, tri_data, 12,',');

        // convert flat data to triangle objects
        std::vector<Triangle> tris;
        convertFlatToTriangle(tri_data, tris, i);

        vec3 offset = vec3(-0.091745f,-0.326011f,0.148212f);

        // vec3 offset = vec3(2.529f, 4.821f, 2.591f);
        // offset += vec3(1.205f/4, 0.0f, 0.775f/4);

        // // change offset to reflect center of gravity
        // offset -= vec3(2.92199515f, 5.14701097f, 2.63653781f);
        // std::cout << "convex offset =" << offset.toString() << std::endl;

        for (size_t j = 0; j < tris.size(); j++) {
            tris[j].n = tris[j].n / tris[j].n.norm();
            tris[j].a += offset;
            tris[j].b += offset;
            tris[j].c += offset;
            tris[j].a *= scale;
            tris[j].b *= scale;
            tris[j].c *= scale;
        }

        OBS obs = OBS(tris);
        obsVec.push_back(obs);
    }
}

void loadStationOBS(std::vector<OBS>& obsVec, float scale) {
    /*
    * instantiate station into obstacle objects
    * @param obsVec: std::vector<OBS>, output data
    */
    size_t num_obstacles = 10;
    std::string model_dir = "../data/model_remeshed/";

    for (size_t i=0; i < num_obstacles; ++i) {
        // load triangle mesh data
        std::string filename = model_dir + std::to_string(i) + "_faces_normals.csv"; // 9 length vector
        std::vector<std::vector<float>> tri_data;

        // each row is a triangle (3 points = 9 numbers) and normals (3 numbers)
        loadCSV(filename, tri_data, 12, ' ');

        // convert flat data to triangle objects
        std::vector<Triangle> tris;
        convertFlatToTriangle(tri_data, tris, i);

        // change offset to reflect center of gravity
        vec3 offset = vec3(2.92199515f, 5.14701097f, 2.63653781f) * -1;

        for (size_t j = 0; j < tris.size(); j++) {
            tris[j].n = tris[j].n / tris[j].n.norm();
            tris[j].a += offset;
            tris[j].b += offset;
            tris[j].c += offset;
            tris[j].a *= scale;
            tris[j].b *= scale;
            tris[j].c *= scale;
        }

        OBS obs = OBS(tris);
        obsVec.push_back(obs);
    }
}

void printHistogram(std::vector<float>& data) {
    /*
    * Print a histogram of data
    * @param data: const std::vector<float>&, input data
    */

    // BIN DATA:
    size_t num_bins = 100;

    // get min element in data and move above 0
    float min = *std::min_element(data.begin(), data.end());
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= min;
    }

    // get max element in data
    float max = *std::max_element(data.begin(), data.end());

    // get bin width
    float bin_width = max / static_cast<float>(num_bins);

    // hist for counting number of elements in each bin
    std::vector<size_t> hist(num_bins, 0);

    for (size_t i = 0; i < data.size(); ++i) {
        // get bin to increment
        size_t bin = static_cast<size_t>(data[i] / bin_width);

        // if data[i] is max value, put in last bin
        if (bin == num_bins) {
            bin -= 1;
        }

        // increment bin
        hist[bin] += 1;
    }

    // get max bin height
    size_t max_bin = *std::max_element(hist.begin(), hist.end());

    for (size_t i = max_bin; i > 0; i--) {
        std::cout << std::setw(2) << i << ": ";
        for (size_t j = 0; j < hist.size(); ++j) {
            if (hist[j] >= i) {
                std::cout << "#";
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }

    // print x-axis
    std::cout << "    ";
    for (size_t i = 0; i < num_bins+5; ++i) {
        std::cout << "-";
    }
    std::cout << std::endl;

    // print bin numbers
    std::cout << "    ";
    for (size_t i = 0; i < num_bins; i += 6) {
        std::cout << "|     ";
    }
    std::cout << "|" << std::endl;

    std::cout << "  ";
    for (size_t i = 0; i <= num_bins+5; i += 6) {
        std::cout << std::to_string(std::exp((max/num_bins * i) + min)).substr(0, 5);
        std::cout << " ";
    }
    std::cout << std::endl;
}


void loadVxStationOBS(std::vector<OBS>& obsVec, float scale) {
    /*
    * instantiate station into obstacle objects
    * @param obsVec: std::vector<OBS>, output data
    */
    std::string model_dir = "../data/model_remeshed/";

    // load triangle mesh data
    std::string filename = model_dir + "vxc_faces_normals.csv"; // 9 length vector
    std::vector<std::vector<float>> tri_data;

    // each row is a triangle (3 points = 9 numbers) and normals (3 numbers)
    loadCSV(filename, tri_data, 12, ',');

    // convert flat data to triangle objects
    std::vector<Triangle> tris;
    convertFlatToTriangle(tri_data, tris, 0);

    // change offset to reflect center of gravity
    // vec3 offset = vec3(2.92199515f, 5.14701097f, 2.63653781f) * -1;
    vec3 offset = vec3( 0.09087864f,  0.75695956f, -0.10063456f) * -1;

    for (size_t j = 0; j < tris.size(); j++) {
        tris[j].n = tris[j].n / tris[j].n.norm();
        tris[j].a += offset;
        tris[j].b += offset;
        tris[j].c += offset;
        tris[j].a *= scale;
        tris[j].b *= scale;
        tris[j].c *= scale;
    }

    OBS obs = OBS(tris);
    obsVec.push_back(obs);
}

void vecToTri(const std::vector<std::vector<std::vector<float>>>& data, std::vector<Triangle>& tris) {
    /*
    * Convert a vector of vectors of vectors to a vector of triangles
    * @param data: std::vector<std::vector<std::vector<float>>>, input data
    * @param vec: std::vector<Triangles>, output data
    */
    for (size_t i = 0; i < data.size(); ++i) {
        tris.push_back(Triangle(
            vec3(data[i][0][0], data[i][0][1], data[i][0][2]),
            vec3(data[i][1][0], data[i][1][1], data[i][1][2]),
            vec3(data[i][2][0], data[i][2][1], data[i][2][2]),
            vec3(data[i][3][0], data[i][3][1], data[i][3][2])
        ));
    }
}

bool allTrue(const std::vector<TriangleCoverage>& arr, size_t module_idx) {
    /*
    * Check if all elements.covered in an array are true
    * @param arr: const std::vector<TriangleCoverage>&, input array
    * @return bool, true if all elements are true
    */
    for (size_t i = 0; i < arr.size(); ++i) {
        if (!arr[i].covered && arr[i].module_idx == module_idx) {
            return false;
        }
    }
    return true;
}

bool allTrue(const std::vector<TriangleCoverage>& arr) {
    /*
    * Check if all elements.covered in an array are true
    * @param arr: const std::vector<TriangleCoverage>&, input array
    * @return bool, true if all elements are true
    */
    for (size_t i = 0; i < arr.size(); ++i) {
        if (!arr[i].covered) {
            return false;
        }
    }
    return true;
}

bool allTrue(const std::vector<bool>& arr) {
    /*
    * Check if all elements in an array are true
    * @param arr: const bool*, input array
    * @param len: size_t, length of array
    * @return bool, true if all elements are true
    */
    std::cout << "Checking all true, no moduleidx" << std::endl;
    for (size_t i = 0; i < arr.size(); ++i) {
        if (!arr[i]) {
            return false; 
        }
    }
    return true;
}

void numTrue(const std::vector<bool>& arr, size_t& num_true) {
    /*
    * Count the number of true elements in an array
    * @param arr: const bool*, input array
    * @param len: size_t, length of array
    * @param num_true: size_t&, output number of true elements
    */
    num_true = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i]) { 
            ++num_true; 
        }
    }
}

bool allZeroGain(const std::vector<VPCoverageGain>& arr) {
    /*
    * Check if all elements.gain in an array are zero
    * @param arr: const std::vector<VPCoverageGain>&, input array
    * @param all_zero: bool&, output true if all elements are zero
    */
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i].gain > 0) {
            return false;
        }
    }
    return true;
}

void getCoverage(const std::vector<Viewpoint>& viewpoints, const std::vector<Triangle*>& triangles, std::vector<std::vector<bool>>& coverage_map) {
    /*
    * Get the coverage map for a set of viewpoints and triangles
    * @param viewpoints: const std::vector<Viewpoint>&, input viewpoints
    * @param triangles: const std::vector<Triangle>&, input triangles
    * @param coverage_map: bool**, output coverage map
    */
    std::ostringstream message;
    auto begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < viewpoints.size(); ++i) {
        std::vector<bool> coverage;
        cuda_kernel_coverage(viewpoints[i], triangles, coverage);
        for (size_t j = 0; j < coverage.size(); j++) {
            coverage[j] = !coverage[j];
        }
        coverage_map.push_back(coverage);
        auto now = std::chrono::high_resolution_clock::now();
        // Calculate the duration
        std::chrono::duration<double> duration = now - begin;
        double period = duration.count();

        // Output the duration in seconds
        double seconds_remaining = period * (viewpoints.size() - i) / (i + 1);
        double minutes_remaining = seconds_remaining / 60.0;
        seconds_remaining = std::fmod(seconds_remaining, 60.0);
        message << " Time remaining: " << int(minutes_remaining) << "m " << int(seconds_remaining) << "s";
        displayProgressBar(static_cast<double>(i) / viewpoints.size(), 150, message);
        message.str("");
    }
}

void displayProgressBar(double progress, int width, std::ostringstream& message) {
    // Clear the current line
    std::cout << "\r";

    // Calculate the number of '#' characters
    int pos = static_cast<int>(width * progress);
    
    // Draw the progress bar
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) 
            std::cout << "#";
        else 
            std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100) << "%";

    std::cout << message.str();
    
    // Flush the output to ensure it updates immediately
    std::cout.flush();
}

void getIncidenceAngle(vec3 viewdir, Triangle tri, float& angle) {
    /*
    * Get the incidence angle between a view direction and a triangle (always positive and between 0 and pi)
    * @param viewdir: vec3, input view direction
    * @param tri: Triangle, input triangle
    * @param angle: float&, output incidence angle
    */
    vec3 normal = tri.n;
    angle = acosf(fabsf(viewdir.dot(normal)) / (viewdir.norm() * normal.norm() + 1e-9f));
}

void pinhole_camera_test(
    bool& visible, 
    vec3 pose, 
    vec3 viewdir, 
    vec3 point,
    float hfov, // rad
    float vfov // rad
    ) {
    // calculate the angle between the view direction and the vector from the viewpoint to the intersection point
    vec3 vec = point - pose;
    float norm_dot = vec.dot(viewdir);

    // check if point is behind camera
    if (norm_dot <= 0) {
        visible = false;
        return;
    }

    // project point onto view plane
    float d = 1.0f; // distance from viewpoint to view plane
    float w = 2 * d * tanf(hfov/2);
    float h = 2 * d * tanf(vfov/2);
    vec3 point_proj = (vec/norm_dot - viewdir) * d;
    vec3 v_hat = vec3(
        viewdir.z * cosf( atan2f(viewdir.y, viewdir.x) ),
        viewdir.y * sinf( atan2f(viewdir.y, viewdir.x) ),
        sqrtf(viewdir.x * viewdir.x + viewdir.y * viewdir.y) // 0 to 1
    );
    vec3 u_hat = v_hat.cross(viewdir);

    // check if point is within field of view
    if (abs(point_proj.dot(u_hat)) < w/2 && abs(point_proj.dot(v_hat)) < h/2) {
        visible = true;
        return;
    }

    visible = false;
    return;
}

void cw_acceleration(vec3& acceleration, vec3 point, vec3 velocity) {
    // compute cw acceleration at a point with velocity

    float mu = 3.986e14; // gravitational parameter
    float a = 6.778137e6; // semi-major axis
    float n = std::sqrt(mu / (a * a * a)); // orbital rate 

    // compute drift acceleration
    acceleration.x = 3 * n * n * point.x + 2 * n * velocity.y;
    acceleration.y = -2 * n * velocity.x;
    acceleration.z = -1 * n * n * point.z;
}

// float cw_dv(vec3 start, vec3 end, float speed, size_t N) {
//     // integrate cw drift opposition delta-v from start to end at speed 
//     // N = number of discretization steps to take -- inclusive

//     float mu = 3.986e14; // gravitational parameter
//     float a = 6.778137e6; // semi-major axis
//     float n = std::sqrt(mu / (a * a * a)); // orbital rate 
//     // float g0 = 9.80665; // standard gravity
//     // float m = 5; // mass of the spacecraft - 5 kg
//     // float Isp = 80; // specific impulse

//     std::vector<vec3> displacement;
//     vec3 velocity = (end - start) / (end - start).norm() * speed;
//     std::vector<vec3> acceleration;

//     // calculate the displacement vector
//     for (size_t i = 0; i < N; i++) {
//         float t = static_cast<float>(i) / static_cast<float>(N);
//         vec3 point = start * (1 - t) + end * t;
//         displacement.push_back(point);
//     }

//     // compute drift acceleration
//     for (size_t i = 0; i < N; i++) {
//         vec3 acc;
//         acc.x = 3 * n * n * displacement[i].x + 2 * n * velocity.y;
//         acc.y = -2 * n * velocity.x;
//         acc.z = -1 * n * n * displacement[i].z;
//         acceleration.push_back(acc);
//     }
//     return 0;
// }

float cw_cost(vec3 start, vec3 end, float speed, size_t N) {
    // compute cost to oppose CW disturbance from start to end at speed in kg
    // n = number of discretization steps to take -- inclusive

    float mu = 3.986e14; // gravitational parameter
    float a = 6.778137e6; // semi-major axis
    float n = std::sqrt(mu / (a * a * a)); // orbital rate 
    float g0 = 9.80665; // standard gravity
    float m = 5; // mass of the spacecraft - 5 kg
    float Isp = 80; // specific impulse

    std::vector<vec3> displacement;
    vec3 velocity = (end - start) / (end - start).norm() * speed;
    std::vector<vec3> acceleration;

    // calculate the displacement vector
    for (size_t i = 0; i < N; i++) {
        float t = static_cast<float>(i) / static_cast<float>(N);
        vec3 point = start * (1 - t) + end * t;
        displacement.push_back(point);
    }

    // compute drift acceleration
    for (size_t i = 0; i < N; i++) {
        vec3 acc;
        acc.x = 3 * n * n * displacement[i].x + 2 * n * velocity.y;
        acc.y = -2 * n * velocity.x;
        acc.z = -1 * n * n * displacement[i].z;
        acceleration.push_back(acc);
    }

    // compute cost
    float impulse = 0;
    float dt = (end - start).norm() / speed / (N);
    for (size_t i = 0; i < N; i++) {
        impulse += acceleration[i].norm() * std::sqrt(dt);
    }

    float cost = impulse * m / (g0 * Isp);

    // Debugging: print CW cost
    // std::cout << "\ncw_cost=";
    // std::cout << std::fixed << std::setprecision(9) << cost; // Write value
    // std::cout << std::endl;

    return cost;
}

float fuel_cost(vec3 pose, vec3 v0, vec3 v1, float speed, float dt) {
    // fuel cost to change directions v0 to v1 and oppose CW disturbance for dt in kg
    vec3 cw_acc = vec3(0.0f, 0.0f, 0.0f);
    cw_acceleration(cw_acc, pose, v0);


    vec3 dv = v1 - v0 - cw_acc * dt; // change in velocity -- integrate cw_acc over dt
    float mf = 5;
    float Isp = 80;
    float m0 = mf * std::exp(dv.norm() / (9.81 * Isp));

    // Debugging: print fuel cost
    // std::cout << "\nfuel_cost=";
    // std::cout << std::fixed << std::setprecision(9) << m0 - mf; // Write value
    // std::cout << std::endl;
    return m0 - mf;
}
// size_t free, total;
// printf("\n");
// cudaMemGetInfo(&free,&total);   
// printf("%d KB free of total %d KB\n",free/1024,total/1024);
void batch_incidence_angle(const std::vector<Viewpoint>& viewpoints, const std::vector<Triangle*>& faces, std::vector<std::vector<float>>& inc_angle_map) {
    // batch incidence angle computation by viewpoint
    size_t batch_size = 10000; // n viewpoints per batch
    size_t n_batches = (viewpoints.size() + batch_size - 1) / batch_size;
    std::ostringstream message;
    message.str("");
    for (size_t i = 0; i < n_batches; i++) {
        displayProgressBar(static_cast<double>(i) / n_batches, 150, message);
        // index batch
        size_t start_idx = i * batch_size;
        size_t end_idx = std::min((i + 1) * batch_size, viewpoints.size());

        // batch viewpoints
        std::vector<Viewpoint> batch_viewpoints(viewpoints.begin() + start_idx, viewpoints.begin() + end_idx);

        // batch incidence angle results
        std::vector<std::vector<float>> batch_inc_angles;

        // compute incidence angles
        cuda_kernel_inc_angle(batch_viewpoints, faces, batch_inc_angles);

        // push back results to inc_angle_map
        for (size_t vp_idx = 0; vp_idx < batch_viewpoints.size(); vp_idx++) {
            inc_angle_map.push_back(batch_inc_angles[vp_idx]);
        }
    }
    displayProgressBar(1.0, 150, message);
    std::cout << std::endl;
}

float compute_fuel_cost_path(const std::string& ufile, const std::string& tfile, float Isp, float m) {
    // load in solution
    std::vector<std::vector<float>> force_data;
    loadCSV("../knot_ocp/ocp_paths/" + ufile, force_data, 3, ' '); // force input

    std::vector<std::vector<float>> time_data;
    loadCSV("../knot_ocp/ocp_paths/" + tfile, time_data, 1); // timestamps

    std::vector<float> dt;
    for (size_t i = 0; i < time_data.size() - 1; i++) {
        dt.push_back(time_data[i+1][0] - time_data[i][0]);
    }


    // integrate force 
    float dv_tot = 0;
    for (size_t i = 0; i < force_data.size()-1; i++) {
        vec3 dv = {
            force_data[i][0] * dt[i] / m,
            force_data[i][1] * dt[i] / m,
            force_data[i][2] * dt[i] / m,
        };
        dv_tot += dv.norm();
    }
    float fuel_cost = m * std::exp(dv_tot / (9.81 * Isp)) - m; // in kg
    return fuel_cost * 1000; // convert to grams
}

float compute_pathtime_path(const std::string& tfile) {
    std::vector<std::vector<float>> time_data;
    loadCSV("../knot_ocp/ocp_paths/" + tfile, time_data, 1); // timestamps
    return time_data[time_data.size()-1][0]; // last element
}

void get_uncoverable_faces(std::vector<OBS>& obsVec, std::vector<size_t>& uncoverable) {
    // clear uncoverables:
    uncoverable.clear();

    // for module, find the 'start' index for proper face indexing
    size_t start_idx = 0;
    for (size_t i = 0; i < obsVec.size(); i++) {
        // for each obstacle, check if each vertex is outside other obstacles

        // first get all vertices with face memberships
        std::vector<vec3> vertices;
        for (size_t j = 0; j < obsVec[i].faces.size(); j++) {
            vertices.push_back( obsVec[i].faces[j].a );
            vertices.push_back( obsVec[i].faces[j].b );
            vertices.push_back( obsVec[i].faces[j].c );
        }

        vertices[0].x = -0.1f;
        vertices[0].y = 0.0f;
        vertices[0].z = 0.3f;

        std::vector<Triangle*> faces_wo_module;
        for (size_t k = 0; k < obsVec.size(); k++) {
            if (k != i) {
                for (size_t j = 0; j < obsVec[k].faces.size(); j++) {
                    faces_wo_module.push_back(&(obsVec[k].faces[j]));
                }
            }
        }

        std::vector<bool> in_collision_convex;
        cuda_kernel_collision_points_vec3(
            vertices,
            faces_wo_module,
            vec3(1e9f, 1e9f, 1e9f), // needs to be collision free
            in_collision_convex
        );

        for (size_t j = 0; j < in_collision_convex.size(); j++) {
            if (in_collision_convex[j]) {
                std::cout << "uncoverable face: " << start_idx + (j/3) << std::endl; // integer division to get index of triangle
                uncoverable.push_back(start_idx + (j/3)); // integer division to get index of triangle
            }
        }

        start_idx += obsVec[i].faces.size();
    }
}

void orientation_moving_average(const std::vector<std::vector<float>> &path_data, std::vector<std::vector<float>> & smoothed_path, size_t window_size) {
    // clear smoothed path
    smoothed_path.clear();
    // moving average of orientation data
    for (size_t i = 0; i < path_data.size(); i++) {
        std::vector<float> smoothed_row(7);
        for (size_t j = 0; j < path_data[i].size(); j++) {
            if (j < 3 || j > 5) {
                smoothed_row[j] = path_data[i][j];
            } else {
                size_t start_idx = (i < window_size) ? 0 : i - window_size;
                size_t end_idx = ((i + window_size) > (path_data.size())) ? path_data.size() : i + window_size;
                for (size_t k = start_idx; k < end_idx; k++) {
                    smoothed_row[j] += path_data[k][j];
                }
                smoothed_row[j] /= (end_idx - start_idx);
            }
        }
        float norm = std::sqrt(smoothed_row[3] * smoothed_row[3] + smoothed_row[4] * smoothed_row[4] + smoothed_row[5] * smoothed_row[5]);
        smoothed_row[3] /= norm;
        smoothed_row[4] /= norm;
        smoothed_row[5] /= norm;
        smoothed_path.push_back(smoothed_row);
    }
}

size_t determine_bin_idx(float aoi, size_t n_bins, float bin_width) {
    // determine bin index for a given angle of incidence
    size_t bin_idx = static_cast<size_t>(aoi / bin_width); // static_cast truncates decimal of float --> rounding down
    return bin_idx;
}

void compute_saturation_path(
    const std::vector<std::vector<float>> &path_data, 
    std::vector<std::vector<float>> &saturation_map, 
    std::vector<std::vector<size_t>> &saturation_bins, 
    size_t n_bins
    ) {
    // saturation map tracks the number of times a face is seen from the path, closest distance seen from, average incidence angle, and minimum incidence angle
    // number of times a face is visible -- count
    // how far it is seen -- take min of distance * don't need this for now
    // average incidence angle --  sum of angles / count
    // minimum incidence angle -- min of each new angle
    // saturation_map = {count, avg_angle, min_angle}
    // saturation_bins = {face0_bins, face1_bins, ...} where face0_bins = {bin0, bin1, ... binN} where bin0 = count of times face0 is seen in bin0 etc. binN holds outliers

    // clear saturation map
    saturation_map.clear();
    saturation_bins.clear();

    // load in station
    std::vector<OBS> obsVec;
    loadStationOBS(obsVec, 4);

    // get viewpoints
    std::vector<Viewpoint> vps;
    for (size_t i = 0; i < path_data.size(); i++) {
        vec3 pose = vec3(path_data[i][0], path_data[i][1], path_data[i][2]);
        vec3 viewdir = vec3(path_data[i][3], path_data[i][4], path_data[i][5]);
        Viewpoint vp = Viewpoint(pose, viewdir, 0);
        vps.push_back(vp);
    }

    // get dt's for 'saturation' in seconds:
    std::vector<float> dts;
    for (size_t i = 0; i < path_data.size() - 1; i++) {
        dts.push_back(path_data[i+1][6] - path_data[i][6]);
    }
    dts.push_back(dts[dts.size()-1]); // repeat last dt


    // get faces:
    std::vector<Triangle*> all_faces;
    for (size_t i = 0; i < obsVec.size(); i++) {
        for (size_t j = 0; j < obsVec[i].faces.size(); j++) {
            all_faces.push_back(&(obsVec[i].faces[j]));
        }
    }

    // set up saturation map. resizing from zero sets all elements to val below
    saturation_map.resize(all_faces.size(), {0.0f, 0.0f, std::numeric_limits<float>::max()});

    // binning behavior
    saturation_bins.resize(all_faces.size(), std::vector<size_t>(n_bins+1, 0));

    // check coverage
    for (size_t i = 0; i < vps.size(); ++i) {
        // get coverage for this viewpoint
        std::vector<bool> coverage;
        cuda_kernel_coverage(vps[i], all_faces, coverage);

        // get incidence angles for this viewpoint
        std::vector<Viewpoint> vp = {vps[i]};
        std::vector<std::vector<float>> inc_angles;
        cuda_kernel_inc_angle(vp, all_faces, inc_angles); // result is n_vp x n_tri. n_vp = 1 so ize = 1 x n_tri


        // true means not visible (intersection before end of ray)
        for (size_t j = 0; j < coverage.size(); j++) {
            coverage[j] = !coverage[j];
        }

        for (size_t tridx = 0; tridx < coverage.size(); tridx++) {
            // if inc angle is valid -- between 0 and 90 degrees
            // AND face is visible
            if (coverage[tridx] && inc_angles[0][tridx] > 0 && inc_angles[0][tridx] < M_PI/2) {
                // increment count 
                saturation_map[tridx][0] += dts[i];

                // accumulate incidence angles
                saturation_map[tridx][1] += inc_angles[0][tridx];

                // update min angle
                if (inc_angles[0][tridx] < saturation_map[tridx][2]) {
                    saturation_map[tridx][2] = inc_angles[0][tridx];
                }

                // determine correct bin index
                size_t bin_idx = determine_bin_idx(inc_angles[0][tridx], n_bins, M_PI/2/n_bins);

                // clamp outliers to last bin (not visible -> invalid angle)
                if (bin_idx > n_bins) {
                    bin_idx = n_bins;
                }

                // increment bin count
                saturation_bins[tridx][bin_idx] += 1;
            }
        }

        double progress = static_cast<double>(i) / vps.size();
        std::ostringstream message;
        message.str("");
        displayProgressBar(progress, 100, message);
    }

    std::vector<size_t>  uncoverable = { 148, 152, 156, 158, 186, 190, 194, 196, 218, 230, 231, 234, 235, 260, 272, 305, 316, 318, 333, 334, 380, 381, 392, 393, 396, 397, 422, 467, 478, 480, 495, 496, 728, 730, 733, 735, 744, 745, 746, 747, 749, 753, 758, 760, 763, 765, 774, 775, 779, 780, 781, 783 };

    for (size_t i = 0; i < saturation_map.size(); i++) {
        bool this_uncoverable = false;
        for (size_t j = 0; j < uncoverable.size(); j++) {
            if (i == uncoverable[j]) {
                this_uncoverable = true;
            }
        }

        if (this_uncoverable) {
            // zero out average incidence angle for uncoverable faces
            saturation_map[i][0] = 0.0f;
            saturation_map[i][1] = 0.0f;
            saturation_map[i][2] = 0.0f;

            // zero out bins for uncoverable faces
            for (size_t j = 0; j < n_bins; j++) {
                saturation_bins[i][j] = 0;
            }
        } else {
            // update average incidence angle from accumulated incidence angle and count
            if (saturation_map[i][0] == 0.0f) {
                saturation_map[i][1] = std::numeric_limits<float>::max();
            } else {
                saturation_map[i][1] /= saturation_map[i][0];
            }
        }
    }
}

float compute_coverage_path(const std::vector<std::vector<float>> &path_data, std::vector<bool>& coverage) {

    // load in station
    std::vector<OBS> obsVec;
    loadStationOBS(obsVec, 4);

    // get viewpoints
    std::vector<Viewpoint> vps;
    for (size_t i = 0; i < path_data.size(); i++) {
        vec3 pose = vec3(path_data[i][0], path_data[i][1], path_data[i][2]);
        vec3 viewdir = vec3(path_data[i][3], path_data[i][4], path_data[i][5]);
        Viewpoint vp = Viewpoint(pose, viewdir, 0);
        vps.push_back(vp);
    }

    // get faces:
    std::vector<Triangle*> all_faces;
    for (size_t i = 0; i < obsVec.size(); i++) {
        for (size_t j = 0; j < obsVec[i].faces.size(); j++) {
            all_faces.push_back(&(obsVec[i].faces[j]));
        }
    }

    // check coverage
    std::vector<bool> covered(all_faces.size(), false);
    for (size_t i = 0; i < vps.size(); ++i) {
        std::vector<bool> coverage;
        cuda_kernel_coverage(vps[i], all_faces, coverage);
        for (size_t j = 0; j < coverage.size(); j++) {
            coverage[j] = !coverage[j];
        }
        for (size_t tridx = 0; tridx < coverage.size(); tridx++) {
            if (coverage[tridx]) {
                covered[tridx] = true;
            }
        }
        double progress = static_cast<double>(i) / vps.size();
        std::ostringstream message;
        message.str("");
        displayProgressBar(progress, 100, message);
    }

    std::cout << std::endl;

    std::vector<size_t>  uncoverable = { 148, 152, 156, 158, 186, 190, 194, 196, 230, 231, 234, 235, 260, 272, 305, 316, 318, 333, 334, 380, 381, 392, 393, 396, 397, 422, 467, 478, 480, 495, 496, 728, 730, 733, 735, 744, 745, 746, 747, 749, 753, 758, 760, 763, 765, 774, 775, 779, 780, 781, 783 };
    size_t num_covered = 0;
    for (size_t i = 0; i < covered.size(); i++) {
        if (covered[i]) {
            num_covered++;
        } else {
            for (size_t j = 0; j < uncoverable.size(); j++) {
                if (i == uncoverable[j]) {
                    covered[i] = true;
                    // num_covered++;
                }
            }
        }
    }
    coverage=covered;

    return static_cast<float>(num_covered) / static_cast<float>(all_faces.size() - uncoverable.size());
}
float compute_coverage_file(const std::string& file, std::vector<bool>& coverage) {
    // load in solution
    std::vector<std::vector<float>> path_data;
    loadCSV(file, path_data, 7);
    return compute_coverage_path(path_data, coverage);
}

float compute_smoothed_coverage(const std::vector<std::vector<float>> &path_data, std::vector<bool>& coverage) {
    // smooth path
    std::vector<std::vector<float>> smoothed_path;
    orientation_moving_average(path_data, smoothed_path, 5);

    return compute_coverage_path(smoothed_path, coverage);
}


std::string getnum(float num) {
    float rem = (num - static_cast<int>(num));
    // std::cout << " rem=" << rem << " round=" << std::round(rem*1000 + 0.5);
    std::string rem_str = std::to_string(static_cast<int>(std::round(rem*1000)));
    size_t num_zeros = 3 - rem_str.size();
    if (num_zeros > 0) {
        for (size_t i = 0; i < num_zeros; i++) {
            rem_str = "0" + rem_str;
        }
    }
    rem_str = removeTrailingZeros(rem_str);
    return std::to_string(static_cast<int>(num)) + "_" + rem_str;

}

std::string removeTrailingZeros(const std::string& str) {
    std::string result = str;
    result.erase(result.find_last_not_of('0') + 1, std::string::npos);

    // If the resulting string is empty or contains only a decimal point, return "0"
    if (result.empty() || result == ".") {
        return "0";
    }

    return result;
}

vec3 slerp(vec3 v0, vec3 v1, float t) {
    // spherical linear interpolation
    // v0 and v1 are unit vectors
    float norm_dot = v0.dot(v1) / (v0.norm() * v1.norm());
    float theta = acosf(norm_dot);
    return v0 * (sinf((1 - t) * theta) / sinf(theta)) + v1 * ( sinf(t * theta) / sinf(theta) );
}

void writeTriToFile(const Triangle &tri, std::ofstream &file) {
    // save each vertex
    file << "v ";
    file << std::fixed << std::setprecision(6) << tri.a.x; // Write value
    file << " "; // Space
    file << std::fixed << std::setprecision(6) << tri.a.y; // Write value
    file << " "; // Space
    file << std::fixed << std::setprecision(6) << tri.a.z; // Write value
    file << "\nv "; // Space

    file << std::fixed << std::setprecision(6) << tri.b.x; // Write value
    file << " "; // Space
    file << std::fixed << std::setprecision(6) << tri.b.y; // Write value
    file << " "; // Space
    file << std::fixed << std::setprecision(6) << tri.b.z; // Write value
    file << "\nv "; // Space

    file << std::fixed << std::setprecision(6) << tri.c.x; // Write value
    file << " "; // Space
    file << std::fixed << std::setprecision(6) << tri.c.y; // Write value
    file << " "; // Space
    file << std::fixed << std::setprecision(6) << tri.c.z; // Write value
    file << "\n"; // End of row
}

void saveObjFile(const std::vector<OBS> &obsVec, const std::string& filename) {
    // Open the file in output mode
    std::ofstream file(filename);
            
    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "# OBJ file\n";
    file << "# Number of meshes: " << obsVec.size() << "\n";
    file << "mtllib sponza.mtl\n";

    // for each mesh, save a new mesh obj
    for (size_t i = 0; i < obsVec.size(); i++) {
        // save obstacles to obj file
        std::vector<std::vector<size_t>> vert_buffer;

        // vertex normal buffer
        std::vector<std::vector<float>> vert_norm_buffer;

        // get current mesh
        OBS cur_obs = obsVec[i];

        // comments
        file << "# Mesh " << i << " vertices\n";

        // save each vertex
        for (size_t j = 0; j < cur_obs.faces.size(); j++) {
            // current face
            Triangle cur_tri = cur_obs.faces[j];

            // write triangle to file
            writeTriToFile(cur_tri, file);

            // save vertex idxs to buffer
            vert_buffer.push_back({j*3 + 1, j*3 + 2, j*3 + 3});

            // save vertex normal to buffer
            vert_norm_buffer.push_back({cur_tri.n.x, cur_tri.n.y, cur_tri.n.z});
        }

        // write all vertex normals to file
        file << "\n# Vertex normals for mesh " << i << ":\n";
        size_t num_normals = 0;
        for (size_t j = 0; j < vert_norm_buffer.size(); j++) {
            file << "vn " << std::setprecision(6) << vert_norm_buffer[j][0] << " " << std::setprecision(6) << vert_norm_buffer[j][1] << " " << std::setprecision(6) << vert_norm_buffer[j][2] << "\n";
            num_normals++;
        }

        // comments
        file << "\n";
        file << "# Vertex idxs for each mesh face " << i << "\n";

        // get material
        file << "usemtl floor" << i << "\n";

        // write vertex idxs to file
        for (size_t j = 0; j < vert_buffer.size(); j++) {
            file << "f " << vert_buffer[j][0] << "//" << j+1 << " " << vert_buffer[j][1] << "//" << j+1 << " " << vert_buffer[j][2] << "//" << j+1 << "\n";
        }

        file << "\n";
    }

    // Close the file
    file.close();
}