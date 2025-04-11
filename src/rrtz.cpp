#include "cuda_kernels.h"
#include "limit_struct.hpp"
#include "node3d_struct.hpp"
#include "obs.hpp"
#include "plane_struct.hpp"
#include "rrtz.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <limits>

RRTZ::RRTZ() {
    this->start = vec3();
    this->goal = vec3();
    this->max_nodes = 0;
    this->tree_nodes = std::vector<Node3D>();
    this->obs = std::vector<OBS>();
}

RRTZ::RRTZ(
    vec3 start,
    vec3 goal,
    std::vector<OBS> obs,
    Limit limits,
    size_t max_nodes,
    bool debug
    ) {

    // save problem config
    this->start = start;
    this->goal = goal;
    this->max_nodes = max_nodes;

    // initialize tree
    this->tree_nodes = std::vector<Node3D>();
    this->tree_nodes.push_back(Node3D(start, start, 0, 0, 0)); // root node

    // get env
    this->obs = obs;
    this->limits = limits;

    // initialize best solution
    this->best_cost = std::numeric_limits<float>::infinity();
    this->best_idx = -1;

    // pass in debug setting
    this->debug = debug;
    if (this->debug) {
        std::cout << "RRTZ initialized with start=" << start.toString() << " goal=" << goal.toString() << std::endl;
        std::cout << "> debug mode enabled" << std::endl;
    }

    // get list of triangles for ray collision check
    for (size_t obs_idx = 0; obs_idx < this->obs.size(); obs_idx++) {
        for (size_t face_idx = 0; face_idx < this->obs[obs_idx].faces.size(); face_idx++) {
            this->triangles.push_back(&(this->obs[obs_idx].faces[face_idx]));
        }
    }
}

bool RRTZ::run(std::vector<vec3>& path) {
    // initialize best solution
    size_t num_failed_iterations = 0;
    while (!this->terminate()) {
        // random number between 0 and 1
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float cost = std::numeric_limits<float>::infinity();

        // if (this->debug) { std::cout << "\rIteration=" << this->tree_nodes.size() << " | Failed Iterations=" << num_failed_iterations << std::flush; }

        if (r < 0.5 && this->tree_nodes.size() > 1) { 
            // extend to goal
            Plane plane = this->sample_plane(this->goal);
            num_failed_iterations = 0;
            if (this->extend(plane, cost)) {
                // success to goal --> save index of this leaf node
                if (cost < this->best_cost) {
                    this->best_cost = cost;
                    this->best_idx = this->tree_nodes.size() - 1;
                    if (this->debug) {
                        std::vector<vec3> best_path;
                        this->reconstruct_path(this->best_idx, best_path);
                        std::cout << "\n Found new best path:" << std::endl;
                        for (size_t i = 0; i < best_path.size(); i++) {
                            std::cout << best_path[i].toString() << std::endl;
                        }
                        std::cout << " Improved Cost=" << this->best_cost << std::endl;
                        std::cout << std::endl;
                    }
                }
            } else { 
                num_failed_iterations++; 
                if (num_failed_iterations > 1000) {
                    std::cout << num_failed_iterations << " failed iterations. quitting" << std::endl;
                    return false;
                }
            }
        } else {
            // extend to random plane
            bool success = false;
            num_failed_iterations = 0;
            while (!success) {
                Plane plane = this->sample_plane();
                success = this->extend(plane, cost);
                num_failed_iterations++;
                if (num_failed_iterations > 1000) {
                    std::cout << num_failed_iterations << " failed iterations. quitting" << std::endl;
                    return false;
                }
            }
        }
    }

    // reconstruct path
    if (this->best_cost != std::numeric_limits<float>::infinity()) {
        this->reconstruct_path(this->best_idx, path);
        return true;
    }
    return false;
}

float RRTZ::getBestCost() {
    return this->best_cost;
}

bool RRTZ::terminate() {
    // TODO: implement convergence criteria condition
    // currently: end at max_iterations
    return this->tree_nodes.size() >= this->max_nodes || this->best_cost < 1e-5f;
}

void RRTZ::reconstruct_path(size_t parent_idx, std::vector<vec3>& path) {
    path.push_back(this->tree_nodes[parent_idx].end);
    while (parent_idx != 0) {
        path.insert(path.begin(),this->tree_nodes[parent_idx].origin);
        parent_idx = this->tree_nodes[parent_idx].parent_idx;
    }
}

Plane RRTZ::sample_plane() {
    vec3 pose = vec3(
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (this->limits.xmax - this->limits.xmin) + this->limits.xmin,
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (this->limits.ymax - this->limits.ymin) + this->limits.ymin,
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (this->limits.zmax - this->limits.zmin) + this->limits.zmin
    );
    vec3 normal = vec3(
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f,
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f,
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f
    );
    // std::cout << "\nSampled plane: point= " << pose.toString() << " | normal= " << normal.toString() << std::endl;
    return Plane(normal, pose);
}

Plane RRTZ::sample_plane(vec3 point) {
    vec3 normal = vec3(
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f,
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f,
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f
    );
    // std::cout << "\nSampled normal: point= " << point.toString() << " | normal= " << normal.toString() << std::endl;
    return Plane(normal, point);
}

bool RRTZ::extend(Plane p, float& cost) {
    // return true if the tree is successfully extended
    // populate cost with cost to extended-to node

    // generate a node from start to the plane point
    if (!(this->collision(this->start, p.point))) {
        // std::cout << "->No collision detected." << std::endl;
        Node3D start_node = Node3D(this->start, p.point, this->tree_nodes.size(), 0, 0);
        this->add_node_to_tree(start_node, cost);
        return true;
    }

    // try connecting plane with other nodes
    std::vector<Node3D*> near_nodes = std::vector<Node3D*>();
    std::vector<vec3> intersection_points = std::vector<vec3>();

    // find lowest cost candidate node
    Node3D *best_node_ptr = nullptr;
    vec3 *best_int_point_ptr = nullptr;
    float best_cost = std::numeric_limits<float>::infinity();

    if (this->get_near_nodes(p, near_nodes, intersection_points)) {
        // rays to send to gpu
        std::vector<vec3> start_ray = std::vector<vec3>();
        std::vector<vec3> end_ray = std::vector<vec3>();
        for (size_t i = 0; i < near_nodes.size(); i++) {
            // get rays to send to gpu
            start_ray.push_back(intersection_points[i]);
            end_ray.push_back(p.point);

            // CPU code:
            // if (!this->collision(intersection_points[i], p.point)) {
            //     float node_cost = (*near_nodes[i]).cost + heading_change(*near_nodes[i], p.point-intersection_points[i]);
            //     if (node_cost < best_cost) {
            //         best_cost = cost;
            //         best_node_ptr = near_nodes[i];
            //         best_int_point_ptr = &intersection_points[i];
            //     }
            // }
        }
        bool* collisions = new bool[near_nodes.size()];
        cuda_kernel_many_ray(start_ray, end_ray, this->triangles, collisions);
        // for (size_t i = 0; i < near_nodes.size(); i++) {
        //     std::cout << start_ray[i].toString() << " -> " << end_ray[i].toString() << " : "<< collisions[i] << " | ";
        // }
        for (size_t i = 0; i < near_nodes.size(); i++) {
            if (!collisions[i]) {
                float node_cost = (*near_nodes[i]).cost + heading_change(*near_nodes[i], p.point-intersection_points[i]);
                if (node_cost < best_cost) {
                    best_cost = node_cost;
                    best_node_ptr = near_nodes[i];
                    best_int_point_ptr = &intersection_points[i];
                }
            }
        }
        delete[] collisions;
    }

    // if a collisionfree candidate node was found, add it to the tree
    if (best_node_ptr != nullptr) {
        Node3D new_node = Node3D(*best_int_point_ptr, p.point, this->tree_nodes.size(), (*best_node_ptr).idx);
        this->add_node_to_tree(new_node, cost);
        return true;
    }
    return false;
}

void RRTZ::print_node(Node3D node) {
    std::cout << " origin=" << node.origin.toString() << " end=" 
    << node.end.toString() << " vector=" << node.vector.toString() 
    << " cost=" << node.cost << " idx=" << node.idx << " parent idx="
    << node.parent_idx << std::endl;
}

bool RRTZ::get_near_nodes(Plane plane, std::vector<Node3D*>& near_nodes, std::vector<vec3>& int_points) {
    // GPU code:
    std::vector<vec3> start_ray = std::vector<vec3>();
    std::vector<vec3> end_ray = std::vector<vec3>();
    for (size_t i = 0; i < this->tree_nodes.size(); i++) {
        start_ray.push_back(this->tree_nodes[i].origin);
        end_ray.push_back(this->tree_nodes[i].end);
    }

    bool* collisions = new bool[this->tree_nodes.size()];
    vec3* int_point_arr = new vec3[this->tree_nodes.size()];
    cuda_kernel_ray_int_plane(
        start_ray,
        end_ray,
        plane.point,
        plane.normal,
        collisions,
        int_point_arr
    );

    // std::cout << "Plane check: point=" << plane.point.toString() << " | normal=" << plane.normal.toString() << std::endl; 
    for (size_t i = 0; i < this->tree_nodes.size(); i++) {
        // std::cout << "RAY " << i << " collision: " << collisions[i];
        // std::cout << " | start_ray=" << start_ray[i].toString() << " | end_ray=" << end_ray[i].toString();
        // std::cout << " | int_point=" << int_point_arr[i].toString() << std::endl;
        if (collisions[i]) {
            near_nodes.push_back(&(this->tree_nodes[i]));
            int_points.push_back(int_point_arr[i]);
        }
    }

    // CPU code:
    // for (size_t i = 0; i < this->tree_nodes.size(); i++) {
    //     vec3 intPoint;
    //     if (ray_int_plane(this->tree_nodes[i], plane, 1e-9f, intPoint)) {
    //         near_nodes.push_back(&(this->tree_nodes[i]));
    //         int_points.push_back(intPoint);
    //     }
    // }
    return !near_nodes.empty();
}

void RRTZ::add_node_to_tree(Node3D& node, float& cost) {
    node.cost = this->tree_nodes[node.parent_idx].cost + heading_change(this->tree_nodes[node.parent_idx], node);
    cost = node.cost;
    this->tree_nodes.push_back(node);
}

bool RRTZ::check_bounds(vec3 point) {
    return point.x >= this->limits.xmin && point.x <= this->limits.xmax &&
           point.y >= this->limits.ymin && point.y <= this->limits.ymax &&
           point.z >= this->limits.zmin && point.z <= this->limits.zmax;
}

bool RRTZ::collision(vec3 origin, vec3 end) {
    // set up rays to send to gpu
    std::vector<vec3> start_ray = std::vector<vec3>();
    start_ray.push_back(origin);
    std::vector<vec3> end_ray = std::vector<vec3>();
    end_ray.push_back(end);

    // get collisions from gpu
    bool ret;
    bool* ray_collision = &ret;
    cuda_kernel_many_ray(start_ray, end_ray, this->triangles, ray_collision);

    // cpu code:
    // for (size_t i = 0; i < this->obs.size(); i++) {
    //     if (this->obs[i].collision(origin, end)) {
    //         return true;
    //     }
    // }
    return ret;
}