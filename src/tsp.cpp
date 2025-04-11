#include "cost_matrix.hpp"
#include "tsp.hpp"
#include "tsp_waypoint_struct.hpp"
#include "vec3_struct.hpp"
#include "viewpoint_struct.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>

TSP::TSP() {
    this->cm = CostMatrix();
    this->path = std::vector<TSPWaypoint>();
}

TSP::TSP(CostMatrix cm) {
    this->cm = cm;
    this->path = std::vector<TSPWaypoint>();
    this->n_vp = cm.getNVP();

    // make sure start viewpoint has module membership
    for (size_t i = 0; i < this->n_vp; i++) {
        TSPWaypoint wpt;
        wpt.vp_idx = i;
        wpt.module_idx = this->cm.viewpoints[i].module_idx;
        this->nodes.push_back(wpt);
    }
}

void TSP::reassignModuleMembership() {
    // std::vector<size_t> module_membership = {0,4,2,4,3,3,3,1,1,4};
    std::vector<size_t> module_membership = {0,3,2,3,2,2,2,1,1,3};
    for (size_t i = 0; i < this->nodes.size(); i++) {
        if (this->nodes[i].module_idx == 2) {
            if (this->cm.viewpoints[i].pose.y < 2.85f) {
                this->nodes[i].module_idx = 3;
            } else {
                this->nodes[i].module_idx = 2;
            }
        } else {
            this->nodes[i].module_idx = module_membership[this->nodes[i].module_idx];
        }
    }
}

void TSP::globalOpt() {
    // set all module idx values to 0 for global optimization
    for (size_t i = 0; i < this->nodes.size(); i++) {
        this->nodes[i].module_idx=0;
    }
}

void TSP::loadCM(int vgd, Viewpoint start, bool vx) {
    std::string vx_str = vx ? "_vx" : "";
    std::string viewpoint_file = "../data" + vx_str + "/coverage_viewpoint_sets/coverage_" + std::to_string(vgd) + "m_vp_set.csv";
    std::string cost_matrix_file = "../data" + vx_str + "/tsp/" + std::to_string(vgd) + "m_cost_matrix.csv";
    std::string path_matrix_file = "../data" + vx_str + "/tsp/" + std::to_string(vgd) + "m_path_matrix.csv";
    this->cm.loadViewpoints(viewpoint_file, start);
    this->cm.loadCostMatrix(cost_matrix_file);
    this->cm.loadPathMatrix(path_matrix_file);
    this->n_vp = cm.getNVP();

    // make sure start viewpoint has module membership
    for (size_t i = 0; i < this->n_vp; i++) {
        TSPWaypoint wpt;
        wpt.vp_idx = i;
        wpt.module_idx = this->cm.viewpoints[i].module_idx;
        this->nodes.push_back(wpt);
    }
}

void TSP::greedyInit() {
    // Greedy algorithm
    // https://en.wikipedia.org/wiki/Greedy_algorithm
    // https://en.wikipedia.org/wiki/Travelling_salesman_problem

    // add start viewpoint to path
    this->path.push_back(this->nodes[0]);
    this->nodes.erase(this->nodes.begin());

    while (this->nodes.size() > 0) {
    // for (size_t j = 0; j < 5; j++) {
    //     if (this->nodes.size() == 0) {
    //         break;
    //     }
        std::cout << "Nodes left: " << this->nodes.size() << std::endl;
        for (size_t i = 0; i < this->nodes.size(); i++) {
            std::cout << this->nodes[i].toString() << " ";
        }
        std::cout << std::endl;
        // get best node to insert to path
        float best_cost;
        size_t node_idx;
        size_t idx_to_insert = this->nearest(best_cost, node_idx);

        if (node_idx >= this->nodes.size()) {break;}

        std::cout << "Node to insert: " << this->nodes[node_idx].toString() << " at index: " << idx_to_insert << " with cost: " << best_cost << std::endl;

        // get iterators to perform insertion
        auto it_erase = this->nodes.begin() + node_idx;
        auto it_insert = this->path.begin() + idx_to_insert;

        // insert node into path
        this->path.insert(it_insert, this->nodes[node_idx]);

        // erase node from nodes
        this->nodes.erase(it_erase);
        std::cout << "Greedy path: ";
        for (size_t i = 0; i < this->path.size(); i++) {
            std::cout << this->path[i].toString() << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Greedy path: ";
    for (size_t i = 0; i < this->path.size(); i++) {
        std::cout << this->path[i].toString() << " ";
    }
    std::cout << std::endl << "Cost: " << this->pathCost() << std::endl;
}

bool TSP::checkModuleContinuity() {
    // check if path has continuity in module membership
    std::vector<size_t> module_found = std::vector<size_t>(1, this->path[0].module_idx);
    for (size_t i = 1; i < this->path.size(); i++) {
        if (this->path[i].module_idx != this->path[i-1].module_idx) {
            if(std::find(module_found.begin(), module_found.end(), this->path[i].module_idx) != module_found.end()) {
                return false;
            } else {
                module_found.push_back(this->path[i].module_idx);
            }
        }
    }
    return true;
}

void TSP::testSwappedIdxs(std::vector<std::vector<size_t>> swap_idxs, float& best_new_cost, float& new_cost) {
    // swap idxs of path in swap_idxs and test if path is better than current best
    std::vector<float> costs = {best_new_cost};

    for (size_t i = 0; i < swap_idxs.size(); i++) {
        size_t idx0 = swap_idxs[i][0];
        size_t idx1 = swap_idxs[i][1];

        // get iterators to perform swap
        auto it0 = this->path.begin() + idx0;
        auto it1 = this->path.begin() + idx1;

        // swap
        std::reverse(it0, it1);

        // check local continuity
        if (!this->checkModuleContinuity()) {
            std::reverse(it0, it1);
            swap_idxs.erase(swap_idxs.begin() + i);
            i--;
        } else {
            // check viable path cost
            costs.push_back(this->pathCost());
        }

        // // if new cost is worse, swap back
        // if (new_cost >= best_new_cost || !this->checkModuleContinuity()) {
        //     std::reverse(it0, it1);
        //     new_cost = best_new_cost; // 'reset' new cost to previous best
        // } else {
        //     std::cout << "New Best: ";
        //     for (size_t i = 0; i < this->path.size(); i++) {
        //         std::cout << this->path[i].vp_idx << " ";
        //     }
        //     std::cout << "| Cost=" << std::to_string(new_cost) << " < " << std::to_string(best_new_cost) << std::endl;
        //     best_new_cost = new_cost;
        // }
    }

    // find the maximum element iterator
    auto min_it = std::min_element(costs.begin(), costs.end());

    // get indexof the maximum element
    size_t argmin = std::distance(costs.begin(), min_it);

    // save best cost
    if (costs[argmin] < best_new_cost) {
        best_new_cost = costs[argmin];
        std::cout << "New Best: " << std::to_string(best_new_cost) << std::endl;
    }

    for (size_t i = swap_idxs.size() - 1; i >= argmin && i < swap_idxs.size(); i--) {
        size_t idx0 = swap_idxs[i][0];
        size_t idx1 = swap_idxs[i][1];

        // get iterators to perform swap
        auto it0 = this->path.begin() + idx0;
        auto it1 = this->path.begin() + idx1;

        // swap back
        std::reverse(it0, it1);
    }
}

void TSP::twoOpt() {
    // 2-opt algorithm
    // https://en.wikipedia.org/wiki/2-opt
    // https://en.wikipedia.org/wiki/Travelling_salesman_problem
    float best_new_cost = this->pathCost();
    float best_cost = best_new_cost + 1e-5;
    float new_cost = std::numeric_limits<float>::max();

    std::cout << "Two Opt Cost Improvement. Current Cost=" << std::to_string(best_new_cost) << std::endl;

    std::vector<std::vector<size_t>> swap_idxs;

    while (best_new_cost < best_cost) {
        // update best cost to best new cost from last iteration
        best_cost = best_new_cost;

        // while (last_cost == std::numeric_limits<float>::max()) {
        for (size_t idx0 = 1; idx0 < this->path.size() - 1; idx0++) {
        // for (size_t idx0 = 1; idx0 < 4; idx0++) {
            for (size_t idx1 = idx0 + 2; idx1 <= this->path.size(); idx1++) {
            // for (size_t idx1 = 3; idx1 < this->path.size(); idx1++) {
                // save swap idxs
                std::vector<size_t> swap_idx = {idx0, idx1};
                swap_idxs.push_back(swap_idx);
            }
        }

        // shuffle swap idxs
        auto rng = std::default_random_engine {};
        std::shuffle(swap_idxs.begin(), swap_idxs.end(), rng);

        size_t n_swaps = 10;
        for (size_t i = 0; i < swap_idxs.size() - n_swaps; i++) {
            std::vector<std::vector<size_t>> swap_idxs_sub(swap_idxs.begin() + i, swap_idxs.begin() + i + n_swaps);
            this->testSwappedIdxs(swap_idxs_sub, best_new_cost, new_cost);
            // size_t idx0 = swap_idxs[i][0];
            // size_t idx1 = swap_idxs[i][1];

            // // get iterators to perform swap
            // auto it0 = this->path.begin() + idx0;
            // auto it1 = this->path.begin() + idx1;

            // // swap
            // std::reverse(it0, it1);

            // // calculate new cost
            // new_cost = this->pathCost();

            // // if new cost is worse, swap back
            // if (new_cost >= best_new_cost || !this->checkModuleContinuity()) {
            //     std::reverse(it0, it1);
            //     new_cost = best_new_cost; // 'reset' new cost to previous best
            // } else {
            //     std::cout << "New Best: ";
            //     for (size_t i = 0; i < this->path.size(); i++) {
            //         std::cout << this->path[i].vp_idx << " ";
            //     }
            //     std::cout << "| Cost=" << std::to_string(new_cost) << " < " << std::to_string(best_new_cost) << std::endl;
            //     best_new_cost = new_cost;
            // }
        }
        //         // get iterators to perform swap
        //         auto it0 = this->path.begin() + idx0;
        //         auto it1 = this->path.begin() + idx1;

        //         // swap
        //         std::reverse(it0, it1);

        //         // calculate new cost
        //         new_cost = this->pathCost();

        //         // if new cost is worse, swap back
        //         if (new_cost >= best_new_cost || !this->checkModuleContinuity()) {
        //             std::reverse(it0, it1);
        //             new_cost = best_new_cost; // 'reset' new cost to previous best
        //         } else {
        //             std::cout << "New Best: ";
        //             for (size_t i = 0; i < this->path.size(); i++) {
        //                 std::cout << this->path[i].vp_idx << " ";
        //             }
        //             std::cout << "| Cost=" << std::to_string(new_cost) << " < " << std::to_string(best_new_cost) << std::endl;
        //             best_new_cost = new_cost;
        //         }
        //     }
        // }
    }

    best_cost = best_new_cost;

    std::cout << "\nTwo Opt Path: ";
    for (size_t i = 0; i < this->path.size(); i++) {
        std::cout << this->path[i].toString() << " ";
    }
    std::cout << std::endl << "Cost: " << best_cost << std::endl;
}

void TSP::getPath(std::vector<std::vector<float>>& path) {
    // get path as vector of vec3
    // std::vector<std::vector<float>> vps;
    for (size_t i = 1; i < this->path.size(); i++) {
        std::vector<vec3> subpath = this->cm.getPath(this->path[i-1].vp_idx, this->path[i].vp_idx);

        size_t start_idx = 1;
        if (i == 1) { start_idx = 0;}

        for (size_t j = start_idx; j < subpath.size(); j++) {
            std::vector<float> point;
            point.push_back(subpath[j].x);
            point.push_back(subpath[j].y);
            point.push_back(subpath[j].z);
            point.push_back(this->cm.viewpoints[this->path[i].vp_idx].viewdir.x);
            point.push_back(this->cm.viewpoints[this->path[i].vp_idx].viewdir.y);
            point.push_back(this->cm.viewpoints[this->path[i].vp_idx].viewdir.z);
            point.push_back(this->path[i].module_idx);
            path.push_back(point);
        }
        // std::vector<float> vp;
        // vp.push_back(this->cm.viewpoints[this->path[i].vp_idx].pose.x);
        // vp.push_back(this->cm.viewpoints[this->path[i].vp_idx].pose.y);
        // vp.push_back(this->cm.viewpoints[this->path[i].vp_idx].pose.z);
        // vp.push_back(this->cm.viewpoints[this->path[i].vp_idx].viewdir.x);
        // vp.push_back(this->cm.viewpoints[this->path[i].vp_idx].viewdir.y);
        // vp.push_back(this->cm.viewpoints[this->path[i].vp_idx].viewdir.z);
        // vp.push_back(this->path[i].module_idx);
        // vps.push_back(vp);
    }
}

float TSP::insertionCost(size_t idx, size_t insert_idx) {
    // cost to insert idx at insert_idx position in path
    // Before:
    // ( insertion_idx - 1 ) --- ( insertion_idx )
    // After
    // ( insertion_idx - 1 ) --- ( idx ) --- ( insertion_idx )

    // can't insert at first position (should never happen, but protect)
    if (insert_idx == 0) {
        return INFINITY;
    }

    // out of bounds
    if (insert_idx > this->path.size()) {
        return INFINITY;
    }

    // insert at end of path (easy case)
    if (insert_idx == this->path.size()) {
        if (this->path.size() == 1) {
            return this->cm.getSimpleCost(this->path[0].vp_idx, idx); // args should be 0 and idx
        } else {
            return this->cm.getCost(this->path[insert_idx-2].vp_idx, this->path[insert_idx-1].vp_idx, idx);
        }
    }

    // insert somewhere in the middle of the path

    // we will look at the difference between pre and post costs of path subsection
    float before_cost = 0.0f;
    float after_cost = 0.0f;

    // inserting one after the start (affeects middle leg cost)
    if (insert_idx == 1) {
        // there is only one node before insertion point
        // std::cout << "One after start, checking vp idxs: " << this->path[insert_idx-1] << " " << this->path[insert_idx] << " " << idx << std::endl;
        before_cost += this->cm.getSimpleCost(this->path[insert_idx-1].vp_idx, this->path[insert_idx].vp_idx);
        after_cost += this->cm.getSimpleCost(this->path[insert_idx-1].vp_idx, idx) + this->cm.getCost(this->path[insert_idx-1].vp_idx, idx, this->path[insert_idx].vp_idx);
    } else {
        // there are two consecutive nodes before insertion point
        // std::cout << "more than one after start, checking vp idxs: " << this->path[insert_idx-2] << " " << this->path[insert_idx-1] << " " << this->path[insert_idx] << " " << idx << std::endl;
        before_cost += this->cm.getCost(this->path[insert_idx-2].vp_idx, this->path[insert_idx-1].vp_idx, this->path[insert_idx].vp_idx);
        after_cost += this->cm.getCost(this->path[insert_idx-2].vp_idx, this->path[insert_idx-1].vp_idx, idx) + this->cm.getCost(this->path[insert_idx-1].vp_idx, idx, this->path[insert_idx].vp_idx);
    }

    // segment after insertion point ( insertion_idx ) --- ( insertion_idx + 1 )
    if (insert_idx < this->path.size() - 2) {
        // std::cout << "Not inserting right before end: " << this->path[insert_idx-1] << " " << this->path[insert_idx] << " " << this->path[insert_idx+1] << std::endl;
        before_cost += this->cm.getCost(this->path[insert_idx-1].vp_idx, this->path[insert_idx].vp_idx, this->path[insert_idx+1].vp_idx);
        after_cost += this->cm.getCost(idx, this->path[insert_idx].vp_idx, this->path[insert_idx+1].vp_idx);
    }

    return after_cost - before_cost;
}
size_t TSP::nearestNeighbor(TSPWaypoint wpt, float& best_cost) {
    // takes vp_idx and returns the best place to insert it into path (for best_cost cost)
    // initialize best values
    best_cost = std::numeric_limits<float>::max();
    size_t best_idx = -1; // largest ulong (probably out of range)

    bool module_in_path = false; // true if viewpoints with same module membership as wpt are in path
    for (size_t i = 0; i < this->path.size(); i++) {
        if (this->path[i].module_idx == wpt.module_idx) {module_in_path = true;}
    }

    // find best place to insert idx into path lowest cost
    // i starts at 1 because we can't insert before start viewpoint
    for (size_t i = 1; i < this->path.size() + 1; i++) {
        size_t prev_module_idx = this->path[i-1].module_idx;
        size_t next_module_idx = -1;
        if (i < this->path.size()) { next_module_idx = this->path[i].module_idx; } // viewpoint after insert

        if (next_module_idx == wpt.module_idx  // continuous module membership
        || prev_module_idx == wpt.module_idx   // continuous module membership
        || (!module_in_path && prev_module_idx != next_module_idx))  // don't break up a string of continuous module membership (different module that wpt)
        {
            float cost = this->insertionCost(wpt.vp_idx, i);
            if (cost < best_cost) {
                best_cost = cost;
                best_idx = i;
            }
        }
    }
    return best_idx;
}

size_t TSP::nearest(float& best_cost, size_t& node_idx) {
    // returns the best position in path to insert node_idx -- will cost best_cost
    best_cost = std::numeric_limits<float>::max();
    size_t best_idx = -1; // place in this->path to insert viewpoint
    node_idx = -1; // node index in this->nodes to insert into path

    // edge conditions : path size is 1
    // find the best vp idx in nodes to insert into path with lowest cost
    for (size_t i = 0; i < this->nodes.size(); i++) {
        // find best place to insert idx into path lowest cost
        float insertion_cost = 0.0f; // cost after inserting 'nodes[i]' into path at closest/cheapest location
        size_t best_insertion_idx = this->nearestNeighbor(this->nodes[i], insertion_cost);
        // std::cout << "Insertion cost for node " << this->nodes[i].toString() << " at index " << best_insertion_idx << " is " << insertion_cost << std::endl;

        if (insertion_cost < best_cost) {
            best_cost = insertion_cost;
            best_idx = best_insertion_idx;
            node_idx = i;
        }
    }
    return best_idx;
}

float TSP::pathCost() {
    // if no viewpoints or one viewpoint, there is no traversal so cost is 0
    if (this->path.size() == 0 || this->path.size() == 1) {
        return 0.0f;
    }

    if (this->path.size() == 2) {
        return this->cm.getSimpleCost(this->path[0].vp_idx, this->path[1].vp_idx);
    }

    // accumulate cost (get simple cost for first two viewpoints)
    float cost = this->cm.getSimpleCost(this->path[0].vp_idx, this->path[1].vp_idx);

    // set up traversal vars
    size_t prev_idx = this->path[0].vp_idx; // this should always be 0
    size_t this_idx = this->path[1].vp_idx;
    size_t next_idx;
    for (size_t i = 2; i < this->path.size(); i++) {
        next_idx = this->path[i].vp_idx;

        // accumulate cost
        cost += this->cm.getCost(prev_idx, this_idx, next_idx);

        // set up traversal vars for next iteration
        prev_idx = this_idx;
        this_idx = next_idx;
    }
    return cost;
}

float TSP::pathCost(std::vector<TSPWaypoint>& path) {
    // if no viewpoints or one viewpoint, there is no traversal so cost is 0
    if (path.size() == 0 || path.size() == 1) {
        return 0.0f;
    }

    if (path.size() == 2) {
        return this->cm.getSimpleCost(path[0].vp_idx, path[1].vp_idx);
    }

    // accumulate cost (get simple cost for first two viewpoints)
    float cost = this->cm.getSimpleCost(path[0].vp_idx, path[1].vp_idx);

    // set up traversal vars
    size_t prev_idx = path[0].vp_idx; // this should always be 0
    size_t this_idx = path[1].vp_idx;
    size_t next_idx;
    for (size_t i = 2; i < path.size(); i++) {
        next_idx = path[i].vp_idx;

        // accumulate cost
        cost += this->cm.getCost(prev_idx, this_idx, next_idx);

        // set up traversal vars for next iteration
        prev_idx = this_idx;
        this_idx = next_idx;
    }
    return cost;
}

/**
 * @brief get the path cost of a series of ordered viewpoints
 * first match each ordered viewpoint to a TSPWaypoint, then call pathCost(std::vector<TSPWaypoint>& path) on that list
 * @param ordered_vp a std vector of ordered viewpoints represented as std::vector<std::vector<float>>
 */
float TSP::pathCost(std::vector<std::vector<float>> &ordered_vp, bool debug) {
    if (debug) {std::cout << "Calculating path cost for ordered viewpoints..." << std::endl;}
    // first find the TSPWaypoint corresponding to each ordered viewpoint
    std::vector<TSPWaypoint> ordered_path;
    for (size_t i = 0; i < ordered_vp.size(); i++) {
        TSPWaypoint wpt;
        // find closest correspondence in cm.viewpoints
        std::vector<float> l1_distances;
        for (size_t j = 0; j < this->cm.viewpoints.size(); j++) {
            // get squared distance between cm viewpoint and ordered viewpoint
            float l1_dist = 
                std::fabs((this->cm.viewpoints[j].pose.x - ordered_vp[i][0])) +
                std::fabs((this->cm.viewpoints[j].pose.y - ordered_vp[i][1])) +
                std::fabs((this->cm.viewpoints[j].pose.z - ordered_vp[i][2]));

            l1_distances.push_back(l1_dist);
            // if (this->cm.viewpoints[i].pose.x == ordered_vp[i][0] &&
            //     this->cm.viewpoints[i].pose.y == ordered_vp[i][1] &&
            //     this->cm.viewpoints[i].pose.z == ordered_vp[i][2]) {
            //     wpt.vp_idx = static_cast<size_t>(i);
            //     break;
            // }
        }

        // get iterator that points to the element closest to the ordered_vp
        auto min_it = std::min_element(l1_distances.begin(), l1_distances.end());
        size_t min_idx = std::distance(l1_distances.begin(), min_it);
        wpt.vp_idx = min_idx;
        wpt.module_idx = static_cast<size_t>(ordered_vp[i][7]);

        // only add if within margin of error
        if (l1_distances[min_idx] < 0.000003f) {
            ordered_path.push_back(wpt);
        }
        else if (debug)
        {
            std::cout << "Failed to find corresponding viewpoint for ordered vp: "
                      << ordered_vp[i][0] << ", "
                      << ordered_vp[i][1] << ", "
                      << ordered_vp[i][2] << std::endl;
            std::cout << "Closest match was viewpoint index: " << min_idx 
                      << " with distance: " << l1_distances[min_idx] << std::endl;
        }
    }
    return this->pathCost(ordered_path);
}