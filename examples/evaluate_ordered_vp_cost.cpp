#include "cost_matrix.hpp"
#include "tsp.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"

#include <string>
#include <vector>

/**
 * Evaluate the ordered viewpoint fuel cost (with drift rejection) at a given velocity
 */

int main() {
    std::vector<float> vgds = {
        2.0f,
        4.0f,
        8.0f,
        16.0f
    };

    std::vector<bool> localities = {
        false,
        true
    };

    std::string vx_str = ""; // empty for non vx station

    std::string dir = "../data/ordered_viewpoints/";

    // // compute the cost of the ordered viewpoints using TSP module
    // for (auto vgd : vgds) {
    //     for (auto local : localities) {
    //         // load in cost matrix corresponding to vgd and locality
    //         Viewpoint start = Viewpoint( vec3(-5.0f, -2.0f, 3.0f), vec3(1.0f, 0.0f, 0.0f), 2);
    //         size_t rrtz_iter = 2000;
    //         CostMatrix cm(rrtz_iter);
    //         std::string viewpoint_file;

    //         std::string locality_str = local ? "local" : "global";
    //         cm.loadViewpoints(
    //             "../data" + vx_str + "/coverage_viewpoint_sets/coverage_" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_vp_set.csv",
    //             start
    //         );

    //         cm.loadPathMatrix("../data/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_path_matrix.csv");
    //         cm.loadSimpleCostMatrix("../data/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_simple_cost_matrix.csv");
    //         cm.loadCostMatrix("../data/tsp/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + "_cost_matrix.csv");

    //         // get TSP object to evaluate cost of ordered viewpoints
    //         TSP tsp(cm);

    //         // load in ordered viewpoints and make sure each one is accounted for -- if one isn't, check that it is the path between the surrounding ones
    //         std::vector<std::vector<float>> path;
    //         std::string file = "../data" + vx_str + "/ordered_viewpoints/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + ".csv";
    //         loadCSV(
    //             file, 
    //             path, 
    //             7 // pose (3), viewdir (3), module membership (1)
    //         );

    //         // compute cost of ordered viewpoint traversal using the cost matrix
    //         float cost = tsp.pathCost(path);

    //         std::cout << std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + " cost: " << cost << std::endl;
    //     }
    // }

    float v_nom = 0.1f;

    /// compute the cost of the ordered viewpoints using the ordered viewpoint trajectory directly
    std::cout << "vgd, locality, cost (g)" << std::endl;
    for (auto vgd : vgds) {
        for (auto local : localities) {
            std::string locality_str = local ? "local" : "global";
            std::vector<std::vector<float>> path;
            std::string file = "../data" + vx_str + "/ordered_viewpoints/" + std::to_string(static_cast<int>(vgd)) + "m_" + locality_str + ".csv";
            loadCSV(
                file, 
                path, 
                7 // pose (3), viewdir (3), module membership (1)
            );

            float cost = 0.0f;
            vec3 last_v = vec3(0.0f, 0.0f, 0.0f);
            for (size_t i = 0; i < path.size(); i++) {
                vec3 next_v;

                // if on last iteration, final velocity is zero
                if (i == path.size()-1) {
                    next_v = vec3(0.0f, 0.0f, 0.0f);
                } 
                else  // otherwise, use the consecutive viewpoints
                {
                    next_v = vec3(
                        path[i+1][0] - path[i][0],
                        path[i+1][1] - path[i][1], 
                        path[i+1][2] - path[i][2]
                    );
                    next_v.normalize();
                    next_v = next_v * v_nom; // set velocity to v_nom
                }

                // std::cout << "last_v: " << last_v.toString() << " next_v: " << next_v.toString() << std::endl;

                vec3 this_pose = vec3(
                    path[i][0],
                    path[i][1],
                    path[i][2]
                );

                float impulsive_fuel_cost = fuel_cost(
                    this_pose,
                    last_v,
                    next_v,
                    v_nom,
                    0.01f // time step
                );

                // std::cout << "impulsive_fuel_cost: " << impulsive_fuel_cost << std::endl;
                cost += impulsive_fuel_cost;

                // only compute drift opposition between two path points
                if (i < path.size()-1) {
                    vec3 next_pose = vec3(
                        path[i+1][0],
                        path[i+1][1],
                        path[i+1][2]
                    );

                    float drift_rejection_cost = cw_cost(
                        this_pose,
                        next_pose,
                        v_nom,
                        100 // number of discretization steps
                    );

                    // std::cout << "drift_rejection_cost: " << drift_rejection_cost << std::endl;
                    cost += drift_rejection_cost;
                }
                last_v = next_v;
            }

            std::cout << std::to_string(static_cast<int>(vgd)) + "m, " + locality_str + ", " << cost*1000 << std::endl;
        }
    }
}