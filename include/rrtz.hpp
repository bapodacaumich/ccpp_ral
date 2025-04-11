#ifndef RRTZ_HPP
#define RRTZ_HPP

#include "limit_struct.hpp"
#include "node3d_struct.hpp"
#include "obs.hpp"
#include "plane_struct.hpp"
#include "triangle_struct.hpp"
#include "utils.hpp"
#include "vec3_struct.hpp"

#include <vector>

class RRTZ {
    public:
        // public members
        RRTZ();

        RRTZ(
            vec3 start,
            vec3 goal,
            std::vector<OBS> obs,
            Limit limits,
            size_t max_nodes,
            bool debug=false
        );

        bool run(std::vector<vec3>& path);
        float getBestCost();

    private:
        // private members
        vec3 start;
        vec3 goal;
        size_t max_nodes;
        Limit limits;
        std::vector<Node3D> tree_nodes;
        std::vector<OBS> obs;
        std::vector<float> costs;
        std::vector<Triangle*> triangles;

        // debug
        bool debug;

        // best solution
        float best_cost;
        size_t best_idx;

        bool terminate();
        void reconstruct_path(size_t parent_idx, std::vector<vec3>& path);
        Plane sample_plane();
        Plane sample_plane(vec3 point);
        bool extend(Plane p, float& cost);
        void print_node(Node3D node);
        bool get_near_nodes(Plane plane, std::vector<Node3D*>& near_nodes, std::vector<vec3>& int_points);
        void add_node_to_tree(Node3D& node, float& cost);
        bool check_bounds(vec3 point);
        // bool collisionfn(Node3D node);
        bool collision(vec3 origin, vec3 end);
};

#endif // RRTZ_HPP