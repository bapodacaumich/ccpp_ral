#ifndef COST_MATRIX_HPP
#define COST_MATRIX_HPP

#include "vec3_struct.hpp"
#include "viewpoint_struct.hpp"

#include <vector>

class CostMatrix {
    public:
        CostMatrix();
        CostMatrix(size_t rrtz_iterations);
        void generateCostMatrix();
        void generatePathMatrix();
        void generatePathMatrixParallel();
        void loadViewpoints(std::string filename, Viewpoint start, bool debug=false);
        void loadSimpleCostMatrix(std::string filename);
        void saveSimpleCostMatrix(std::string filename);
        void loadCostMatrix(std::string filename);
        void saveCostMatrix(std::string filename);
        void loadPathMatrix(std::string filename);
        void savePathMatrix(std::string filename);
        float getCost(size_t i, size_t j, size_t k);
        float getSimpleCost(size_t i, size_t j);
        std::vector<vec3> getPath(size_t i, size_t j);
        size_t getNVP() { return this->n_vp; }

        size_t rrtz_iterations; // number of iterations to run rrtz
        size_t n_vp; // number of viewpoints

        // coverage viewpoints
        std::vector<Viewpoint> viewpoints;
        std::vector<vec3> viewpoint_dirs;

        // (n, n, n) matrix of costs from viewpoint j to k given i->j->k.
        std::vector<std::vector<std::vector<float>>> cost_matrix; 

        // (n, n) matrix of costs from viewpoint i to j.
        std::vector<std::vector<float>> simple_cost_matrix;

        // (n, n, pathlength) matrix (symmetric) paths from viewpoint i to j.
        std::vector<std::vector<std::vector<vec3>>> path_matrix;
    private:
    size_t N_discretization = 10; // subpath discretization for cost calculations
    float speed = 0.2f; // speed of the agent for CW computations
    float totalCost(std::vector<vec3> path); // compute total cost of a path
    float CWCost(vec3 start, vec3 end, size_t n); // compute cost to oppose CW disturbance from start to end at speed.
};

#endif // COST_MATRIX_HPP