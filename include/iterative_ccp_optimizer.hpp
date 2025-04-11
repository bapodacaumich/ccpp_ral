#ifndef ITERATIVE_CCP_OPTIMIZER_HPP
#define ITERATIVE_CCP_OPTIMIZER_HPP

#include "model.hpp"
#include "viewpoint_struct.hpp"

// iterative ccp optimizer class
class ICCPO {
    public:
        // Constructor
        ICCPO(std::vector<Viewpoint>& viewpoints); // assume default model (scale 4, non vx)
        ICCPO(Model& model, std::vector<Viewpoint>& viewpoints);

        // get current solution
        void getCurrentSolution(std::vector<Viewpoint>& viewpoints_out);
        void getCurrentOrder(std::vector<size_t>& viewpoint_order_out);
        void getCurrentCost(float& cost_out);
        void getCurrentCoverageQuality(std::vector<float>& coverage_quality_out);
    private:
        // contains triangles, normals, vertices
        Model model;

        // coverage quality map size(n_triangles)
        std::vector<float> current_coverage_quality;

        // all viewpoints
        std::vector<Viewpoint> viewpoints;

        // order of viewpoints
        std::vector<size_t> viewpoint_order;
};

#endif // ITERATIVE_CCP_OPTIMIZER_HPP