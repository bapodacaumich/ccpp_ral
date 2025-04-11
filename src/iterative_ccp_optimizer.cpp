#include "iterative_ccp_optimizer.hpp"


/**
 * @brief Construct a new ICCPO::ICCPO object
 * @param viewpoints vector of viewpoints to query for paths
 * @return ICCPO object
 */
ICCPO::ICCPO(std::vector<Viewpoint>& viewpoints) {
    Model model = Model(false, 4.0f);
    this->model = model;
    this->viewpoints = viewpoints;
}

/**
 * @brief Construct a new ICCPO::ICCPO object
 * @param viewpoints vector of viewpoints to query for paths
 * @return ICCPO object
 */
ICCPO::ICCPO(Model &model, std::vector<Viewpoint>& viewpoints) {
    this->model = model;
    this->viewpoints = viewpoints;
}

/**
 * @brief Get the current solution
 * @param viewpoints_out vector of viewpoints to output
 * @return void
 */
void ICCPO::getCurrentSolution(std::vector<Viewpoint>& viewpoints_out) {
    viewpoints_out.clear();

    for (size_t i = 0; i < this->viewpoint_order.size(); i++) {
        viewpoints_out.push_back(this->viewpoints[this->viewpoint_order[i]]);
    }
}

/**
 * @brief Get the current solution
 * @param viewpoint_order_out vector of viewpoint idxs of current solution
 * @return void
 */
void ICCPO::getCurrentOrder(std::vector<size_t>& viewpoint_order_out) {
    viewpoint_order_out = this->viewpoint_order;
}
