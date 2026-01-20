/* space station inspection planning using optimal control problem
Copyright (C) 2026 Brandon Apodaca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
    */
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
