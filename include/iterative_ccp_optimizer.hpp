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
