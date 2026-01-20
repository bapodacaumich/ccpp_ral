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
#ifndef TSP_HPP
#define TSP_HPP

#include "cost_matrix.hpp"
#include "tsp_waypoint_struct.hpp"
#include "viewpoint_struct.hpp"

#include <vector>

class TSP {
    public:

        TSP();
        TSP(CostMatrix cm);
        void loadCM(int vgd, Viewpoint start, bool vx);
        void greedyInit();
        void twoOpt();
        float pathCost();
        float pathCost(std::vector<TSPWaypoint>& path);
        float pathCost(std::vector<std::vector<float>>& path, bool debug=false); // get path cost by matching ordered viewpoints with existing ones
        void getPath(std::vector<std::vector<float>>& path); // return path with view directions
        void reassignModuleMembership();
        void globalOpt(); // reassign module membership for global constraint

    private:

        size_t n_vp;
        CostMatrix cm;
        // std::vector<size_t> path;
        // std::vector<size_t> nodes;
        std::vector<TSPWaypoint> path;
        std::vector<TSPWaypoint> nodes;

        float insertionCost(size_t idx, size_t insert_idx);
        size_t nearestNeighbor(TSPWaypoint wpt, float& best_cost);
        size_t nearest(float& best_cost, size_t& node_idx);

        // two opt helper
        void testSwappedIdxs(std::vector<std::vector<size_t>> swap_idxs, float& best_new_cost, float& new_cost);

        // check path module continuity for this->path
        bool checkModuleContinuity();
};

#endif // TSP_HPP
