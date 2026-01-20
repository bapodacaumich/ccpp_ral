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
#ifndef MODEL_HPP
#define MODEL_HPP

#include "obs.hpp"
#include "viewpoint_struct.hpp"

#include <vector>

class Model {
    public:
        // Constructor
        Model();
        Model(bool vx, float scale);

        // ------------------------------------------------------------
        // Binary coverage is given in a mask matrix or integer value
        // ------------------------------------------------------------

        // get coverage mask for viewpoint pose
        void getBinaryCoverage(std::vector<bool>& coverage_out, Viewpoint& pose);

        // get summed coverage
        void getBinaryCoverage(size_t& coverage_out, Viewpoint& pose);

        // get marginal coverage (new coverage) given a coverage prior
        void getMarginalBinaryCoverage(std::vector<bool>& coverage_out, Viewpoint& pose, std::vector<bool>& coverage_prior);

        // get summed marginal coverage (new coverage) given a coverage prior
        void getMarginalBinaryCoverage(size_t& coverage_out, Viewpoint& pose, std::vector<bool>& coverage_prior);

        // ------------------------------------------------------------
        // Coverage quality (float) is the area of visible triangles
        // projected onto the camera plane
        // ------------------------------------------------------------

        // get coverage quality for a single viewpoint per triangle
        void getCoverageQuality(std::vector<float>& coverage_quality, Viewpoint& pose);

        // get summed coverage quality for a single viewpoint
        void getCoverageQuality(float& coverage_quality, Viewpoint& pose);

        // get marginal coverage quality per triangle for a single viewpoint given coverage quality prior
        void getMarginalCoverageQuality(
            std::vector<float>& marginal_coverage_quality, 
            Viewpoint& pose, 
            std::vector<float>& coverage_prior
        );

        // get summed marginal coverage quality per triangle for a single viewpoint given coverage quality prior
        void getMarginalCoverageQuality(
            float& marginal_coverage_quality, 
            Viewpoint& pose, 
            std::vector<float>& coverage_prior
        );

    private:
        // contains triangles, normals, vertices
        std::vector<Triangle*> all_faces;
        std::vector<vec3> normals;
        std::vector<OBS> obsVec;

        void loadOBSVec(std::vector<OBS> obsVec);
        void loadTriangles();

        float coverageQuality(vec3 pose, size_t tri_idx, float inc_angle);
};

#endif // MODEL_HPP
