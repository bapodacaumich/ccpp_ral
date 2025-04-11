#include "utils.hpp"
#include "obs.hpp"

int main() {
    std::vector<OBS> convexObsVec;
    loadConvexStationOBS(convexObsVec, 4);

    std::vector<OBS> obsVec;
    loadStationOBS(obsVec, 4);

    float min_x = 0;
    float min_y = 0;
    float min_z = 0;
    float max_x = 0;
    float max_y = 0;
    float max_z = 0;
    for (size_t i = 0; i < convexObsVec.size(); i++) {
        OBS obs = convexObsVec[i];
        for (size_t j = 0; j < obs.faces.size(); j++) {
            Triangle tri = obs.faces[j];
            std::vector<vec3> vertices = {tri.a, tri.b, tri.c};
            for (size_t k = 0; k < 3; k++) {
                vec3 vertex = vertices[k];
                if (vertex.x < min_x) {
                    min_x = vertex.x;
                }
                if (vertex.y < min_y) {
                    min_y = vertex.y;
                }
                if (vertex.z < min_z) {
                    min_z = vertex.z;
                }
                if (vertex.x > max_x) {
                    max_x = vertex.x;
                }
                if (vertex.y > max_y) {
                    max_y = vertex.y;
                }
                if (vertex.z > max_z) {
                    max_z = vertex.z;
                }
            }
        }
    }

    std::cout << "Convex Obs Vec" << std::endl;
    std::cout << "xrange: " << min_x << ", " << max_x << std::endl;
    std::cout << "yrange: " << min_y << ", " << max_y << std::endl;
    std::cout << "zrange: " << min_z << ", " << max_z << std::endl;

    min_x = 0;
    min_y = 0;
    min_z = 0;
    max_x = 0;
    max_y = 0;
    max_z = 0;
    for (size_t i = 0; i < obsVec.size(); i++) {
        OBS obs = obsVec[i];
        for (size_t j = 0; j < obs.faces.size(); j++) {
            Triangle tri = obs.faces[j];
            std::vector<vec3> vertices = {tri.a, tri.b, tri.c};
            for (size_t k = 0; k < 3; k++) {
                vec3 vertex = vertices[k];
                if (vertex.x < min_x) {
                    min_x = vertex.x;
                }
                if (vertex.y < min_y) {
                    min_y = vertex.y;
                }
                if (vertex.z < min_z) {
                    min_z = vertex.z;
                }
                if (vertex.x > max_x) {
                    max_x = vertex.x;
                }
                if (vertex.y > max_y) {
                    max_y = vertex.y;
                }
                if (vertex.z > max_z) {
                    max_z = vertex.z;
                }
            }
        }
    }
    std::cout << "\nObs Vec" << std::endl;
    std::cout << "xrange: " << min_x << ", " << max_x << std::endl;
    std::cout << "yrange: " << min_y << ", " << max_y << std::endl;
    std::cout << "zrange: " << min_z << ", " << max_z << std::endl;
    return 0;
}