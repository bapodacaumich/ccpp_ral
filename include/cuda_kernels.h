#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "vec3_struct.hpp"
#include "viewpoint_struct.hpp"
#include "triangle_struct.hpp"

#include <vector>

// __device__ void pinhole_camera(
//     bool &visible,
//     vec3 pose,
//     vec3 viewdir,
//     vec3 point,
//     float hfov, // rad
//     float vfov // rad
// );


// given a viewpoint: determine visible (can see all three vertices) triangles -- true means not visible (collision during raycast)
// populate int_points with intersection points between rays cast to each vertex of triangle in question and other triangles
extern "C" void cuda_kernel_coverage(
    const Viewpoint& viewpoint,
    const std::vector<Triangle*>& triangles,
    std::vector<bool>& collisions // empty vector to be filled with true/false for each triangle
    // vec3** int_points
);

extern "C" void cuda_kernel_ray_int_plane(
    const std::vector<vec3>& ray_starts,
    const std::vector<vec3>& ray_ends,
    const vec3& plane_point,
    const vec3& plane_normal,
    bool* collisions,
    vec3* int_points
);

// given a list of rays, determine if each ray is in collision with the list of triangles
extern "C" void cuda_kernel_many_ray(
    const std::vector<vec3>& start_ray,
    const std::vector<vec3>& end_ray,
    const std::vector<Triangle*>& triangles,
    bool* collisions
    // vec3** int_points // intersection points if not nullptr
);

// given a list of viewpoints mapped to triangles, determine if each viewpoint can see its respective triangle
extern "C" void cuda_kernel_many(
    const std::vector<Viewpoint>& viewpoints,
    const std::vector<size_t>& triangle_indices,
    const std::vector<Triangle*>& triangles,
    bool* collisions,
    vec3** int_points
);

// given a list of start points, single endpoint, list of faces, determine if each startpoint-endpoint combination is in collision with the faces
extern "C" void cuda_kernel_collision_points_vec3(
    const std::vector<vec3>& triangle_points,
    const std::vector<Triangle*>& faces,
    const vec3 free_space_point,
    std::vector<bool>& in_collision // number of collisions
);

// given a list of viewpoints and triangles and a point in the free space, determine if each viewpoint is in collision
// * even number of intersections between viewpoint and free space point means no collision
extern "C" void cuda_kernel_collision_points(
    const std::vector<Viewpoint>& viewpoints,
    const std::vector<Triangle*>& faces,
    const vec3 free_space_point,
    std::vector<bool>& in_collision // populate with true/false for each viewpoint
);

extern "C" void cuda_kernel_inc_angle(
    const std::vector<Viewpoint>& viewpoints,
    const std::vector<Triangle*>& faces,
    std::vector<std::vector<float>>& inc_angles // populate with incidence angles for each triangle
);

#endif // CUDA_KERNELS_H