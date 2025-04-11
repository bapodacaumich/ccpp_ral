#ifndef TRIANGLE_COVERAGE_STRUCT_HPP
#define TRIANGLE_COVERAGE_STRUCT_HPP

#include <stddef.h>

struct TriangleCoverage {
    bool covered;
    bool covered_any;
    float best_inc_angle;
    size_t module_idx;
};

#endif // TRIANGLE_COVERAGE_STRUCT_HPP