#ifndef LIMIT_STRUCT_HPP
#define LIMIT_STRUCT_HPP

struct Limit {
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    Limit() {
        xmin = 0;
        xmax = 0;
        ymin = 0;
        ymax = 0;
        zmin = 0;
        zmax = 0;
    }
    Limit(float xmi, float xma, float ymi, float yma, float zmi, float zma) {
        xmin = xmi;
        xmax = xma;
        ymin = ymi;
        ymax = yma;
        zmin = zmi;
        zmax = zma;
    }
    void set(float xmi, float xma, float ymi, float yma, float zmi, float zma) {
        xmin = xmi;
        xmax = xma;
        ymin = ymi;
        ymax = yma;
        zmin = zmi;
        zmax = zma;
    }
};

#endif // LIMIT_STRUCT_HPP