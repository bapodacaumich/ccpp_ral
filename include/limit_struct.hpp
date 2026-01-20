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
