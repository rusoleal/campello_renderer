#!/bin/bash
cmake -B build -DCAMPELLO_RENDERER_BUILD_TEST=ON
cmake --build build
cd build && ctest --output-on-failure
