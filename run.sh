#!/bin/sh
# Configures, builds and runs the repro. Extra arguments are passed to the program.
set -e
cd "$(dirname "$0")"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
exec build/dmat4_repro "$@"
