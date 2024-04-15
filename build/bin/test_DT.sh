#!/bin/sh
#cd ..
#cmake --build . --target install --parallel
#cd bin/
./skyline-OT-DT r=1  p=10000 & ./skyline-OT-DT r=2 p=10000
