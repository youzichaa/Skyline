#!/bin/sh
#cd ..
#cmake --build . --target install --parallel
#cd bin/
./skyline-OT-D r=1  p=9000 & ./skyline-OT-D r=2 p=9000
