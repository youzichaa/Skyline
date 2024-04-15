#!/bin/bash
for i in {0..2}
do
    ./skyline-OT-QT r=1 p=9000 Pname=$i Psize=1 & ./skyline-OT-QT r=2 p=9000 Pname=$i Psize=1
    sleep 1
done
