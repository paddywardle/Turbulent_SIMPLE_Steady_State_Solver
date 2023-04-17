#!/bin/bash

python3 -OO ./src/main.py --SIM_num $1
cp config/config.json Results/SIM_$1/config.json
