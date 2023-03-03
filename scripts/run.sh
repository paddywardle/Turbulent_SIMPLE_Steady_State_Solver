#!/bin/bash

python -OO ./src/main.py --SIM_num $1
cp config/config.json Results/"SIM $1"/config.json