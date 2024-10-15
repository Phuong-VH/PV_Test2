#!/bin/bash
python3 -m pip install --upgrade pip
pip3 install catboost
# pip3 install Pillow image mealpy;
pip3 install --upgrade tensorflow
pip3 install --upgrade numpy pandas sklearn pillow;
python3 ./PV_Test/CNN_PSO.py;
