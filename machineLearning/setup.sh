#!/bin/bash
# Script for setting up the python environment for this tutorial
python3 -m venv venv
source ./venv/bin/activate
pip3 install scikit-learn
pip3 install matplotlib
pip3 install numpy
pip3 install torch
pip3 install torchvision