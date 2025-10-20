#!/bin/bash

# setup the virtual environment
python3 -m venv venv
source ./venv/bin/activate
pip3 install gymnasium
pip3 install gymnasium[classic_control] tqdm
pip3 install pathfinding
pip3 install matplotlib

# grab the pathfinding datasets
mkdir data
pushd data
mkdir maps
wget https://movingai.com/benchmarks/bgmaps/bgmaps-map.zip
unzip bgmaps-map.zip -d maps
mkdir scenarios
wget https://movingai.com/benchmarks/bgmaps/bgmaps-scen.zip
unzip bgmaps-scen.zip -d scenarios

