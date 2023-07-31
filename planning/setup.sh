#!/bin/bash

# setup the virtual environment
python3 -m venv venv
pip3 install gymnasium
pip3 install gymnasium[all] #replace with whatever the smallest reasonable thing you use in the tutorial, all is gigs
pip3 install pathfinding

# grab the pathfinding datasets
mkdir data
pushd data
mkdir maps
wget https://movingai.com/benchmarks/bgmaps/bgmaps-map.zip
unzip bgmaps-map.zip -d maps
mkdir scenarios
wget https://movingai.com/benchmarks/bgmaps/bgmaps-scen.zip
unzip bgmaps-scen.zip -d scenarios

