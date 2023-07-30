#!/bin/bash
# Script for setting up the python environment for this tutorial
python3 -m venv venv
source ./venv/bin/activate
pip3 install vaderSentiment
mkdir data
pushd data
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz
popd
