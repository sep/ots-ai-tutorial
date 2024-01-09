#!/bin/bash
# Commands used to configure the VM to distribute to students Should
# also allow students to configure their own machines automatically
# and support building a remote box for them to connect to on digital
# ocean or similar

# Grab some standard development tools
sudo apt-get install emacs git screen python3 python3-pip python3-venv
# Tools for the ML tutorials
sudo apt-get install python3-matplotlib python3-scikit-learn python3-torch python3-torchvision
# Tools for the NLP tutorials


