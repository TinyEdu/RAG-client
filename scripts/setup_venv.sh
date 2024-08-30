#!/bin/bash

sudo apt install python3.10
python3 -m venv ai
source ai/bin/activate
pip3 install -r config/requirements.txt