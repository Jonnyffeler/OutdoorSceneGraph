#!/bin/bash
PROJECT_PATH=""
SCENE="LIN"
CONDA_BIN=""

source $CONDA_BIN/activate osg

cd $PROJECT_PATH
cd src

python build_objects.py --path $PROJECT_PATH --scene $SCENE #--show_runtimes
python build_descriptions.py --path $PROJECT_PATH --scene $SCENE
python build_edges.py --path $PROJECT_PATH --scene $SCENE