#!/bin/bash
PROJECT_PATH=""
SCENES=("LIN" "CAB" "HGE")
SCENE="LIN"
CONDA_BIN=""

source $CONDA_BIN/activate osg

cd $PROJECT_PATH
cd sgaligner_proprocessing

python create_bow_attrs.py --path $PROJECT_PATH --scenes "${SCENES[@]}"
python create_subscenes.py --path $PROJECT_PATH --split train --scene $SCENE --roots_only
python create_subscenes.py --path $PROJECT_PATH --split val --scene $SCENE --roots_only
python create_subscenes.py --path $PROJECT_PATH --split test --scene $SCENE --roots_only