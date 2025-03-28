# Hierarchical 3D Scene Graphs Construction Outdoors

## Overview 🌐

This project introduces an advanced pipeline for constructing hierarchical 3D scene graphs from outdoor environments. The research addresses critical challenges in spatial understanding for applications such as:

- Robotics
- Urban Planning
- Autonomous Navigation

### Key Contributions

- **Comprehensive Scene Representation**: Extracts and organizes objects from entire buildings down to individual windows
- **Robust Hierarchical Modeling**: Leverages geometric and semantic relationships
- **Scalable Approach**: Efficiently handles large outdoor environments
- **Downstream Application**: Demonstrates utility in 3D scene alignment tasks

## Code Structure 📂

```
├── outdoor_scenegraph
│   ├── Lamar                         <- LaMAR scenes                 
│   │   │── CAB                       
│   │   │── HGE                       
│   │   │── LIN                       
│   ├── out                           <- save folder for created files (scene graphs)
│   ├── SAM_checkpoint                <- checkpoint for the SAM model
|   |── sgaligner                     <- SGAligner code
│   │── sgaligner_preprocessing       <- functions to prepare scene graphs for SGAligner
│   ├── src
│   │   │── build_descriptions.py     <- function that builds the descriptions of the objects        
│   │   │── build_edges.py            <- function that builds the edges in the scene graph
│   │   │── build_objects.py          <- function that builds the object instances in the scene graph
|   |── utils                         <- util functions
|   |── build_scenegraph.sh           <- script to build the full scene graph for a scene
|   |── create_all_subscenes.sh       <- script to create the subscenes for all splits for a scene
│   │── README.md                    
│
```


## Dependencies 💻
The main dependencies of the project are the following:
```yaml
python: 3.9.0
cuda: 11.6
```

You can set up a conda environment by cloning this repository and using the provided yml file.

If you're experiencing installation issues, try installing all non-Torch packages first, then install the Torch-related packages separately using:

```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-geometric==2.4.0
pip install torch-cluster==1.6.0+pt112cu116 torch-scatter==2.1.0+pt112cu116 torch-sparse==0.6.15+pt112cu116 torch-spline-conv==1.2.1+pt112cu116 -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-tb-profiler==0.4.1 torchsummary==1.5.1
```


## Scene Graph Generation :hammer:
After installing the dependencies, download the [LaMAR dataset](https://lamar.ethz.ch/) and move the files in the corresponding folders.

Change the ``PROJECT_PATH `` in ``build_scenegraph.sh`` and select what scene you want to run by chaninging ``SCENE``. Then run: 

```bash
bash build_scenegraph.sh
```


## Preprocessing for SGAligner :hammer:
To train and run SGAligner on the generated outdoor scenes, first run the preprocessing to generate the subscenes used for training.

Change the ``PROJECT_PATH `` in ``create_all_subscenes.sh`` and select the scene (to train or test on) by channging ``SCENE`` (and ``SCENES`` if you didn't construct all three scenes). Then run: 

```bash
bash create_all_subscenes.sh
```


## SGAligner for LaMAR :vertical_traffic_light:
To evaluate the generated 3D scene graphs on the LaMAR dataset, SGAligner, a method to align 3D scene graphs, is adapted to the LaMAR data in the sgaligner_lamar folder. The main changes are the addition of a HGN network, as well as the new data class lamar. 

After chaning the ``PROJECT_PATH`` and choosing the scenes, please run

```bash
bash create_all_subscenes.sh
```

To create the subscenes used for training and testing in the SGAligner adaptation. Config files for training and testing different model variants are provided.


## Acknowledgments :recycle:
In this project we use (parts of) the official implementations of the following works and thank the respective authors for open sourcing their methods: 

- [LaMAR](https://github.com/microsoft/lamar-benchmark) (LaMAR Dataset)
- [SGAligner](https://github.com/sayands/sgaligner) (Scene Graph Alignment Method)
