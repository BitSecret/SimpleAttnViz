# A Simple Method for Attention Visualization

The official implementation of the paper
["A Simple Method for Attention Visualization"](https://arxiv.org/abs/2310.18021).

This paper formally defines the concepts of intra-token information mixing and inter-token information mixing, while
proposing a visualization analysis method based on attention matrix forward propagation. The proposed method achieves
interpretable analysis of Transformer model decision-making processes by quantitatively measuring the contribution of
different input tokens to model outputs. To validate the method's effectiveness, we constructed an image classification
model based on the ViT and conducted experiments on the CIFAR-10 dataset. The proposed method generated attention
heatmaps that visualize the model's focus regions during the classification process.

## Installation

This project uses [pyproject.toml](https://packaging.python.org/en/latest/specifications/declaring-project-metadata) to
store project metadata. The command `pip install -e .` reads file `pyproject.toml`, automatically installs project
dependencies, and installs the current project in an editable mode into the environment's library. It is convenient for
project development and testing.

    $ git clone --depth 1 https://github.com/BitSecret/SimpleAttnViz.git
    $ cd FormalGeo
    $ conda create -n <your_env_name> python=3.10
    $ conda activate <your_env_name>
    $ pip install -e .

## Running

Create directory:

    |--data
    |  |--checkpoints
    |  |--outputs
    |  |--CIFAR-10
    |  |--config.json
    |
    |--src
    |  |--sav
    |     |--data.py
    |     |--model.py
    |     |--main.py
    |     |--utils.py
    |
    |--pyproject.toml
    |
    |--README.md

Download datasets:

    $ python data.py

Training and visualization:

    $ python main.py


## Citation

To cite FormalGeo in publications use:
> Zhang, X. (2025). A Simple Method for Attention Visualization.

A BibTeX entry for LaTeX users is:
> @misc{arxiv2025simple,  
> title={A Simple Method for Attention Visualization},  
> author={Xiaokai Zhang},  
> year={2025},  
> eprint={2310.18021},  
> archivePrefix={arXiv},  
> primaryClass={cs.AI},  
> url={https://arxiv.org/abs/2310.18021}  
> }