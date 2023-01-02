<h1 align="center">SUREL+: Moving from Walks to Sets for Scalable Subgraph-based Graph Representation Learning</h1>

SUREL+ is a novel set-based computation framework for scalable SGRL.
It avoids the costly procedure of query-specific subgraph extraction while using sampled node sets whose join can be used as proxies of the subgraphs to make prediction. 
SUREL+ benefits from the reusability of these node sets across different queries, which substantially reduces memory/time consumption by avoiding having heavy node duplication in walk-based representaion of SUREL.
SUREL+ designs dedicated sparse storage **SpG** and sparse join operation **SpJoin** to handle the irregularity of node sets. 
SUREL+ also adopts a modular design, where users may flexibly choose among different set samplers, structure encoders and set neural encoders to compensate the loss of structural information caused by set-based representations for their own SGRL tasks.

Currently, we support:
- Large-scale graph learning tasks: link prediction / relation prediction / higher-order pattern prediction
- Preprocessing and training on seven datasets in OGB format
- Flexible modules:
  - Set Samplers: walk-based, metric-based
  - Structure Encoders: Landing Probabilities, Shortest Path Distance, PPR scores
  - Neural Set Encoders: MLP+meanp pooling, LSTM-based, Attention-based
- Single GPU training and evaluation
- Structural Features + Node Features

## Requirements ##
(Other versions may work, but are untested)
* Ubuntu 20.04
* CUDA >= 11.3
* python >= 3.8
* pytorch >= 1.11.0
* PyG >= 2.1.0
* gcc >= 8.4
* cmake >= 3.16
* make >= 4.2
* sureal_gacc = 1.0

## SGRL Environment Setup ##

Requirements: Python >= 3.8, [Anaconda3](https://www.anaconda.com/)

- Update conda:
```bash
conda update -n base -c defaults conda
```

- Install basic dependencies to virtual environment and activate it: 
```bash
conda env create -f environment.yml
conda activate sgrl-env
```

- Example commends of installation for PyTorch (>= 1.11.0) and torch-geometric (>=2.1.0) with CUDA 11.3:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
```
For more details, please refer to the [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). The code of this repository is lately tested with Python 3.8.13 + PyTorch 1.11.0 + torch-geometric 2.1.0.

## Quick Start

1. Install required version of PyTorch that is compatible with your CUDA driver

2. Clone the repository `git clone https://github.com/Graph-COM/SUREL_Plus.git`

3. Build and install SUREL_GAcc `cd surel_gacc;python3 setup.py install`

- To train **SUREL+** for link prediction on Collab:
```bash
python main.py --dataset ogbl-collab --metric Hits --sencoder LP --num_steps 3 --num_walks 200 --aggr attn --use_val
```

- To train **SUREL+** for link prediction on Citation2:
```bash
python main.py --dataset ogbl-citation2 --metric MRR --sencoder ppr --topk 100  --aggr mean
```

- To train **SUREL+** for relation type prediction on MAG(A-P):
```bash
python main.py --dataset mag --relation write --metric MRR --sencoder LP --num_step 3 --num_walk 100 --k 10 
```

- To train **SUREL+** for higher-order pattern prediction on DBLP:
```bash
python main_horder.py --dataset DBLP-coauthor --metric mrr --num_step 3 --num_walk 100 
```

- All detailed training logs can be found at `<log_dir>/<dataset>/<training-time>.log`.


## Usage
```
usage: main.py [-h] [--device DEVICE] [--log_steps LOG_STEPS]
               [--num_layers NUM_LAYERS] [--hidden_channels HIDDEN_CHANNELS]
               [--dropout DROPOUT] [--batch_size BATCH_SIZE] [--lr LR]
               [--train_ratio TRAIN_RATIO] [--valid_perc VALID_PERC]
               [--epochs EPOCHS] [--eval_steps EVAL_STEPS] [--runs RUNS]
               [--alpha ALPHA] [--eps EPS] [--topk TOPK]
               [--num_walks NUM_WALKS] [--num_steps NUM_STEPS] [--k K]
               [--nthread NTHREAD]
               [--dataset {ogbl-ppa,ogbl-ddi,ogbl-citation2,ogbl-collab,ogbl-vessel,mag}]
               [--relation {write,cite}] [--metric {AUC,MRR,Hits}]
               [--aggrs {mean,lstm,attn}] [--sencoder {LP,ppr,spd,deg}]
               [--reduced] [--use_raw] [--use_node_embedding] [--use_weight]
               [--use_val] [--load_ppr] [--save_ppr] [--log_dir LOG_DIR]
               [--debug]

<details>
  <summary>Optional Arguments</summary>

```
optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --log_steps LOG_STEPS
  --num_layers NUM_LAYERS
  --hidden_channels HIDDEN_CHANNELS
  --dropout DROPOUT
  --batch_size BATCH_SIZE
  --lr LR
  --train_ratio TRAIN_RATIO
  --valid_perc VALID_PERC
  --epochs EPOCHS
  --eval_steps EVAL_STEPS
  --runs RUNS
  --alpha ALPHA         teleport probability in PPR
  --eps EPS             precision of PPR approx
  --topk TOPK           sample size of node set
  --num_walks NUM_WALKS
                        number of walks
  --num_steps NUM_STEPS
                        step of walks
  --k K                 negative samples
  --nthread NTHREAD     number of threads
  --dataset {ogbl-ppa,ogbl-ddi,ogbl-citation2,ogbl-collab,ogbl-vessel,mag}
                        dataset name
  --relation {write,cite}
                        relation type
  --metric {AUC,MRR,Hits}
                        metric for evaluating performance
  --aggrs {mean,lstm,attn}
                        type of set neural encoder
  --sencoder {LP,ppr,spd,deg}
                        type of structure encoder
  --reduced             whether to compress structural features
  --use_raw             whether to use raw features as input
  --use_node_embedding  whether to load node embedding
  --use_weight          whether to use edge weight as input
  --use_val             whether to use val as input
  --load_ppr            whether to load precomputed ppr
  --save_ppr            whether to save calculated ppr
  --log_dir LOG_DIR     log directory
  --debug               whether to use debug mode
```
</details>