﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿Copyright (c) 2019 ETH Zurich, Lukas Cavigelli, Georg Rutishauser, Luca Benini

# EBPC: Extended Bit-Plane Compression for Deep Neural Network Inference and Training Accelerators

If you find this work useful in your research, please cite

```
@article{cavigelli2019epbc,
  title={{EBPC}: {E}xtended {B}it-{P}lane {C}ompression for {D}eep {N}eural {N}etwork {I}nference and {T}raining {A}ccelerators},
  author={Cavigelli, Lukas and Rutishauser, Georg and Benini, Luca},
  year={2019}
}
@inproceedings{cavigelli2018bitPlaneCompr,
  title={{E}xtended {B}it-{P}lane {C}ompression for {C}onvolutional {N}eural {N}etwork {A}ccelerators},
  author={Cavigelli, Lukas and Benini, Luca},
  booktitle={Proc. IEEE AICAS}, year={2018}
}
```
The paper is available on arXiv at https://arxiv.org/abs/1908.11645 and https://arxiv.org/abs/1810.03979.
The code provided here is not intended as a general framework or library of any kind, but is 
barely-cleaned research code to help reproduce our experimental results.  

The code for the hardware implementation is available here: https://github.com/pulp-platform/stream-ebpc

### Installation
##### Clone repo and setup conda environment

Clone this repo including its submodules: 

`git clone --recursive [URL to Git repo]`

Setup the conda environment with the list of provided dependencies: 

`conda env create -f=./conda-env.yaml -p ./conda-env`

##### Download and prepare dataset
You can get the ILSVRC 2012 data directly from [here](http://image-net.org/challenges/LSVRC/2012/), or much faster via torrent. 
Follow the instructions within QuantLab to prepare the dataset (and use the scripts in QuantLab/ImageNet to prepare a standard ILSVRC12 dataset setup). Point the `ilsvrc12` link to the right destination or extract the data directly in there. 

##### Install and train models with QuantLab
You can download QuantLab [here](https://github.com/spallanzanimatteo/QuantLab). For some analyses, e.g. the sparsity analysis 
by epoch, train the corresponding network (full precision/no quantization; choose to log a snapshot of the network every epoch). 
This generates several 10s of GBs of data, hence we prefer not to share them unless necessary. 

### Structure

##### Algorithm evaluations

Implementation:

|  filename  |  content |
|---|---|
| dataCollect.py | functions to 1) load the models, 2) obtain the feature maps within a model, 3) read data from tensorboard log files |
| referenceImpl.py | implementations of the baseline compression algorithms (CSC, zeroRLE, ZVC) |
| bpcUtils.py | contains the implementation of the compressor incl. all its building blocks as well as the quantization function and the value-to-binary conversion |
| analysisTools.py | provides the Analyzer class which is instantiated for a specific default quantization method and compressor. It provides functions to get properties such as the compression ratio for a given feature map tensor, its sparsity, ... |
| groupedBarPlot.py | utility function: creates nice grouped bar plots based on PyPlot |
| reporting.py | contains functions to read, write, and parse CSV files |

Analyses and visualizations: 

|  filename  |  content |
|---|---|
| totalComprRatio_generate.py  | script to generate/collect all the statistics for the total compression ratio figure; writes results to file; long run-timescript to generate/collect all the statistics for the total compression ratio figure; writes results to file; long run-time |
| totalComprRatio_plot.ipynb  | visualizes all the results on the total compression ratio |
| sparsityByEpoch.ipynb  | generates the figure as suggested by the filename |
| sparsityBoxplot.ipynb  | generates the figure as suggested by the filename |
| histogram.ipynb  | generates the figure as suggested by the filename |


##### Hardware implementation
(TBD)

### Licensing
Please refer to the LICENSE file for the licensing of the algorithm evaluation code and to the LICENSE_HARDWARE file for the licensing of the files related to the hardware implementation. 



