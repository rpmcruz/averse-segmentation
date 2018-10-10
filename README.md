# Averse Segmentation

A paper has been submitted to a conference based on this work:

* Averse Deep Semantic Segmentation. Cruz et al. (2019) [not published yet]

This paper is an extension for image segmentation, based on our previous work on averse classifiers:

* [Constraining Type II Error: Building Intentionally Biased Classifiers.](https://link.springer.com/chapter/10.1007/978-3-319-59147-6_47) Cruz et al. 14th International Work-Conference on Artificial Neural Networks. LNCS Springer. [Oral] (2017)

## Usage

Run "python3 run.py [OPTIONS]" where OPTIONS are some of the definitions to be used to train the model. The most important follow. Please consult `run.py` for the full list.

* `--dataset` choose one of the datasets, loaded from mydataset.py (mandatory)
* `--model` choose one of the models, created on mymodel.py (mandatory)
* `--depth` depth of each encoding/decoder part of the model (default: 2)
* `--rho` the amount of false negatives to tolerate
* `--term [squared/relu_squared/entropy/old]` the A-terms from the paper
* `--pingpong` the ping-pong method from the paper
* `--warmup` the warmup method from the paper
