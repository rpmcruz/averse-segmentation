# Averse Segmentation

Traditionally, the trade-off between false positive and false negatives is specified in terms of relative costs using a cost matrix. But it might be more natural in some contexts to apply this trade-off in terms of an absolute cost: for example, we want to minimize false negatives while keeping false positives below a certain $$\rho$$.

![Changing threshold](https://raw.githubusercontent.com/rpmcruz/averse-segmentation/master/figure_threshold.png "Changing threshold")

This work is the implementation used in the paper:

* Averse Deep Semantic Segmentation. Cruz et al. 41st Engineering in Medicine and Biology Conference. (2019)

This paper is an extension for image segmentation, based on our previous work on averse classifiers, which are applied to tabular data:

* [Constraining Type II Error: Building Intentionally Biased Classifiers.](https://link.springer.com/chapter/10.1007/978-3-319-59147-6_47) Cruz et al. 14th International Work-Conference on Artificial Neural Networks. LNCS Springer. (2017)

## Usage

Run "python3 run.py [OPTIONS]" where OPTIONS are some of the definitions to be used to train the model. The most important follow. Please consult `run.py` for the full list.

* `--dataset` choose one of the datasets, loaded from mydataset.py (mandatory)
* `--model` choose one of the models, created on mymodel.py (mandatory)
* `--depth` depth of each encoding/decoder part of the model (default: 2)
* `--rho` the amount of false negatives to tolerate
* `--term [squared/relu_squared/entropy/old]` the A-terms from the paper
* `--pingpong` the ping-pong method from the paper
* `--warmup` the warmup method from the paper
