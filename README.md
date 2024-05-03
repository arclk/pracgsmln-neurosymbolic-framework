# PyTorch Implementation of Neurosymbolic Framework with Markov Logic Networks (MLNs)

This is a PyTorch implementation of a neurosymbolic framework that integrates neural networks with Markov Logic Networks (MLNs). The framework allows for merging symbolic and sub-symbolic reasoning by introducing neural networks to provide grounding-specific weights for different instantiations of the same first-order logic formula in MLNs.

## Description

Markov Logic Networks (MLNs) are a powerful approach for combining first-order logic and probabilistic graphical models. However, standard MLNs have limitations in capturing complex interactions between features and handling different types of data. This framework addresses these limitations by assigning different weights to different groundings (instantiations) of the same first-order logic rule, effectively injecting sub-symbolic capabilities into MLNs.

The key features of this framework include:

1. **Grounding-Specific Weights**: Instead of using a single weight for an entire first-order logic formula, the framework learns grounding-specific weights, i.e., different weights for different instantiations of the formula's variables. This allows for capturing complex interactions between features and handling different types of data, such as real-valued attributes.

2. **Neural Network Integration**: The grounding-specific weights are computed using neural networks, which take the instantiated variables (or their feature representations) as input. This integration of neural networks with MLNs enables the framework to leverage the representational power of deep learning.

3. **Scalable Learning**: The framework employs a pseudo-likelihood approach for weight learning, which does not require expensive inference over the model, making it scalable to large datasets.

4. **Modular Design**: The symbolic (first-order logic rules) and sub-symbolic (neural network) components are decoupled, allowing for easy modification or replacement of either component without affecting the other.


Installation
------------

    $ conda env create -f environment.yml
    
    $ python test.py -h
    
        usage: test.py [-h] [-d DATASET] [-mln MLN] [-db DB] [-f FEATURE] [-ep EPOCHS] [-pr PRETRAIN] [-es EARLYSTOP]

        A neurosymbolic framework for Markov Logic Networks!

        optional arguments:
          -h, --help                                       show this help message and exit
          -d DATASET, --dataset DATASET                    This is the dataset directory
          -mln MLN, --mln MLN                              This is the path for the mln file
          -db DB, --db DB                                  This is the path for the db file
          -f FEATURE, --feature FEATURE                    This is the path for the feature file
          -ep EPOCHS, --epochs EPOCHS                      This is the number of epochs for learning
          -pr PRETRAIN, --pretrain PRETRAIN                This is the number of epochs for the pretraining
          -es EARLYSTOP, --earlystop EARLYSTOP             This is the number of epochs of patience for the early stopping

Example
-------------
To run MNIST experiment with the reduced dataset for 100 epochs:

    $ python test.py -d mnist -mln mnist.mln -db mnist_train_simple.db -f mnist_train_simple.features -ep 100

To run abstRCT experiment with the reduced dataset for 200 epochs with a pretraining of 100 epochs and early stopping of 20 epochs:

    $ python test.py -d abstrct -mln abstrct.mln -db neoplasm25_train_simple.db -f neoplasm25_train.features -ep 200 -pr 100 -es 20
