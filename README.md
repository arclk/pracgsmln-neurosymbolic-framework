
pracgsmln -- Ground Specific Markov logic networks in Python
============================================================

pracgsmln is a a neuro-symbolic framework combining the nerual networks with the symbolic method of the Markov Logic Networks


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
