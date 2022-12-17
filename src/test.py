"""
Created on Oct 28, 2015

@author: nyga
"""
import os

from pracmln.mln.learning import *
from pracmln.mln.network import *
from pracmln import MLN, Database
from pracmln import query, learn
from pracmln.mln.grounding import DefaultGroundingFactory
from pracmln.mlnlearn import EVIDENCE_PREDS
import time
import torch
import sys
import argparse

from pracmln.utils import locs



def test_GSMLN():

    device = "cpu"

    parser = argparse.ArgumentParser(description='A neurosymbolic framework for Markov Logic Networks!')

    parser.add_argument('-d', '--dataset', type=str, help="This is the dataset directory")
    parser.add_argument('-mln', '--mln', type=str, help="This is the path for the mln file")
    parser.add_argument('-db', '--db', type=str, help="This is the path for the db file")
    parser.add_argument('-f', '--feature', type=str, help="This is the path for the feature file")
    parser.add_argument('-ep', '--epochs', type=int, help="This is the number of epochs for learning")
    parser.add_argument('-pr', '--pretrain', type=int, help="This is the number of epochs for the pretraining")
    parser.add_argument('-es', '--earlystop', type=int, help="This is the number of epochs of patience for the early stopping")

    args = parser.parse_args()
    dataset = args.dataset
    mlnfile = dataset + '/' + args.mln
    dbfile = dataset + '/' + args.db
    featfile = dataset + '/' + args.feature
    epochs = args.epochs
    pretrain = args.pretrain
    earlystop_epochs = args.earlystop
    if (earlystop_epochs):
        earlystop = True
    else:
        earlystop = False

    methods = {'mnist': GSMLN_MNIST, 'abstrct': GSMLN_ABSTRCT}
    method = methods[dataset]

    print(mlnfile, dbfile)
    mln = MLN(mlnfile=mlnfile, grammar='GSMLNGrammar')
    db = Database.load(mln, dbfiles=dbfile)
    mrf = mln.ground(db[0])
    mrf.build_network(featfile)

    # mrf.itergroundings()

    learned_mln = mln.gsmln_learn(mrf=mrf, val=True, method=method, epochs=epochs, pretrain=pretrain, 
        early_stopping=earlystop, early_stopping_epochs=earlystop_epochs)

    # mln.gsmln_learn(databases=dbs, verbose=True)

    # this uses the method from mlnlearn.py
    # learn(method='GSMLN_L', mln=mln, db=dbs, verbose=True).run()
    
    # learned_mln.write()


def main():
    test_GSMLN()
    # sys.exit("Error message")
    

if __name__ == '__main__':
    main()
