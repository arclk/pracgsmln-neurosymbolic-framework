import re
import sys
import time
from math import *
import torch

from dnutils import out, logs

from .network import *
from .database import Database
from .errors import (MRFValueException, NoSuchDomainError, NoSuchPredicateError)
from .methods import InferenceMethods
from .constants import HARD
from .mrfvars import (MutexVariable, SoftMutexVariable, FuzzyVariable,
    BinaryVariable)
from .util import fstr, logx, mergedom, CallByRef, Interval
from ..logic import FirstOrderLogic
from ..logic.common import Logic
from ..logic.fuzzy import FuzzyLogic
from .grounding import *


class GSMLN_MRF(object):
    def __init__(self, mln, val, test, mrfs):
        self.mrf_train = None
        self.mrf_val = None
        self.mrf_test = None

        # self.mrf_train = mln.ground(dbs['train'])
        # if val:
        #     mrf_val = mln.ground(dbs['val'])
        # if test:
        #     mrf_test = mln.ground(dbs['test'])

        self.mrf_train = mrfs['train']
        if val:
            self.mrf_val = mrfs['val']
        if test:
            self.mrf_test = mrfs['test']

        # ground members
        self.formulas = list(mln.formulas)
        self.ftype = ['nn', 'nn', 'hard', 'hard']
        
        # AA: add the neural part of the MRF
        self.feat_preds = []
        self.feat_dict = {}
        self.idx_to_feat = {}
        self.idxnnform = []
        self.nnformulas = torch.nn.ModuleList()

        # Store the feature predicates
        for pred in mln.predicates:
            if pred.feature:
                self.feat_preds.append(pred.name)

        return


    def build_network(self):
        '''
        Build the neural part of the MRF
        '''
        # # Store the feature predicates
        # for pred in self.predicates:
        #     if pred.feature:
        #         self.feat_preds.append(pred.name)

        self.feat_dict, feat_len = self.get_features_dict(self.feat_preds)

        # Define the modules of the formulas
        for idx, formula in enumerate(self.formulas):
            if not formula.ishard:
                if formula.check_neural():
                    self.idxnnform.append(idx)
                    nn_input = 0
                    for atom in formula.atomic_constituents():
                        if atom.predname in feat_len.keys():
                            nn_input += feat_len[atom.predname]
                    # self.nnformulas.append(Network(nn_input))
                    # self.nnformulas.append(SA_Network())
                    # self.nnformulas.append(ABSTRCT_Network2())

                    if idx == 0:
                        self.nnformulas.append(ABSTRCT_Network1())
                    else:
                        self.nnformulas.append(ABSTRCT_Network2())
                else:
                    self.nnformulas.append(Standard_Formula())


    def get_features_dict(self, feat_preds):
        '''
        computes the dictionaries for each feature predicate with
        respective vectors and the lenghts
        '''
        k = []
        v = []
        feat_dict = {}
        feat_len = {}

        with open('abstrct/neoplasm25_train.features', 'r') as file:
            features = file.read().split('\n')

        for feature in features:
            k.append(feature.split()[0])
            v.append(list(map(int, feature.split()[1:])))

        temp = dict(zip(k,v))

        for feat_pred in feat_preds:
            fp = self.mln.predicate(feat_pred)
            feat_dict[feat_pred] = {i:temp[i] for i in self.domains[fp.argdoms[fp.feature_idx]]}
            feat_len[feat_pred] = len(feat_dict[feat_pred][self.domains[fp.argdoms[fp.feature_idx]][0]])
        
        print(feat_len)
        return feat_dict, feat_len

