from dnutils import ProgressBar

from .common import *
from ..grounding.default import DefaultGroundingFactory
from ..constants import HARD
from ..errors import SatisfiabilityException


class GSMLN_L(AbstractLearner):

    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._stat = None
        self._ls = None
        self._eworld_idx = None
        self._lastw = None

    def _f(self, w):
        return -(w[0]-1)**2
        # return 0
                
    
    def _grad(self, w):
        return [-2*(w[0]-1), 0]
        # return numpy.zeros(len(self.mrf.formulas), numpy.float64)


    def _prepare(self):
        feat_preds = []
        for pred in self.mrf.predicates:
            if pred.feature:
                feat_preds.append(pred.name)

        feat_dict, feat_len = self.get_features_dict(feat_preds)
        print(feat_dict, feat_len)

        for formula in self.mrf.formulas:
            if formula.check_neural():
                self.build_nn(formula, feat_dict, feat_len)


        # print(self.mrf.formulas[0].neural)
        # print(features)
        # print(self.mrf.countworlds())
        # grounder = DefaultGroundingFactory(self.mrf)
        # self.mrf.print_evidence_vars()

        # print(self.mrf.domains)
        # for gf in grounder.itergroundings():
        #     print(gf)
            # for lit in gf.cnf().children:
            #     print(lit.predname, lit.gndatom.args)

    def get_features_dict(self, feat_preds):
        '''
        computes the dictionaries for each feature predicate with
        respective vectors and the lenghts
        '''
        k = []
        v = []
        feat_dict = {}
        feat_len = {}

        with open('smokers.features', 'r') as file:
            features = file.read().split('\n')

        for feature in features:
            k.append(feature.split()[0])
            v.append(feature.split()[1:])

        temp = dict(zip(k,v))

        for feat_pred in feat_preds:
            fp = self.mrf.mln.predicate(feat_pred)
            feat_dict[feat_pred] = {i:temp[i] for i in self.mrf.domains[fp.argdoms[fp.feature_idx]]}
            feat_len[feat_pred] = len(feat_dict[feat_pred][self.mrf.domains[fp.argdoms[fp.feature_idx]][0]])
        
        return feat_dict, feat_len
    

    def build_nn(self, formula, feat_dict, feat_len):
        '''
        build the neural network for each function
        '''
        nn_input = 0
        for atom in formula.atomic_constituents():
            if atom.predname in feat_len.keys():
                nn_input += feat_len[atom.predname]

        print(nn_input)



