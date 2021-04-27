from dnutils import ProgressBar

from .common import *
from ..grounding.default import DefaultGroundingFactory
from ..constants import HARD
from ..errors import SatisfiabilityException
from ..util import fsum, temporary_evidence

from collections import defaultdict

import torch
from torchviz import make_dot


class GSMLN_L(AbstractLearner):

    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None
        self._varidx2fidx = None
        self._lastw = None
        self.wt = []
        self.nn = []
        self.nn_inputs = {}
        # self.network = Network()


#     def _prepare(self):
#         logger.debug("computing statistics...") 
#         self._compute_statistics()
#         # self.network._compute_statistics()
# #         print self._stat    


    def _pl(self, varidx):
        '''
        Computes the pseudo-likelihoods for the given variable under weights w. 
        '''

        var = self.mrf.variable(varidx)
        
        # AA: return [0,1] for feature predicates
        if var.predicate.name in self.mrf.feat_preds:
            return torch.tensor([0,1], dtype=torch.float)
        
        values = var.valuecount()
        gfs = self._varidx2fidx.get(varidx)
        if gfs is None: # no list was saved, so the truth of all formulas is unaffected by the variable's value
            # uniform distribution applies
            p = 1.0 / values
            return p * torch.ones(values)
        sums = torch.zeros(values)#numpy.zeros(values)
        for fidx, groundings in gfs.items():
            for gidx in groundings:
                for validx, n in enumerate(self._stat[fidx][gidx][varidx]):
                    # if w[fidx] == HARD: 
                    #     # set the prob mass of every value violating a hard constraint to None
                    #     # to indicate a globally inadmissible value. We will set those ones to 0 afterwards.
                    #     if n == 0: sums[validx] = None
                    if sums[validx] is not None:
                        # don't set it if this value has already been assigned marked as inadmissible.
                        # nn_input = self.nn_inputs[fidx][gidx]
                        # print(nn_input)
                        # w = self.wt[fidx][gidx]
                        # print(n*w)
                        sums[validx] = sums[validx] + n * self.wt[fidx][gidx]
        expsums = torch.exp(sums)
        z = torch.sum(expsums)
        if z == 0: raise SatisfiabilityException('MLN is unsatisfiable: all probability masses of variable %s are zero.' % str(var))
        return expsums/z
#         sum_max = numpy.max(sums)
#         sums -= sum_max
#         expsums = numpy.sum(numpy.exp(sums))
#         s = numpy.log(expsums)    
#         return numpy.exp(sums - s)


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self._pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls(self):
        # print(w.requires_grad)
        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(nn(self.nn_inputs[fidx]))
        # print(self.wt)

        # if self._pls is None or self._lastw is None or self._lastw != list(self.wt):
        self._pls = [self._pl(var.idx) for var in self.mrf.variables]
        self._lastw = list(self.wt)
#             self.write_pls()
        self._pls = torch.stack(self._pls)
        # print(self._pls)
        # print(self._pls)
        # print(self._pls.size())
    

    def _f(self, w):
        # print(w.requires_grad)
        self._compute_pls(w)
        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.idx][var.evidence_value_index()]
            if p == 0: p = 1e-10 # prevent 0 probabilities
            probs.append(p)
        temp = torch.tensor(probs)
        temp = torch.log(temp)
        temp = torch.sum(temp)

        # temp = torch.tensor(temp)
        return temp


    def forward(self):
        # print(w.requires_grad)
        self._compute_pls()
        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.idx][var.evidence_value_index()]
            if p == 0: p = 1e-10 # prevent 0 probabilities
            probs.append(p)
        print(probs)
        temp = torch.stack(probs)
        temp = torch.log(temp)
        print(temp)
        # temp = torch.tensor(temp)
        temp = -torch.sum(temp)
        # print(temp.requires_grad)

        return temp

    def one_step(self):
        # f = self.forward()  
        # make_dot(f).save()
        # # print(self.wt.grad)
        # # optimizer.zero_grad()
        # f.backward()
        # # optimizer.step()
        # # print(self.wt.grad)
        self.training_step()

        return [1,2]

    def _grad(self, w):
        # self._compute_pls(w)
        grad = numpy.zeros(len(self.mrf.formulas), numpy.float64)        
        for fidx, groundings in self._stat.items():
            for gidx, varval in groundings.items():
                for varidx, counts in varval.items():
                    evidx = self.mrf.variable(varidx).evidence_value_index()
                    g = counts[evidx]
                    for i, val in enumerate(counts):
                        g -= val * self._pls[varidx][i]
                    grad[fidx] += g
        # self.grad_opt_norm = sqrt(float(fsum([x * x for x in grad])))
        # print(grad)
        return numpy.array(grad)


    def training_step(self):
        # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        optimizer = torch.optim.Adam(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        for epoch in range(10):
            optimizer.zero_grad()
            f = self.forward()
            # make_dot(f).save()
            f.backward()
            optimizer.step()
            # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))


    def _addstat(self, fidx, gidx, varidx, validx, inc=1):
        if fidx not in self._stat:
            self._stat[fidx] = {}
        if gidx not in self._stat[fidx]:
            self._stat[fidx][gidx] = {}
        d = self._stat[fidx][gidx]
        if varidx not in d:
            d[varidx] = [0] * self.mrf.variable(varidx).valuecount()
        d[varidx][validx] += inc
        

    def _addvaridx2fidx(self, fidx, gidx, varidx):
        if varidx not in self._varidx2fidx:
            self._varidx2fidx[varidx] = {}
        if fidx not in self._varidx2fidx[varidx]:
            self._varidx2fidx[varidx][fidx] = set()

        self._varidx2fidx[varidx][fidx].add(gidx)


    def _compute_statistics(self):
        '''
        computes the statistics upon which the optimization is based
        '''
        self._stat = {}
        self._varidx2fidx = {}
        grounder = DefaultGroundingFactory(self.mrf, simplify=False, unsatfailure=True, verbose=self.verbose, cache=0)
        gidx = 0
        old_idx = None
        for f in grounder.itergroundings():
            # AA: to take in consideration the various groundings but have to be improved
            if f.idx == old_idx:
                gidx += 1
            else:
                gidx = 0
            old_idx = f.idx 
            print(f.atomic_constituents())
            for gndatom in f.gndatoms():
                var = self.mrf.variable(gndatom)
                with temporary_evidence(self.mrf):
                    for validx, value in var.itervalues():
                        var.setval(value, self.mrf.evidence)
                        truth = f(self.mrf.evidence) 
                        if truth != 0:
                            # self._varidx2fidx[var.idx].add(f.idx)
                            self._addvaridx2fidx(f.idx, gidx, var.idx)
                            self._addstat(f.idx, gidx, var.idx, validx, truth)
        print(self._varidx2fidx)
        print(self._stat)


    # def _f(self, w):
    #     return -(w[0]-1)**2
    #     # return 0
                
    
    # def _grad(self, w):
    #     return [-2*(w[0]-1), 0]
    #     # return numpy.zeros(len(self.mrf.formulas), numpy.float64)


    def _prepare(self):
        # self.feat_preds = []
        # for pred in self.mrf.predicates:
        #     if pred.feature:
        #         self.feat_preds.append(pred.name)

        # feat_dict, feat_len = self.get_features_dict(self.feat_preds)
        # print(feat_dict, feat_len)

        # for formula in self.mrf.formulas:
        #     if formula.check_neural():
        #         self.nn.append(self.build_nn(formula, feat_dict, feat_len))

        # print(self.nn)

        print(self.mrf.nnformulas)
        # print(self.mrf.formulas[0].neural)
        # print(features)
        # print(self.mrf.countworlds())
        grounder = DefaultGroundingFactory(self.mrf)
        # self.mrf.print_evidence_vars()

        # print(self.mrf.domains)
        
        # Create the function nn inputs
        for gf in grounder.itergroundings():
            if gf.idx not in self.nn_inputs:
                self.nn_inputs[gf.idx] = []
            print(gf.atomic_constituents())
            temp = []
            for atom in gf.atomic_constituents():
                if atom.predname in self.mrf.feat_dict.keys():
                    temp += self.mrf.feat_dict[atom.predname][atom.args[1]]
            self.nn_inputs[gf.idx].append(list(map(float,temp)))

        prova = []
        for k, v in self.nn_inputs.items():
            prova.append(torch.tensor(v))
        # self.nn_inputs = torch.tensor(self.nn_inputs)
        print(self.nn_inputs)
        print(prova)
        self.nn_inputs = prova
        self._compute_statistics()


    # def get_features_dict(self, feat_preds):
    #     '''
    #     computes the dictionaries for each feature predicate with
    #     respective vectors and the lenghts
    #     '''
    #     k = []
    #     v = []
    #     feat_dict = {}
    #     feat_len = {}

    #     with open('smokers.features', 'r') as file:
    #         features = file.read().split('\n')

    #     for feature in features:
    #         k.append(feature.split()[0])
    #         v.append(feature.split()[1:])

    #     temp = dict(zip(k,v))

    #     for feat_pred in feat_preds:
    #         fp = self.mrf.mln.predicate(feat_pred)
    #         feat_dict[feat_pred] = {i:temp[i] for i in self.mrf.domains[fp.argdoms[fp.feature_idx]]}
    #         feat_len[feat_pred] = len(feat_dict[feat_pred][self.mrf.domains[fp.argdoms[fp.feature_idx]][0]])
        
    #     return feat_dict, feat_len
    

    # def build_nn(self, formula, feat_dict, feat_len):
    #     '''
    #     build the neural network for each function
    #     '''
    #     nn_input = 0
    #     for atom in formula.atomic_constituents():
    #         if atom.predname in feat_len.keys():
    #             nn_input += feat_len[atom.predname]

    #     print(nn_input)
    #     model = torch.nn.Linear(nn_input, 1)
    #     return model



