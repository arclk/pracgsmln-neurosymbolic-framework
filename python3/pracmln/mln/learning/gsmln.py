from dnutils import ProgressBar

from .common import *
from ..grounding.default import DefaultGroundingFactory
from ..constants import HARD
from ..errors import SatisfiabilityException
from ..util import fsum, temporary_evidence

from collections import defaultdict

import torch
from torchviz import make_dot

import time


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


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self._pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls(self):
        # if self._pls is None or self._lastw is None or self._lastw != list(self.wt):
        self._pls = [self._pl(var.idx) for var in self.mrf.variables]
        self._lastw = list(self.wt)
            # self.write_pls()
        self._pls = torch.stack(self._pls)
        # print(self._pls)
        # print(self._pls)
        # print(self._pls.size())
    

    def forward(self):
        
        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            # print(self.nn_inputs[fidx])
            if (self.nn_inputs[fidx].nelement() == 0):
                self.wt.append(nn(torch.ones(1)).repeat(2))
            else:
                self.wt.append(nn(self.nn_inputs[fidx]))

        # print(self.wt)

        self._compute_pls()
        
        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.idx][var.evidence_value_index()]
            if p == 0: p = 1e-10 # prevent 0 probabilities
            probs.append(p)
        # print(probs)
        temp = torch.stack(probs)
        temp = torch.log(temp)
        # print(temp)
        # temp = torch.tensor(temp)
        temp = -torch.sum(temp)
        # print(temp.requires_grad)

        return temp


    def train(self):
        # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        optimizer = torch.optim.Adam(self.mrf.nnformulas[0].parameters())
        # optimizer = torch.optim.Adam(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        start_train = time.time()

        for epoch in range(100):
            optimizer.zero_grad()
            f = self.forward()
            # make_dot(f).save()
            f.backward()
            optimizer.step()
            print(f'Epoch {epoch+1} \t Loss: {f}')
            # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))

        end_train= time.time()
        print(f'Time for training: {end_train-start_train}')

        print(self.wt)

        return [1,2]


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


    def _prepare(self):

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

# This class was for separate method
class GSMLN_LSeparate(AbstractLearner):

    
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


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self._pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls(self):
        # print(w.requires_grad)
        
        # print(self.wt)

        # if self._pls is None or self._lastw is None or self._lastw != list(self.wt):
        self._pls = [self._pl(var.idx) for var in self.mrf.variables]
        self._lastw = list(self.wt)
        self._pls = torch.stack(self._pls)
        # print(self._pls)
        # print(self._pls)
        # print(self._pls.size())

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
        self.training_step()
        return [1,2]

    def _grad(self):
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
        optimizer = torch.optim.Adam(self.mrf.nnformulas[0].parameters())
        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        for epoch in range(10):
            optimizer.zero_grad()
 
            self.wt = []
            for fidx, nn in enumerate(self.mrf.nnformulas):
                self.wt.append(nn(self.nn_inputs[fidx]))

            f = self.forward()
            grad = self._grad()

            for fidx, nn in enumerate(self.mrf.nnformulas):
                loss = criterion(self.wt[fidx], None)
                loss.backward()
                for par in fc.parameters():
                    par.grad *= grad[fidx]

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

# This works only with MNIST
class GSMLN_LC_MNIST(AbstractLearner):

    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None
        self._varidx2fidx = None
        self._lastw = None
        self.wt = []
        self.nn = []
        self.nn_inputs = {}
        self.train_set = None
        self.train_set = None
        # self.network = Network()


    def _pl(self, varidx):
        '''
        Computes the pseudo-likelihoods for the given variable under weights w. 
        '''

        var = self.mrf.variable(varidx)
        
        # AA: return [0,1] for feature predicates
        if var.predicate.name in self.mrf.feat_preds:
            return torch.tensor([0,1], dtype=torch.float)

        # WARNING: This works only with MNIST 
        if len(var.name) == 13:
            wt_idx = int(var.name[9])
            wt_idx2 = int(var.name[11])
        if len(var.name) == 14:
            wt_idx = int(var.name[9:11])
            wt_idx2 = int(var.name[12])
        

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
                    if sums[validx] is not None:
                        sums[validx] = sums[validx] + n * self.wt[fidx][wt_idx][wt_idx2]
        expsums = torch.exp(sums)
        z = torch.sum(expsums)
        if z == 0: raise SatisfiabilityException('MLN is unsatisfiable: all probability masses of variable %s are zero.' % str(var))
        return expsums/z


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self._pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls(self):
        self._pls = [self._pl(var.idx) for var in self.mrf.variables]
        self._lastw = list(self.wt)
        self._pls = torch.stack(self._pls)


    def forward(self):

        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(nn(torch.unsqueeze(self.train_set[0][:100],1).float()))

        self._compute_pls()
        
        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.idx][var.evidence_value_index()]
            if p == 0: p = 1e-10 # prevent 0 probabilities
            probs.append(p)
        temp = torch.stack(probs)
        temp = torch.log(temp)
        temp = -torch.sum(temp)
        return temp


    def train(self):
        # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        optimizer = torch.optim.Adam(self.mrf.nnformulas[0].parameters())
        # optimizer = torch.optim.Adam(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        start_train = time.time()

        for epoch in range(500):
            optimizer.zero_grad()
            f = self.forward()
            # make_dot(f).save()
            f.backward()
            optimizer.step()
            # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))
            print(f'Epoch {epoch+1} \t Loss: {f}')
            # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))

        end_train= time.time()
        print(f'Time for training: {end_train-start_train}')

        return [1,2]


    def evaluate(self):
        for fidx, nn in enumerate(self.mrf.nnformulas):
            pred = nn(torch.unsqueeze(self.test_set[0][:100],1).float())
            print(torch.argmax(pred, dim=1))


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
        start_comp = time.time()
        self._stat = {}
        self._varidx2fidx = {}
        grounder = DefaultGroundingFactory(self.mrf, simplify=False, unsatfailure=True, verbose=self.verbose, cache=0)
        gidx = 0
        old_idx = None
        for f in grounder.itergroundings():
            # print(f.idx)
            # AA: to take in consideration the various groundings but have to be improved
            if f.idx == old_idx:
                gidx += 1
            else:
                gidx = 0
            old_idx = f.idx 
            # print(f.atomic_constituents())
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
        # print(self._varidx2fidx)
        # print(self._stat)
        end_comp = time.time()
        print(f'Time for function comp_stat: {end_comp-start_comp}')


    def _prepare(self):
        start_prep = time.time()

        print(self.mrf.nnformulas)
        grounder = DefaultGroundingFactory(self.mrf)
        
        # Create the function nn inputs
        for gf in grounder.itergroundings():
            if gf.idx not in self.nn_inputs:
                self.nn_inputs[gf.idx] = []
            # print(gf.atomic_constituents())
            temp = []
            for atom in gf.atomic_constituents():
                if atom.predname in self.mrf.feat_dict.keys():
                    temp += self.mrf.feat_dict[atom.predname][atom.args[1]]
            self.nn_inputs[gf.idx].append(list(map(float,temp)))

        prova = []
        for k, v in self.nn_inputs.items():
            prova.append(torch.tensor(v))
        # self.nn_inputs = torch.tensor(self.nn_inputs)
        # print(prova)
        self.nn_inputs = prova
        # print(self.nn_inputs)
        self.train_set = torch.load('mnist/mnist_training.pt')
        self.test_set = torch.load('mnist/mnist_test.pt')

        self._compute_statistics()

        end_prep = time.time()
        print(f'Time for function prepare: {end_prep-start_prep}')

# This works only with FEVER
class GSMLN_LC_FEVER(AbstractLearner):
    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None
        self._varidx2fidx = None
        self._lastw = None
        self.wt = []
        self.nn = []
        self.nn_inputs = {}
        self.train_set = None
        self.train_set = None
        # self.network = Network()


    def _pl(self, varidx):
        '''
        Computes the pseudo-likelihoods for the given variable under weights w. 
        '''

        var = self.mrf.variable(varidx)
        
        # AA: return [0,1] for feature predicates
        if var.predicate.name in self.mrf.feat_preds:
            return torch.tensor([0,1], dtype=torch.float)

        # if var.predicate.name == 'Link':
        #     return torch.tensor([0,1], dtype=torch.float)

        wt_idx = int(var.name[8])
        # if var.predicate.name == 'Link':
        wt_idx2 = int(var.name[13])

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
                    if sums[validx] is not None:
                        sums[validx] = sums[validx] + n * self.wt[fidx][wt_idx][wt_idx2]
        expsums = torch.exp(sums)
        z = torch.sum(expsums)
        if z == 0: raise SatisfiabilityException('MLN is unsatisfiable: all probability masses of variable %s are zero.' % str(var))

        # print(var.name, expsums)

        return expsums/z


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self._pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls(self):
        # print(w.requires_grad)
        
        self.wt = []
        # for fidx, nn in enumerate(self.mrf.nnformulas):
        #     self.wt.append(nn(self.train_set[0][:10].T, self.train_set[1][:10].T))

        out = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            for claim in self.train_set[0][:10]:
                out.append(nn(claim.repeat(10,1).T, self.train_set[1][:10].T))

            self.wt.append(torch.stack(out))

        # print(self.wt[0])

        self._pls = [self._pl(var.idx) for var in self.mrf.variables]
        self._lastw = list(self.wt)
        self._pls = torch.stack(self._pls)


    def forward(self):
        self._compute_pls()
        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.idx][var.evidence_value_index()]
            if p == 0: p = 1e-10 # prevent 0 probabilities
            probs.append(p)
        temp = torch.stack(probs)
        temp = torch.log(temp)
        temp = -torch.sum(temp)
        return temp

    def one_step(self):
        start_train = time.time()
        self.training_loop()
        # print(self.wt)
        end_train= time.time()
        print(f'Time for training: {end_train-start_train}')

        for fidx, nn in enumerate(self.mrf.nnformulas):
            pred = nn(self.test_set[0][:10].T, self.test_set[1][:10].T)
            print(pred)

        # for fidx, nn in enumerate(self.mrf.nnformulas):
        #     pred = nn(torch.unsqueeze(self.test_set[0][:100],1).float())
        #     print(torch.argmax(pred, dim=1))

        return [1,2]

    def _grad(self):
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


    def training_loop(self):
        # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        optimizer = torch.optim.Adam(self.mrf.nnformulas[0].parameters())
        # optimizer = torch.optim.Adam(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        for epoch in range(100):
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
        start_comp = time.time()
        self._stat = {}
        self._varidx2fidx = {}
        grounder = DefaultGroundingFactory(self.mrf, simplify=False, unsatfailure=True, verbose=self.verbose, cache=0)
        gidx = 0
        old_idx = None
        for f in grounder.itergroundings():
            # print(f.idx)
            # AA: to take in consideration the various groundings but have to be improved
            if f.idx == old_idx:
                gidx += 1
            else:
                gidx = 0
            old_idx = f.idx 
            # print(f.atomic_constituents())
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
        print(self.mrf.variables)
        end_comp = time.time()
        print(f'Time for function comp_stat: {end_comp-start_comp}')


    def _prepare(self):
        start_prep = time.time()

        print(self.mrf.nnformulas)
        grounder = DefaultGroundingFactory(self.mrf)
        
        # Create the function nn inputs
        for gf in grounder.itergroundings():
            print(gf)
            if gf.idx not in self.nn_inputs:
                self.nn_inputs[gf.idx] = []
            # print(gf.atomic_constituents())
            temp = []
            for atom in gf.atomic_constituents():
                if atom.predname in self.mrf.feat_dict.keys():
                    temp += self.mrf.feat_dict[atom.predname][atom.args[1]]
            self.nn_inputs[gf.idx].append(list(map(float,temp)))

        prova = []
        for k, v in self.nn_inputs.items():
            prova.append(torch.tensor(v))
        # self.nn_inputs = torch.tensor(self.nn_inputs)
        # print(prova)
        self.nn_inputs = prova
        # print(self.nn_inputs)
        self.train_set = torch.load('fever/fever_train.pt')
        self.test_set = torch.load('fever/fever_test.pt')

        self._compute_statistics()

        end_prep = time.time()
        print(f'Time for function prepare: {end_prep-start_prep}')

# This works only with IMDB
class GSMLN_LC_IMDB(AbstractLearner):
    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None
        self._varidx2fidx = None
        self._lastw = None
        self.wt = []
        self.nn = []
        self.nn_inputs = {}
        self.train_set = None
        self.train_set = None
        # self.network = Network()


    def _pl(self, varidx):
        '''
        Computes the pseudo-likelihoods for the given variable under weights w. 
        '''

        var = self.mrf.variable(varidx)
        predname = var.predicate.name

        # AA: return [0,1] for feature predicates
        if predname in self.mrf.feat_preds:
            return torch.tensor([0,1], dtype=torch.float)

        # Retreive the index of the weight
        idx = var.gndatoms[0].args[0]
        feat_idx = self.mrf.idx_to_feat[idx]
        wt_idx = self.mrf.feat_dict['Text'][feat_idx][0]


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
                    if sums[validx] is not None:
                        sums[validx] = sums[validx] + n * self.wt[fidx][wt_idx]
        expsums = torch.exp(sums)
        z = torch.sum(expsums)
        if z == 0: raise SatisfiabilityException('MLN is unsatisfiable: all probability masses of variable %s are zero.' % str(var))

        # print(var.name, expsums)

        return expsums/z


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self._pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls(self):
        self._pls = [self._pl(var.idx) for var in self.mrf.variables]
        self._lastw = list(self.wt)
        self._pls = torch.stack(self._pls)


    def forward(self):
        self.wt = []

        # Pass through all the neural networks
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(torch.squeeze(nn(self.train_set[0].T[:100].T)))

        self._compute_pls()

        probs = []
        for var in self.mrf.variables:
            p = self._pls[var.idx][var.evidence_value_index()]
            if p == 0: p = 1e-10 # prevent 0 probabilities
            probs.append(p)
        temp = torch.stack(probs)
        temp = torch.log(temp)
        temp = -torch.sum(temp)
        return temp


    def train(self):
        # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        optimizer = torch.optim.Adam(self.mrf.nnformulas[0].parameters())
        # optimizer = torch.optim.Adam(dict([nn.parameters() for nn in self.mrf.nnformulas]))
        start_train = time.time()

        for epoch in range(50):
            optimizer.zero_grad()
            f = self.forward()
            # make_dot(f).save()
            f.backward()
            optimizer.step()
            print(f'Epoch {epoch+1} \t Loss: {f}')
            # print(dict([nn.parameters() for nn in self.mrf.nnformulas]))

        end_train= time.time()
        print(f'Time for training: {end_train-start_train}')

        return [1,2]


    def evaluate(self):
        with torch.no_grad():
            for fidx, nn in enumerate(self.mrf.nnformulas):
                pred = torch.squeeze(nn(self.test_set[0].T[:100].T))
                print(pred)


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
        start_comp = time.time()
        self._stat = {}
        self._varidx2fidx = {}
        grounder = DefaultGroundingFactory(self.mrf, simplify=False, unsatfailure=True, verbose=self.verbose, cache=0)
        gidx = 0
        old_idx = None
        for f in grounder.itergroundings():
            # print(f.idx)
            # AA: to take in consideration the various groundings but have to be improved
            if f.idx == old_idx:
                gidx += 1
            else:
                gidx = 0
            old_idx = f.idx 
            # print(f.atomic_constituents())
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
        print(self.mrf.variables)
        end_comp = time.time()
        print(f'Time for function comp_stat: {end_comp-start_comp}')


    def _prepare(self):
        start_prep = time.time()

        print(self.mrf.nnformulas)
        grounder = DefaultGroundingFactory(self.mrf)
        
        # Create the function nn inputs
        for gf in grounder.itergroundings():
            print(gf)
            if gf.idx not in self.nn_inputs:
                self.nn_inputs[gf.idx] = []
            # print(gf.atomic_constituents())
            temp = []
            for atom in gf.atomic_constituents():
                if atom.predname in self.mrf.feat_dict.keys():
                    temp += self.mrf.feat_dict[atom.predname][atom.args[1]]
            self.nn_inputs[gf.idx].append(list(map(float,temp)))

        prova = []
        for k, v in self.nn_inputs.items():
            prova.append(torch.tensor(v))
        # self.nn_inputs = torch.tensor(self.nn_inputs)
        # print(prova)
        self.nn_inputs = prova
        # print(self.nn_inputs)
        self.train_set = torch.load('imdb/train_simple.pt')
        self.test_set = torch.load('imdb/valid_simple.pt')

        self._compute_statistics()

        end_prep = time.time()
        print(f'Time for function prepare: {end_prep-start_prep}')

