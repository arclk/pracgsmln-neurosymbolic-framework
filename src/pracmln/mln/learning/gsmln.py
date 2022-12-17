from dnutils import ProgressBar

from .common import *
from ..grounding.default import DefaultGroundingFactory
from ..grounding.bpll import BPLLGroundingFactory
from ..constants import HARD
from ..errors import SatisfiabilityException
from ..util import fsum, temporary_evidence

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from torchviz import make_dot
import numpy as np

import copy
import time

class ABSTRCT_TypeNetwork(nn.Module):
    def __init__(self):
        super(ABSTRCT_TypeNetwork, self).__init__()
        self.fc1 = nn.Linear(25, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)
        
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()

    def forward(self, input):
        output = self.activation(self.fc1(input))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        output = self.fc4(output)

        return output


class ABSTRCT_LinkNetwork(nn.Module):
    def __init__(self):
        super(ABSTRCT_LinkNetwork, self).__init__()
        self.fc1 = nn.Linear(50, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)
        
        self.dropout = nn.Dropout(0.4)
        self.activation = nn.ReLU()

    def forward(self, input):
        output = self.activation(self.fc1(input))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        output = self.fc4(output)

        return output


class GSMLN_L(AbstractLearner):
    pass

class GSMLN_MNIST(AbstractLearner):

    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None
        self._varidx2fidx = None
        self._lastw = None
        self.ftype = ['nn', 'nn', 'hard', 'hard']
        self.wt = []
        self.nn = []
        self.nn_inputs = {}
        self.gidx_mat = None
        self.evidence_mask = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._weights = None
        self.device = params.get('device')
        print(self.device)
        # self.network = Network()


    def _pl(self, varidx):
        '''
        Computes the pseudo-likelihoods for the given variable under weights w. 
        '''

        values = 2
        var = self.var_dict[varidx]
        name, wt_idx1, wt_idx2, wt_idx3 = var

        # if the predicate is a Feature Predicate return the tensor [0, 1]
        if name == 'Digit':
            return torch.tensor([0, 1], dtype=torch.float)

        gfs = self._varidx2fidx.get(varidx)
        if gfs is None: 
            # no list was saved, so the truth of all formulas is unaffected by the variable's value
            # uniform distribution applies
            p = 1.0 / values
            return p * torch.ones(values, device=self.device)
        sums = torch.zeros(values, device=self.device)#numpy.zeros(values)
        for fidx, groundings in gfs.items():
            for gidx in groundings:
                for validx, n in enumerate(self._stat[fidx][gidx][varidx]):
                    if self.ftype[fidx] == 'hard': 
                        # penalize the prob mass of every value violating a hard constraint
                        if n == 0: 
                            sums[validx] = sums[validx] - 1000 * self.wt[fidx][wt_idx1][wt_idx2]
                    else:
                        sums[validx] = sums[validx] + n * self.wt[fidx][wt_idx1][wt_idx2]

        return sums


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self.var_pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls2(self):
        self._pls = torch.zeros((len(self.mrf.variables),2), device=self.device)
        for var in self.mrf.variables:
            self._pls[var.idx] = self._pl(var.idx)

        self._lastw = list(self.wt)


    def forward(self):
        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(nn(self.nn_inputs[fidx]))


    def _compute_pls(self):
        '''
        Computes the pseudo-likelihoods for all the variables based on the
        weights wt which constitutes the outputs of the neural networks
        '''
        self._pls = []
        self._pls.append(torch.zeros((self.numbervar,2)))
        for varidx in self.number_dict:
            self._pls[0][varidx] = self._pl(varidx+self.digitvar)


    def grad(self, w):
        '''
        Computes the gradient taking into consideration the pseudo-likelihoods
        '''
        self._compute_pls()
        grad = torch.zeros(len(self.mrf.nnformulas), dtype=torch.float64)
        for fidx, groundval in self._stat.items():
            if fidx > 1:
                break
            for gidx, varval in groundval.items():
                for varidx, counts in varval.items():
                    var = self.var_dict[varidx]
                    name, _, _, evidx = var
                    g = counts[evidx]
                    if name == 'Digit':
                        continue
                    if name == 'Number':
                        plsidx = 0
                        varidx -= self.digitvar
                    for i, val in enumerate(counts):
                        g -= val * self._pls[plsidx][varidx][i]
                    grad[fidx] += g
        
        return grad


    def forward(self):
        '''
        Computes the forward step of the nural networks
        '''
        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(nn(self.nn_inputs[fidx]))

        return self.wt


    def train(self, grad_mod):
        '''
        Computes an epoch of the full training step
        '''
        for model in self.mrf.nnformulas:
            model.train()

        self.optimizer.zero_grad()
        preds = self.forward()
        # print(preds[1])

        loss = criterion[0](preds[0], y_true)

        loss.backward()
        if (grad_mod):
            gradient = self.grad(preds)
            for fidx, nn in enumerate(self.mrf.nnformulas):
                for par in nn.parameters():
                    par.grad *= gradient[fidx]

        self.optimizer.step()
        return loss


    def evaluate(self):
        '''
        Evaluate the model
        '''
        for model in self.mrf.nnformulas:
            model.eval()

        pred = []
        with torch.no_grad():
            for fidx, nn in enumerate(self.mrf.nnformulas):
                pred.append(nn(self.val_inputs[fidx]))
            
            y = pred[0]
            y_pred = y.argmax(dim=1)
            y_pred = y_pred.to('cpu')

            acc = accuracy_score(self.val_true, y_pred)

            return acc


    def training_loop(self, epochs=500, pretrain=None,
        early_stopping=True, early_stopping_epochs=1000, verbose=True, **params):
        '''
        Computes the training algorithm with all the epochs and evaluate the model
        after each training step. It is possible to pretrain the model for a number
        of epochs given by the pretrain param.
        '''
        start_train = time.time()
        best_val_f1 = 0
        best_epoch = 0
        epochs_no_improve = 0
        # best_params = copy.deepcopy(nnformulas.state_dict())
        grad_mod = False

        for epoch in range(epochs):
            start_epoch = time.time()
            
            if (pretrain):
                if (epoch > pretrain):
                    grad_mod = True
            else:
                grad_mod = True

            loss_train = self.train(grad_mod)
            train_time = time.time()

            val_f1_link, val_f1_type = self.evaluate()

            end_epoch = time.time()

            # Early stopping
            if val_f1_link > best_val_f1:
                epochs_no_improve = 0
                best_val_f1 = val_f1_link
                # best_params = copy.deepcopy(nnformulas.state_dict())
                best_epoch = epoch
            else: 
                epochs_no_improve += 1
            
            if early_stopping and epochs_no_improve == early_stopping_epochs:
                if verbose:
                    print('Early stopping!' )
                break

            if verbose and (epoch+1)%1 == 0:
                print(f'Epoch: {epoch+1} '
                        f' Loss: Train = [{loss_train:.4f}] '
                        f' F1: Val_Link = [{val_f1_link:.4f}] Val_Type = [{val_f1_type:.4f}] '
                        f' Time one epoch (s): {end_epoch-start_epoch:.4f} ')

        end_train= time.time()
        print(f"Best epoch {best_epoch+1}, F1_macro: {best_val_f1:.4f}")
        print(f'Time for training: {end_train-start_train}')

        return best_val_f1, best_epoch#, best_params


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
        print('Inizio comp stat')
        for f in grounder.itergroundings():
            # AA: to take in consideration the various groundings but have to be improved
            if f.idx == old_idx:
                gidx += 1
            else:
                gidx = 0
            old_idx = f.idx 
            # print(f.atomic_constituents())
            for gndatom in f.gndatoms():
                var = self.mrf.variable(gndatom)
                # Save the evidence value for that variable
                evidence_value = var.evidence_value()
                for validx, value in var.itervalues():
                    # print(value)
                    var.setval(value, self.mrf.evidence)
                    truth = f(self.mrf.evidence) 
                    if truth != 0:
                        # self._varidx2fidx[var.idx].add(f.idx)
                        self._addvaridx2fidx(f.idx, gidx, var.idx)
                        self._addstat(f.idx, gidx, var.idx, validx, truth)
                # Restore the original evidence value 
                var.setval(evidence_value, self.mrf.evidence)
        end_comp = time.time()
        print(f'Time for function comp_stat: {end_comp-start_comp}')


    def _prepare(self):
        start_prep = time.time()

        self.train_set = torch.load('mnist/neoplasm25_train.pt')
        self.type_true = torch.load('mnist/neo_type_true_simple.pt')
        self.link_true = torch.load('mnist/neo_link_true_simple.pt')

        self.val_inputs = torch.load('mnist/neoplasm25_val_inputs.pt')
        self.type_val_true = torch.load('mnist/neo_type_val_true.pt')
        self.link_val_true = torch.load('mnist/neo_link_val_true.pt')
        
        self.mrf.nnformulas = torch.nn.ModuleList()
        self.mrf.nnformulas.append(ABSTRCT_TypeNetwork())
        self.mrf.nnformulas.append(ABSTRCT_LinkNetwork())

        self.optimizer = torch.optim.Adam(self.mrf.nnformulas.parameters(), lr=0.001)
        self.criterion = [nn.CrossEntropyLoss()]

        print(self.mrf.nnformulas)
        grounder = DefaultGroundingFactory(self.mrf)
        
        self.gidx_mat = torch.zeros((2267,2267), dtype=int)

        print(self.mrf.feat_dict)

        
        
        # Create the function nn inputs
        for gidx, gf in enumerate(grounder.itergroundings()):
            if gf.idx not in self.nn_inputs:
                self.nn_inputs[gf.idx] = []
            # print(gf.atomic_constituents())
            temp = []
            gf_temp = []
            for atom in gf.atomic_constituents():
                if atom.predname in self.mrf.feat_dict.keys():
                    feat_idx = self.mrf.feat_dict[atom.predname][atom.args[1]]
                    temp += self.train_set[feat_idx[0]]
                    
                    if (gf.idx == 1):
                        gf_temp.append(int(atom.args[1][1:]))
            self.nn_inputs[gf.idx].append(list(map(float,temp)))
            if (gf.idx == 1):
                # print(atom, gf_temp[0], gf_temp[1])
                self.gidx_mat[gf_temp[0], gf_temp[1]] = len(self.nn_inputs[gf.idx])
            # else:
            #     print(atom)

        prova = []
        for k, v in self.nn_inputs.items():
            prova.append(torch.tensor(v))
        # prova[0] = prova[0][range(0,len(prova[0]),2)]
        self.nn_inputs = prova
        
        self._compute_statistics()

        self.var_dict = {}
        self.numbervar = 0
        self.digitvar = 0
        for var in self.mrf.variables:
            name = var.predicate.name

            if name == 'Number':
                wt_idx1 = int(var.gndatoms[0].args[0][2:])
                wt_idx2 = int(var.gndatoms[0].args[1][2:])
                self.numbervar += 1

            if name == 'Digit':
                wt_idx1 = 0
                wt_idx2 = 0
                self.digitvar += 1

            self.var_dict[var.idx] = [var.predicate.name, wt_idx1, wt_idx2]

        self.number_dict = {}
        for i in range(self.digitvar, self.digitvar+self.numbervar):
            self.number_dict[i-self.digitvar] = self.var_dict[i]

        end_prep = time.time()
        print(f'Time for function prepare: {end_prep-start_prep}')
        print(self.nn_inputs)
        print(self.nn_inputs[0].size())
        print(self.nn_inputs[1].size())
        print(self.nn_inputs[2].size())
        print(self.nn_inputs[3].size())


    def run(self, **params):
        '''
        Learn the weights of the MLN given the training data previously 
        loaded 
        '''
        self._prepare()
        print(params)
        self.training_loop(**params)


class GSMLN_ABSTRCT(AbstractLearner):

    
    def __init__(self, mrf, **params):
        AbstractLearner.__init__(self, mrf, **params)
        self._pls = None
        self._stat = None
        self._varidx2fidx = None
        self._lastw = None
        self.ftype = ['nn', 'nn', 'hard', 'hard']
        self.wt = []
        self.nn = []
        self.nn_inputs = {}
        self.gidx_mat = None
        self.evidence_mask = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._weights = None
        self.device = params.get('device')
        print(self.device)
        # self.network = Network()


    def _pl(self, varidx):
        '''
        Computes the pseudo-likelihoods for the given variable under weights w. 
        '''

        values = 2
        var = self.var_dict[varidx]
        name, wt_idx1, wt_idx2, wt_idx3 = var

        # if the predicate is a Feature Predicate return the tensor [0, 1]
        if name == 'Text':
            return torch.tensor([0, 1], dtype=torch.float)

        if name == 'Link':        
            wt_idx = self.gidx_mat[wt_idx1][wt_idx2]-1

        gfs = self._varidx2fidx.get(varidx)
        if gfs is None: 
            # no list was saved, so the truth of all formulas is unaffected by the variable's value
            # uniform distribution applies
            p = 1.0 / values
            return p * torch.ones(values, device=self.device)
        sums = torch.zeros(values, device=self.device)#numpy.zeros(values)
        for fidx, groundings in gfs.items():
            for gidx in groundings:
                for validx, n in enumerate(self._stat[fidx][gidx][varidx]):
                    if self.ftype[fidx] == 'hard': 
                        # penalize the prob mass of every value violating a hard constraint
                        if n == 0: 
                            if fidx == 0:
                                sums[validx] = sums[validx] - 1000 * self.wt[fidx][wt_idx1][wt_idx3]
                            if fidx == 1:
                                sums[validx] = sums[validx] - 1000 * self.wt[fidx][wt_idx][wt_idx3]
                    else:
                        if fidx == 0:
                            sums[validx] = sums[validx] + n * self.wt[fidx][wt_idx1][wt_idx3]
                        if fidx == 1:
                            sums[validx] = sums[validx] + n * self.wt[fidx][wt_idx][wt_idx3]

        return sums


    def write_pls(self):
        for var in self.mrf.variables:
            print(repr(var))
            for i, value in var.itervalues():
                print('    ', barstr(width=50, color='magenta', percent=self.var_pls[var.idx][i]) + ('*' if var.evidence_value_index() == i else ' '), i, value)


    def _compute_pls2(self):
        self._pls = torch.zeros((len(self.mrf.variables),2), device=self.device)
        for var in self.mrf.variables:
            self._pls[var.idx] = self._pl(var.idx)

        self._lastw = list(self.wt)


    def forward(self):
        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(nn(self.nn_inputs[fidx]))


    def _compute_pls(self):
        '''
        Computes the pseudo-likelihoods for all the variables based on the
        weights wt which constitutes the outputs of the neural networks
        '''
        self._pls = []
        self._pls.append(torch.zeros((self.typevar,2)))
        self._pls.append(torch.zeros((self.linkvar,2)))
        for varidx in self.type_dict:
            self._pls[0][varidx] = self._pl(varidx+self.linkvar+self.textvar)
        for varidx in self.link_dict:
            self._pls[1][varidx] = self._pl(varidx)


    def grad(self, w):
        '''
        Computes the gradient taking into consideration the pseudo-likelihoods
        '''
        self._compute_pls()
        grad = torch.zeros(len(self.mrf.nnformulas), dtype=torch.float64)
        for fidx, groundval in self._stat.items():
            if fidx > 1:
                break
            for gidx, varval in groundval.items():
                for varidx, counts in varval.items():
                    var = self.var_dict[varidx]
                    name, _, _, evidx = var
                    g = counts[evidx]
                    if name == 'Text':
                        continue
                    if name == 'Type':
                        plsidx = 0
                        varidx -= (self.linkvar+self.textvar)
                    if name == 'Link':
                        plsidx = 1
                    for i, val in enumerate(counts):
                        g -= val * self._pls[plsidx][varidx][i]
                    grad[fidx] += g
        
        return grad


    def forward(self):
        '''
        Computes the forward step of the nural networks
        '''
        self.wt = []
        for fidx, nn in enumerate(self.mrf.nnformulas):
            self.wt.append(nn(self.nn_inputs[fidx]))

        return self.wt


    def train(self, grad_mod):
        '''
        Computes an epoch of the full training step
        '''
        for model in self.mrf.nnformulas:
            model.train()

        self.optimizer.zero_grad()
        preds = self.forward()
        # print(preds[1])

        loss1 = self.criterion[0](preds[0], self.type_true)
        loss2 = self.criterion[1](preds[1], self.link_true)
        loss = loss1+loss2

        loss.backward()
        if (grad_mod):
            gradient = self.grad(preds)
            for fidx, nn in enumerate(self.mrf.nnformulas):
                for par in nn.parameters():
                    par.grad *= gradient[fidx]

        self.optimizer.step()
        return loss


    def evaluate(self):
        '''
        Evaluate the model
        '''
        for model in self.mrf.nnformulas:
            model.eval()

        pred = []
        with torch.no_grad():
            for fidx, nn in enumerate(self.mrf.nnformulas):
                pred.append(nn(self.val_inputs[fidx]))
            y_link = pred[1]
            y_link_pred = y_link.argmax(dim=1)
            f1_link = f1_score(self.link_val_true, y_link_pred, average='macro', labels=[1])

            y_type = pred[0]
            y_type_pred = y_type.argmax(dim=1)
            f1_type = f1_score(self.type_val_true, y_type_pred, average='macro')

            return f1_link, f1_type


    def training_loop(self, epochs=500, pretrain=None,
        early_stopping=True, early_stopping_epochs=1000, verbose=True, **params):
        '''
        Computes the training algorithm with all the epochs and evaluate the model
        after each training step. It is possible to pretrain the model for a number
        of epochs given by the pretrain param.
        '''
        start_train = time.time()
        best_val_f1 = 0
        best_epoch = 0
        epochs_no_improve = 0
        # best_params = copy.deepcopy(nnformulas.state_dict())
        grad_mod = False

        for epoch in range(epochs):
            start_epoch = time.time()
            
            if (pretrain):
                if (epoch > pretrain):
                    grad_mod = True
            else:
                grad_mod = True

            loss_train = self.train(grad_mod)
            train_time = time.time()

            val_f1_link, val_f1_type = self.evaluate()

            end_epoch = time.time()

            # Early stopping
            if val_f1_link > best_val_f1:
                epochs_no_improve = 0
                best_val_f1 = val_f1_link
                # best_params = copy.deepcopy(nnformulas.state_dict())
                best_epoch = epoch
            else: 
                epochs_no_improve += 1
            
            if early_stopping and epochs_no_improve == early_stopping_epochs:
                if verbose:
                    print('Early stopping!' )
                break

            if verbose and (epoch+1)%1 == 0:
                print(f'Epoch: {epoch+1} '
                        f' Loss: Train = [{loss_train:.4f}] '
                        f' F1: Val_Link = [{val_f1_link:.4f}] Val_Type = [{val_f1_type:.4f}] '
                        f' Time one epoch (s): {end_epoch-start_epoch:.4f} ')

        end_train= time.time()
        print(f"Best epoch {best_epoch+1}, F1_macro: {best_val_f1:.4f}")
        print(f'Time for training: {end_train-start_train}')

        return best_val_f1, best_epoch#, best_params


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
        print('Inizio comp stat')
        for f in grounder.itergroundings():
            # AA: to take in consideration the various groundings but have to be improved
            if f.idx == old_idx:
                gidx += 1
            else:
                gidx = 0
            old_idx = f.idx 
            # print(f.atomic_constituents())
            for gndatom in f.gndatoms():
                var = self.mrf.variable(gndatom)
                # Save the evidence value for that variable
                evidence_value = var.evidence_value()
                for validx, value in var.itervalues():
                    # print(value)
                    var.setval(value, self.mrf.evidence)
                    truth = f(self.mrf.evidence) 
                    if truth != 0:
                        # self._varidx2fidx[var.idx].add(f.idx)
                        self._addvaridx2fidx(f.idx, gidx, var.idx)
                        self._addstat(f.idx, gidx, var.idx, validx, truth)
                # Restore the original evidence value 
                var.setval(evidence_value, self.mrf.evidence)
        end_comp = time.time()
        print(f'Time for function comp_stat: {end_comp-start_comp}')


    def _prepare(self):
        start_prep = time.time()

        self.train_set = torch.load('abstrct/neoplasm25_train.pt')
        self.type_true = torch.load('abstrct/neo_type_true_simple.pt')
        self.link_true = torch.load('abstrct/neo_link_true_simple.pt')

        self.val_inputs = torch.load('abstrct/neoplasm25_val_inputs.pt')
        self.type_val_true = torch.load('abstrct/neo_type_val_true.pt')
        self.link_val_true = torch.load('abstrct/neo_link_val_true.pt')
        
        self.mrf.nnformulas = torch.nn.ModuleList()
        self.mrf.nnformulas.append(ABSTRCT_TypeNetwork())
        self.mrf.nnformulas.append(ABSTRCT_LinkNetwork())

        self.optimizer = torch.optim.Adam(self.mrf.nnformulas.parameters(), lr=0.001)
        self.criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))]

        print(self.mrf.nnformulas)
        grounder = DefaultGroundingFactory(self.mrf)
        
        self.gidx_mat = torch.zeros((2267,2267), dtype=int)

        print(self.mrf.feat_dict)

        
        
        # Create the function nn inputs
        for gidx, gf in enumerate(grounder.itergroundings()):
            if gf.idx not in self.nn_inputs:
                self.nn_inputs[gf.idx] = []
            # print(gf.atomic_constituents())
            temp = []
            gf_temp = []
            for atom in gf.atomic_constituents():
                if atom.predname in self.mrf.feat_dict.keys():
                    feat_idx = self.mrf.feat_dict[atom.predname][atom.args[1]]
                    temp += self.train_set[feat_idx[0]]
                    
                    if (gf.idx == 1):
                        gf_temp.append(int(atom.args[1][1:]))
            self.nn_inputs[gf.idx].append(list(map(float,temp)))
            if (gf.idx == 1):
                # print(atom, gf_temp[0], gf_temp[1])
                self.gidx_mat[gf_temp[0], gf_temp[1]] = len(self.nn_inputs[gf.idx])
            # else:
            #     print(atom)

        prova = []
        for k, v in self.nn_inputs.items():
            prova.append(torch.tensor(v))
        prova[0] = prova[0][range(0,len(prova[0]),2)]
        self.nn_inputs = prova
        
        self._compute_statistics()

        self.var_dict = {}
        self.typevar = 0
        self.linkvar = 0
        self.textvar = 0
        for var in self.mrf.variables:
            name = var.predicate.name

            if name == 'Type':
                wt_idx1 = int(var.gndatoms[0].args[0][2:])
                wt_idx2 = 0 if 'Claim' in var.name else 1
                self.typevar += 1

            if name == 'Link':
                wt_idx1 = int(var.gndatoms[0].args[0][2:])
                wt_idx2 = int(var.gndatoms[0].args[1][2:])
                self.linkvar += 1

            if name == 'Text':
                wt_idx1 = 0
                wt_idx2 = 0
                self.textvar += 1

            self.var_dict[var.idx] = [var.predicate.name, wt_idx1, wt_idx2, var.evidence_value_index()]
        print(self.var_dict)
        self.link_dict = {}
        for i in range(self.linkvar):
            self.link_dict[i] = self.var_dict[i]

        self.type_dict = {}
        for i in range(self.linkvar+self.textvar, self.linkvar+self.textvar+self.typevar):
            self.type_dict[i-(self.linkvar+self.textvar)] = self.var_dict[i]

        end_prep = time.time()
        print(f'Time for function prepare: {end_prep-start_prep}')
        print(self.nn_inputs)
        print(self.nn_inputs[0].size())
        print(self.nn_inputs[1].size())
        print(self.nn_inputs[2].size())
        print(self.nn_inputs[3].size())


    def run(self, **params):
        '''
        Learn the weights of the MLN given the training data previously 
        loaded 
        '''
        self._prepare()
        print(params)
        self.training_loop(**params)
