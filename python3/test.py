"""
Created on Oct 28, 2015

@author: nyga
"""
import os

from pracmln import MLN, Database
from pracmln import query, learn
from pracmln.mlnlearn import EVIDENCE_PREDS
import time

from pracmln.utils import locs


def test_inference_smokers():
    p = os.path.join(locs.examples, 'smokers', 'smokers')
    # mln = MLN(mlnfile=('%s:wts.pybpll.smoking-train-smoking.mln' % p),
    #           grammar='StandardGrammar')
    # db = Database(mln, dbfile='%s:smoking-test-smaller.db' % p)
    print(p)
    mln = MLN(mlnfile=('%s.mln' % p),
              grammar='StandardGrammar')
    db = Database(mln, dbfile='%s.db' % p)
    for method in ('GibbsSampler',):
        print('=== INFERENCE TEST:', method, '===')
        query(queries='Cancer,Smokes,Friends',
              method=method,
              mln=mln,
              db=db,
              verbose=True,
              multicore=False).run()


def test_inference_taxonomies():
    p = os.path.join(locs.examples, 'taxonomies', 'taxonomies.pracmln')
    mln = MLN(mlnfile=('%s:wts.learned.taxonomy.mln' % p),
              grammar='PRACGrammar',
              logic='FuzzyLogic')
    db = Database(mln, dbfile='%s:evidence.db' % p)
    for method in ('EnumerationAsk', 'WCSPInference'):
        print('=== INFERENCE TEST:', method, '===')
        query(queries='has_sense, action_role',
              method=method,
              mln=mln,
              db=db,
              verbose=False,
              cw=True).run().write()
    
    
def test_learning_smokers():
    p = os.path.join(locs.examples, 'smokers', 'smokers.pracmln')
    mln = MLN(mlnfile=('%s:smoking.mln' % p), grammar='StandardGrammar')
    mln.write()
    db = Database(mln, dbfile='%s:smoking-train.db' % p)
    for method in ('BPLL', 'BPLL_CG', 'CLL'):
        for multicore in (True, False):
            print('=== LEARNING TEST:', method, '===')
            learn(method=method,
                  mln=mln,
                  db=db,
                  verbose=True,
                  multicore=multicore).run()


def test_learning_taxonomies():
    p = os.path.join(locs.examples, 'taxonomies', 'taxonomies.pracmln')
    mln = MLN(mlnfile=('%s:senses_and_roles.mln' % p), grammar='PRACGrammar')
    mln.write()
    dbs = Database.load(mln, dbfiles='%s:training.db' % p)
    for method in ('DPLL', 'DBPLL_CG', 'DCLL'):
        for multicore in (True, False):
            print('=== LEARNING TEST:', method, '===')
            learn(method=method,
                  mln=mln,
                  db=dbs,
                  verbose=True,
                  multicore=multicore,
                  epreds='is_a',
                  discr_preds=EVIDENCE_PREDS).run()


def test_GSMLN():
    # mln = MLN(grammar='GSMLNGrammar')
    # mln << 'residue(id, profile)'
    # mln << 'partners(id, id)'

    # f = "residue(a, $pa) v residue(b, $pb) => partners(a,b)"
    # # f = "((a(x) ^ b(x)) v (c(x) ^ !(d(x) ^ e(x) ^ g(x)))) => f(x)"
    # # f = "(a(x) v (b(x) ^ c(x))) => f(x)"
    # f = mln.logic.grammar.parse_formula(f)
    # f.print_structure()
    # print(list(f.literals()))

    # g = "partners(id, id)"
    # g = mln.logic.grammar.parse_predicate(g)
    # print(g)
    
    # print(mln.predicates)
    mln = MLN(mlnfile='beta_simple.mln', grammar='GSMLNGrammar')
    # mln.write()
    # print(mln.predicates)
    dbs = Database.load(mln, dbfiles='beta-train_simple.db')
    # dbs[0].write()
    print(mln.nnformulas[0].formula)
    print(mln.nnformulas[0].predicates)
    # print(mln.nnformulas[0].idx)

    # mln = MLN(grammar='GSMLNGrammar')
    # mln << 'Cancer(&person)'
    # mln << 'Friends(&person,&person)'
    # mln << 'Smokes(&person)'

    # f = 'Smokes($x) => Cancer($x)'
    # g = 'Friends($x,$y) => (Smokes($x) <=> Smokes($y))'
    # print(mln.logic.grammar.parse_formula(f))
    # mln.formula(f)
    # mln.formula(g)
    # print(mln.predicates)
    # print(mln.formulas)
    # mln.formulas[0].print_structure()
    # print(mln.domains)
    # print(mln.formulas[0].cnf())

    # this uses the method from base.py
    # mln.learn(databases=dbs, verbose=True, optimizer='ciao')

    # this uses the method from mlnlearn.py
    mln.gsmln_learn(method='GSMLN_L', mln=mln, databases=dbs, verbose=True)
    



def runall():
    start = time.time()
    test_inference_smokers()
    test_inference_taxonomies()
    test_learning_smokers()
    test_learning_taxonomies()
    print()
    print('all test finished after', time.time() - start, 'secs')

def main():
    test_GSMLN()
    

if __name__ == '__main__':
    main()
