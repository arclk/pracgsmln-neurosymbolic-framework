'''
Created on Jan 5, 2017

@author: nyga
'''
import time
from math import sqrt

from dnutils import out
from .threads import Lock
from tabulate import tabulate


def _matshape(m):
    '''Returns the shape, ie a tuple #rows,#cols of a matrix'''
    assert all([len(row) == len(m[0]) for row in m])
    return len(m), len(m[0])


class Gaussian(object):
    '''
    A Gaussian distribution that can be incrementally updated with new samples
    '''

    def __init__(self, mean=None, cov=None, data=None, keepsamples=False):
        '''
        Creates a new Gaussian distribution.
        :param mean:    the mean of the Gaussian. May be a scalar (univariante) or an array (multivariate).
        :param cov:     the covariance of the Gaussian. May be a scalar (univariate) or a matrix (multivariate).
        :param data:    if ``mean`` and ``cov`` are not provided, ``data`` may be a data set (matrix) from which
                        the parameters of the distribution are estimated.
        '''
        self.mean = mean
        self.cov = cov
        self.samples = 0 if not keepsamples else []
        if data is not None:
            self.estimate(data)

    @property
    def numsamples(self):
        return len(self.samples) if type(self.samples) is list else self.samples

    @property
    def mean(self):
        if self._mean is not None and len(self._mean) == 1:
            return self._mean[0]
        else:
            return self._mean

    @mean.setter
    def mean(self, mu):
        if mu is not None and not hasattr(mu, '__len__'):
            self._mean = [mu]
        else:
            self._mean = mu

    @property
    def cov(self):
        if self._cov is not None and _matshape(self._cov) == (1, 1):
            return self._cov[0][0]
        else:
            return self._cov

    @cov.setter
    def cov(self, cov):
        if cov is not None and not hasattr(cov, '__len__'):
            self._cov = [[cov]]
        else:
            self._cov = cov

    @property
    def dim(self):
        if self._mean is None:
            raise ValueError('no dimensionality specified yet.')
        return len(self._mean)

    def update(self, x):
        '''update the Gaussian distribution with a new data point `x`.'''
        if not hasattr(x, '__len__'):
            x = [x]
        if self._mean is None or self._cov is None:
            self._mean = [0] * len(x)
            self._cov = [[0] * len(x) for _ in range(len(x))]
        else:
            assert len(x) == len(self._mean) and _matshape(self._cov) == (len(x), len(x))
        n = self.numsamples
        oldmean = list(self._mean)
        oldcov = list([list(row) for row in self._cov])
        for i, (m, d) in enumerate(zip(self._mean, x)):
            self._mean[i] = ((n * m) + d) / (n + 1)
        if type(self.samples) is list:
            self.samples.append(x)
        else:
            self.samples += 1
        if n:
            for j in range(self.dim):
                for k in range(self.dim):
                    self._cov[j][k] = (oldcov[j][k] * (n - 1) + n * oldmean[j] * oldmean[k] + x[j] * x[k] - (n + 1) * self._mean[j] * self._mean[k]) / float(n)

    def update_all(self, data):
        '''Update the distribution with new data points given in `data`.'''
        for x in data:
            self.update(x)
        return self

    def estimate(self, data):
        '''Estimate the distribution parameters with subject to the given data points.'''
        self.mean = self.cov = None
        return self.update_all(data)

    def sample(self, n=1):
        '''Return `n` samples from the distribution subject to the parameters.
        .. warning::
            This method requires the ``numpy`` package installed.'''
        import numpy as np
        if self.mean is None or self.cov is None:
            raise ValueError('no parameters. You have to set mean and covariance before you draw samples.')
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    @property
    def var(self):
        if self._cov is None: return None
        return [self._cov[i][i] for i in range(self.dim)]

    def reset(self):
        self.samples = []
        self.mean = None
        self.cov = None

    def __repr__(self):
        try:
            dim = '%s-dim' % str(self.dim)
            if self.dim == 1:
                dim = 'mu=%.2f, var=%.2f' % (self.mean, self.cov)
        except ValueError:
            dim = '(undefined)'
        return '<Gaussian %s at 0x%s>' % (dim, hex(id(self)))

    def __str__(self):
        try:
            if self.dim > 1:
                args = '\nmean=\n%s\ncov=\n%s' % (self.mean, tabulate(self.cov))
            else:
                args = 'mean=%.2f, var=%.2f' % (self.mean, self.cov)
        except ValueError:
            args = 'undefined'
        return '<Gaussian %s>' % args

    def kldiv(self, g2):
        '''
        Compute the KL-divergence of two multivariate Gaussian distributions.

        :param g1: instance of ``dnutils.Gaussian``
        :param g2: instance of ``dnutils.Gaussian``
        :return:
        '''
        import numpy as np
        mu1 = np.array(self._mean)
        mu2 = np.array(g2._mean)
        sigma1 = np.array(self._cov)
        sigma2 = np.array(g2._cov)
        det1 = np.linalg.det(sigma1)
        det2 = np.linalg.det(sigma2)
        if all([d < 1e-12 for d in (det1, det2)]):
            if tuple(mu1) == tuple(mu2):
                return 0
            else:
                return float('inf')
        elif any([d < 1e-12 for d in (det1, det2)]):
            return float('inf')
        res = np.log(det2 / det1) - self.dim + np.matrix.trace(np.linalg.inv(sigma2).dot(sigma1)) + (
            (mu2 - mu1).dot(np.linalg.inv(sigma2).dot((mu2 - mu1))))
        return res * .5


class Timespan:
    '''
    A ``Timespan`` implements the context manager protocol
    of a stopwatch
    '''

    def __init__(self, watch):
        self.watch = watch
        self.start = None
        self.stop = None
        self.events = {} # maps timestamp -> string

    @property
    def duration(self):
        return self.stop - self.start

    def event(self, description):
        self.event[time.time()] = description

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.stop = time.time()
        self.watch.add(self)


_watches = {}
_watchlock = Lock()


def stopwatch(name):
    '''
    Returns a ``Timespan`` object representing a slice of the
    execution.

    :param name:
    :return:
    '''
    with _watchlock:
        if name not in _watches:
            _watches[name] = StopWatch(name)
        watch = _watches[name]
    return Timespan(watch)


def stopwatches():
    yield from _watches.values()


def get_stopwatch(name):
    '''
    Returns the stop watch with the given name, or ``None``, fif no
    such stop watch exists.

    :param name:
    :return:
    '''
    return _watches.get(name)


def print_stopwatches():
    '''
    Prints to the console a tabular of all stop watches that have been
    recoreded so far.
    :return:
    '''
    headers = ('name', 'avg', 'std', 'calls')
    data = [[w.tojson()[h] for h in headers] for w in _watches.values()]
    print(tabulate(data, headers))


def reset_stopwatches():
    '''
    Delete all stopwatches so far.
    '''
    global _watches
    _watches = {}


class StopWatch:
    '''
    Easy-to-use and lightweight stop watch to measure execution time
    of code passages. Supports Python's the context manager protocol.

    A stop watch is characterized by its name that the user can choose freely,
    and a Gaussian distribution that is maintained collecting a sufficient
    statistics about execution times.

    Stop watches are thread-safe.
    '''

    def __init__(self, name):
        self.name = name
        self.dist = Gaussian()
        self._lock = Lock()

    def add(self, ts):
        with self._lock:
            self.dist.update(ts.duration)

    @property
    def avg(self):
        return self.dist.mean

    @property
    def std(self):
        return sqrt(self.dist.cov)

    @property
    def calls(self):
        return self.dist.numsamples

    def tojson(self):
        return {'name': self.name, 'avg': self.avg, 'std': self.std, 'calls': self.calls}

