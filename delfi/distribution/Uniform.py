import numpy as np

from delfi.distribution.BaseDistribution import DistributionBase


class Uniform(DistributionBase):
    def __init__(self, lower=0., upper=1., seed=None):
        """Uniform distribution

        Parameters
        ----------
        lower : list, or np.array, 1d
            Lower bound(s)
        upper : list, or np.array, 1d
            Upper bound(s)
        seed : int or None
            If provided, random number generator will be seeded
        """
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)

        assert self.lower.ndim == self.upper.ndim
        assert self.lower.ndim == 1

        super().__init__(ndim=len(self.lower), seed=seed)

    @property
    def mean(self):
        """Means"""
        return (0.5 * (self.lower + self.upper)).reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt(1/12. * (self.upper - self.lower)**2).reshape(-1)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseDistribution.py for docstring
        if ii is not None:
            raise NotImplementedError

        N = np.asarray(x).shape[0]

        p = 1/np.prod(self.upper - self.lower)
        p = p*np.ones((N,))  # broadcasting

        if log:
            return np.log(p)
        else:
            return p

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseDistribution.py for docstring
        ms = self.rng.rand(n_samples, self.ndim) * (self.upper - self.lower) + self.lower
        return ms
