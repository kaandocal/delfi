import numpy as np

from delfi.summarystats.BaseSummaryStats import SummaryStatsBase


class Identity(SummaryStatsBase):
    """Just apply the identity instead of reducing data.
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)

    @copy_ancestor_docstring
    def calc(self, repetition_list):
        # See BaseSummaryStats.py for docstring

        # get the number of samples contained
        n_reps = len(repetition_list)

        # get the size of the data inside a sample
        self.n_summary = repetition_list[0]['data'].size

        # build a matrix of n_reps x n_summary
        data_matrix = np.zeros((n_reps, self.n_summary))
        for rep_idx, rep_dict in enumerate(repetition_list):
            data_matrix[rep_idx, :] = rep_dict['data']

        return data_matrix
