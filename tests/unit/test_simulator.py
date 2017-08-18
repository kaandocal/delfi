import numpy as np

from delfi.simulator.Gauss import Gauss


def test_gauss_1d_simulator_output():
    """Test the output of the simulator using the example of a 1D Gaussian
    """
    dim = 1
    s = Gauss(dim=dim)

    n_samples = 10
    thetas = np.tile(np.array([0.]), (n_samples, 1))
    sample_list = s.gen(thetas)

    assert len(sample_list) == n_samples, 'the list should have as many entries as there are samples'
    assert isinstance(sample_list[0][0], dict), 'the entries should be dictionaries'

def test_gauss_2d_data_dimension():
    """Test the data dimensionality output of the Gauss Simulator using a 2D Gaussian
    """
    dim = 2
    s = Gauss(dim=dim)

    n_samples = 10
    thetas = np.tile(np.array([0., 1.]), (n_samples, 1))
    sample_list = s.gen(thetas)

    assert sample_list[0][0]['data'].shape == (dim, ), \
        'the dimensionality of the data is wrong. ' \
        'should be {} is {}'.format((dim, 1), sample_list[0][0]['data'].shape)
