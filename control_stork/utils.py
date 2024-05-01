import gzip
import pickle
import torch
import string
import random
import time
import numpy as np
from typing import Union


def get_random_string(string_length: int = 5) -> str:
    """Generates a random string of fixed length.

    Args:
        string_length (int): the length of the random string

    Returns:
        random string (str)
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))


def get_basepath(dir: str = ".", prefix: str = "default", salt_length: int = 5) -> str:
    """Returns pre-formatted and time stamped basepath given a base directory and file prefix.

    Args:
        dir (str): the base directory
        prefix (str): the file prefix
        salt_length (int): the length of the random salt to append to the filename

    Returns:
        basepath (str): the basepath

    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if salt_length:
        salt = get_random_string(salt_length)
        basepath = "%s/%s-%s-%s" % (dir, prefix, timestr, salt)
    else:
        basepath = "%s/%s-%s" % (dir, prefix, timestr)
    return basepath


def write_to_file(data: object, filename: str) -> None:
    """Writes an object/dataset to zipped pickle.

    Args:
        data: the (data) object
        filename (str): the filename to write to
    """
    fp = gzip.open("%s" % filename, "wb")
    pickle.dump(data, fp)
    fp.close()


def load_from_file(filename: str) -> object:
    """Loads an object/dataset from a zipped pickle.

    Args:
        filename (str): the filename to load from

    Returns:
        data (object): the loaded object

    """
    fp = gzip.open("%s" % filename, "r")
    data = pickle.load(fp)
    fp.close()
    return data


def to_sparse(x: torch.Tensor) -> torch.Tensor:
    """converts dense tensor x to sparse format.

    Args:
        x (torch.Tensor): dense tensor

    Returns:
        sparse tensor (torch.sparse.FloatTensor)

    """

    indices = torch.nonzero(x)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse.FloatTensor(indices, values, x.size(), device=x.device)


def get_lif_kernel(
    tau_mem: Union[float, torch.Tensor] = 20e-3,
    tau_syn: Union[float, torch.Tensor] = 10e-3,
    dt: float = 1e-3,
) -> torch.Tensor:
    """Computes the linear filter kernel of a simple LIF neuron with exponential current-based synapses.

    Args:
        tau_mem: The membrane time constant(s)
        tau_syn: The synaptic time constant(s)
        dt: The time_step size

    Returns:
        Tensor of length 10x of the longest time constant containing the filter kernel (dim, t)

    """
    if not torch.is_tensor(tau_mem):
        tau_mem = torch.Tensor([tau_mem])
    if not torch.is_tensor(tau_syn):
        tau_syn = torch.Tensor([tau_syn])
    mem_max = torch.max(tau_mem)
    syn_max = torch.max(tau_syn)
    tau_max = torch.max(torch.tensor([mem_max, syn_max]))
    T_max = int(tau_max * 10 / dt)
    ts = np.arange(0, T_max + 1) * dt
    n = len(ts)
    kernel = []
    I = 1.0  # Initialize current variable for single spike input
    U = 0.0
    dcy1 = torch.exp(-dt / tau_mem)
    dcy2 = torch.exp(-dt / tau_syn)
    for i, t in enumerate(ts):
        U = dcy1 * U + (1.0 - dcy1) * I
        I *= dcy2
        kernel.append(U)
    k = torch.stack(kernel, axis=-1)
    return k


def convlayer_size(
    nb_inputs: int, kernel_size: int, padding: int, stride: int
) -> np.array:
    """
    Calculates output size of convolutional layer

    Args:
        nb_inputs: number of input channels
        kernel_size: size of kernel
        padding: padding
        stride: stride

    Returns:
        output size of convolutional layer
    """
    res = ((np.array(nb_inputs) - kernel_size + 2 * padding) / stride) + 1
    return res
