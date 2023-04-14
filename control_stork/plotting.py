import torch
import math
import matplotlib.pyplot as plt
from .extratypes import *


def plot_spikes(
    spk: torch.Tensor,
    dim: Optional[tuple] = (3, 5),
    title: str = "",
    savefig: str = "",
    show: bool = False,
    retrn: bool = True,
) -> None:
    """Plots a grid of spike raster plots.
    Args:
        spk: A tensor of shape (batch_size, n_steps, n_neurons) containing the spike times.
        dim: A tuple of (n_rows, n_cols) for the grid.
        title: A string for the title of the plot.
        savefig: A string for the path to save the figure.
        show: A boolean to show the figure.
        retrn: A boolean to return the figure.

    Returns:
        None
    """

    n_examples = spk.shape[0]

    if dim is None:
        n_rows = int(math.ceil(math.sqrt(n_examples)))
        n_cols = int(math.ceil(n_examples / n_rows))
    else:
        n_rows = min(n_examples, dim[0])
        n_cols = min(math.ceil(n_examples / n_rows), dim[1])
    dim = (n_rows, n_cols)

    fig = plt.figure(figsize=(10, 4))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    if title:
        fig.suptitle(title)
    N = min(math.prod(dim), n_examples)
    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        if i == 0:
            a0 = ax = plt.subplot(gs[row, col])
        else:
            ax = plt.subplot(gs[row, col], sharey=a0)
        ax.imshow(spk[i].T, cmap=plt.cm.gray_r, aspect="auto", interpolation="none")
        ax.spines[["right", "top"]].set_visible(False)
        if row == n_rows - 1 and col == 0:
            ax.axis("on")
            ax.set_ylabel("Neuron Number")
            ax.set_xlabel("Time Step")
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    if retrn:
        return fig
    plt.close()


def plot_traces(
    trace: torch.Tensor,
    spk: Optional[torch.Tensor] = None,
    dim: tuple = (3, 5),
    spike_height: float = 10,
    ylabel: str = "V(t)",
    title: str = "",
    savefig: str = "",
    show: bool = False,
    retrn: bool = True,
) -> None:
    """Plots a grid of traces.
    Args:
        trace: A tensor of shape (batch_size, n_steps, n_neurons) containing the traces.
        spk: (Optional) A tensor of shape (batch_size, n_steps, n_neurons) containing the spike times.
        dim: A tuple of (n_rows, n_cols) for the grid.
        spike_height: A float for the height of the spike.
        ylabel: A string for the ylabel of the plot.
        title: A string for the title of the plot.
        savefig: A string for the path to save the figure.
        show: A boolean to show the figure.
        retrn: A boolean to return the figure.

    Returns:
        None
    """

    n_examples = trace.shape[0]

    if dim is None:
        n_rows = int(math.ceil(math.sqrt(n_examples)))
        n_cols = int(math.ceil(n_examples / n_rows))
    else:
        n_rows = min(n_examples, dim[0])
        n_cols = min(math.ceil(n_examples / n_rows), dim[1])
    dim = (n_rows, n_cols)

    fig = plt.figure(figsize=(10, 4))
    if title:
        fig.suptitle(title)
    gs = plt.GridSpec(*dim, figure=fig)
    if spk is not None:
        dat = 1.0 * trace
        dat[spk > 0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = trace.detach().cpu().numpy()
    N = min(math.prod(dim), n_examples)
    for i in range(N):
        row = i // n_cols
        col = i % n_cols
        if i == 0:
            a0 = ax = plt.subplot(gs[row, col])
        else:
            ax = plt.subplot(gs[row, col], sharey=a0)
        ax.axis("off")
        ax.plot(dat[i], alpha=0.6)
        if row == n_rows - 1 and col == 0:
            ax.axis("on")
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time Step")
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    if retrn:
        return fig
    plt.close()
