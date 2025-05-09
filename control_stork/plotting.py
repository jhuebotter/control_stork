import torch
import math
import matplotlib.pyplot as plt
from .extratypes import *
import numpy as np


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

    fig = plt.figure(figsize=(n_cols * 2, n_rows * 1.5))
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

    plt.tight_layout()
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
    ylabel: str = "U(t)",
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

    fig = plt.figure(figsize=(n_cols * 2, n_rows * 1.5))
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

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    if retrn:
        return fig
    plt.close()


def plot_histogram(
    data: torch.Tensor,
    bins: int = 20,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "Count",
    color: Optional[str] = None,
    savefig: str = "",
    show: bool = False,
    retrn: bool = True,
    xlim_min: Optional[float] = None,
    xlim_max: Optional[float] = None,
) -> plt.Figure:
    """
    Plot a histogram of the values in a 1D tensor, with optional x-axis limits and minimal styling.

    Args:
        data (torch.Tensor): 1D tensor of values to plot.
        bins (int): Number of histogram bins.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (Optional[str]): Matplotlib color spec for the bars.
        savefig (str): Path to save the figure (if non-empty).
        show (bool): Whether to call plt.show().
        retrn (bool): Whether to return the Figure. If False, the figure is closed after creation.
        xlim_min (Optional[float]): Lower x-axis limit. If None, set to 90% of min(data).
        xlim_max (Optional[float]): Upper x-axis limit. If None, set to 110% of max(data).

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the histogram.
    """
    # prepare data array
    arr = data.detach().cpu().numpy().flatten()
    # compute data range and axis limits
    data_min, data_max = arr.min(), arr.max()
    axis_lower = data_min * 0.9 if xlim_min is None else xlim_min
    axis_upper = data_max * 1.1 if xlim_max is None else xlim_max

    # figure with smaller default size
    fig, ax = plt.subplots(figsize=(3, 2))
    # determine if single-value
    unique_vals = np.unique(arr)
    if unique_vals.size == 1:
        # single-value case: bar spanning narrower [95%,105%] around the value
        v = unique_vals[0]
        bar_lower = v * 0.95
        bar_upper = v * 1.05
        ax.hist(arr, bins=[bar_lower, bar_upper], color=color, alpha=0.7)
    else:
        # multi-value case: histogram over padded axis limits
        ax.hist(arr, bins=bins, range=(axis_lower, axis_upper), color=color, alpha=0.7)

    # plot mean line
    mean_val = arr.mean()
    ax.axvline(mean_val, color="k", linestyle="--", linewidth=1)

    # axis labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # apply axis limits
    ax.set_xlim(axis_lower, axis_upper)
    # remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # save or show
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig)
    if show:
        plt.show()
    # return or close
    if retrn:
        return fig
    plt.close()
