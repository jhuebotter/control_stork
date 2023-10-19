import numpy as np
import time

import torch
import torch.nn as nn

from . import monitors
from .extratypes import *
from . import generators
from . import loss_stacks


# TODO: completely rewrite this class


class RecurrentSpikingModel(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> None:
        super(RecurrentSpikingModel, self).__init__()

        self.device = device
        self.dtype = dtype

        self.groups = []
        self.connections = []
        self.monitors = []
        self.hist = []

        self.optimizer = None
        self.input_group = None
        self.output_group = None

    def cofigure_groups(self):
        for g in self.groups:
            g.configure(
                self.time_step,
                self.device,
                self.dtype,
            )

    def configure_connections(self):
        for c in self.connections:
            c.configure(
                self.time_step,
                self.device,
                self.dtype,
            )

    def configure_objects(self):
        self.cofigure_groups()
        self.configure_connections()

    def configure(
        self,
        input,
        output,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_kwargs: Optional[dict] = None,
        time_step: float = 1e-3,
        loss_stack=None,
        generator=None,
    ):
        self.input_group = input
        self.output_group = output
        self.time_step = time_step

        if loss_stack is not None:
            self.loss_stack = loss_stack
        else:
            self.loss_stack = loss_stacks.MaxOverTimeCrossEntropy()

        if generator is None:
            self.data_generator_ = generators.StandardGenerator()
        else:
            self.data_generator_ = generator

        self.configure_objects()

        if optimizer is None:
            optimizer = torch.optim.Adam

        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-3, betas=(0.9, 0.999))

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.configure_optimizer(self.optimizer_class, self.optimizer_kwargs)
        self.to(self.device)

    def prepare_data(self, dataset):
        return self.data_generator_.prepare_data(dataset)

    def data_generator(self, dataset, shuffle=True):
        return self.data_generator_(dataset, shuffle=shuffle)
    
    def configure_optimizer(self, optimizer_class, optimizer_kwargs):
        if optimizer_kwargs is not None:
            self.optimizer_instance = optimizer_class(
                self.parameters(), **optimizer_kwargs
            )
        else:
            self.optimizer_instance = optimizer_class(self.parameters())

    def set_loss_stack(self, loss_stack):
        self.loss_stack = loss_stack

    def add_group(self, group):
        self.groups.append(group)
        self.add_module("group%i" % len(self.groups), group)
        return group

    def add_connection(self, con):
        self.connections.append(con)
        self.add_module("con%i" % len(self.connections), con)
        return con

    def add_monitor(self, monitor):
        self.monitors.append(monitor)
        return monitor

    def reset_state(self, batch_size: int = 1, monitors: bool = True):
        for g in self.groups:
            g.reset_state(batch_size)
        if monitors:
            self.reset_monitors()

    def reset_monitors(self):
        for m in self.monitors:
            m.reset()

    def evolve_all(self):
        for g in self.groups:
            g.evolve()
            g.clear_input()

    def apply_constraints(self):
        for c in self.connections:
            c.apply_constraints()

    def propagate_all(self):
        for c in self.connections:
            c.propagate()

    def monitor_all(self):
        for m in self.monitors:
            m.execute()

    def get_monitor_data(self, exclude: Optional[list] = None) -> dict:
        data = {}
        for m in self.monitors:
            k = f"{m.__class__.__name__}"
            if exclude is not None and k in exclude:
                continue
            if hasattr(m, "group"):
                k += f" on {m.group.name}"
            if hasattr(m, "key"):
                k += f" for {m.key}"
            data[k] = m.get_data()
        return data
    
    def compute_activity_regularizer_losses(self, reduction="mean"):
        reg_loss = torch.zeros(1, device=self.device)
        for g in self.groups:
            reg_loss = reg_loss + g.get_regularizer_loss(reduction=reduction)
        return reg_loss
    
    def compute_weight_regularizer_losses(self):
        reg_loss = torch.zeros(1, device=self.device)
        for c in self.connections:
            reg_loss = reg_loss + c.get_regularizer_loss()
        return reg_loss

    def compute_regularizer_losses(self):
        reg_loss = torch.zeros(1, device=self.device)
        reg_loss = reg_loss + self.compute_activity_regularizer_losses()
        reg_loss = reg_loss + self.compute_weight_regularizer_losses()
        return reg_loss

    def remove_regularizers(self):
        for g in self.groups:
            g.remove_regularizers()
        for c in self.connections:
            c.remove_regularizers()

    def forward(self, x, record=False):
        N, T, _ = x.shape

        self.input_group.feed_data(x)
        for t in range(T):
            self.evolve_all()
            self.propagate_all()
            if record:
                self.monitor_all()
        self.out = self.output_group.get_out_sequence()
        return self.out

    def get_total_loss(self, output, target_y, regularized=True):
        if type(target_y) in (list, tuple):
            target_y = [ty.to(self.device) for ty in target_y]
        else:
            target_y = target_y.to(self.device)

        # TODO: rework how loss is computed

        self.out_loss = self.loss_stack(output, target_y)

        if regularized:
            self.reg_loss = self.compute_regularizer_losses()
            total_loss = self.out_loss + self.reg_loss
        else:
            total_loss = self.out_loss

        return total_loss

    def regtrain_epoch(self, dataset, shuffle=True):
        # TODO: rework how training is done especially with the loss computation

        self.train(True)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            self.reg_loss = self.compute_regularizer_losses()

            total_loss = self.get_total_loss(output, local_y)
            loss = self.reg_loss

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            loss.backward()
            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def train_epoch(self, dataset, shuffle=True):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            self.reset_state(len(local_X))
            output = self.forward(local_X)
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            self.optimizer_instance.step()
            self.apply_constraints()

        return np.mean(np.array(metrics), axis=0)

    def get_metric_names(self, prefix="", postfix=""):
        metric_names = ["loss", "reg_loss"] + self.loss_stack.get_metric_names()
        return ["%s%s%s" % (prefix, k, postfix) for k in metric_names]

    def get_metrics_string(self, metrics_array, prefix="", postfix=""):
        s = ""
        names = self.get_metric_names(prefix, postfix)
        for val, name in zip(metrics_array, names):
            s = s + " %s=%.3g" % (name, val)
        return s

    def get_metrics_history_dict(self, metrics_array, prefix="", postfix=""):
        " Create metrics history dict. " ""
        s = ""
        names = self.get_metric_names(prefix, postfix)
        history = {name: metrics_array[:, k] for k, name in enumerate(names)}
        return history

    def fit(self, dataset, nb_epochs=10, verbose=True, shuffle=True, wandb=None):
        self.hist = []
        self.wall_clock_time = []
        self.train()
        for ep in range(nb_epochs):
            t_start = time.time()
            ret = self.train_epoch(dataset, shuffle=shuffle)
            self.hist.append(ret)

            if self.wandb is not None:
                self.wandb.log(
                    {key: value for (key, value) in zip(self.get_metric_names(), ret)}
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s t_iter=%.2f" % (ep, self.get_metrics_string(ret), t_iter)
                )

        self.fit_runs.append(self.hist)
        history = self.get_metrics_history_dict(np.array(self.hist))
        return history

    def fit_validate(
        self, dataset, valid_dataset, nb_epochs=10, verbose=True, wandb=None
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []
        for ep in range(nb_epochs):
            t_start = time.time()
            self.train()
            ret_train = self.train_epoch(dataset)

            self.train(False)
            ret_valid = self.evaluate(valid_dataset)
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            if self.wandb is not None:
                self.wandb.log(
                    {
                        key: value
                        for (key, value) in zip(
                            self.get_metric_names()
                            + self.get_metric_names(prefix="val_"),
                            ret_train.tolist() + ret_valid.tolist(),
                        )
                    }
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s --%s t_iter=%.2f"
                    % (
                        ep,
                        self.get_metrics_string(ret_train),
                        self.get_metrics_string(ret_valid, prefix="val_"),
                        t_iter,
                    )
                )

        self.hist = np.concatenate(
            (np.array(self.hist_train), np.array(self.hist_valid))
        )
        self.fit_runs.append(self.hist)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        history = {**dict1, **dict2}
        return history

    def predict(self, data, train_mode=False):
        self.train(train_mode)
        if type(data) in [torch.Tensor, np.ndarray]:
            output = self.forward_pass(data, cur_batch_size=len(data))
            pred = self.loss_stack.predict(output)
            return pred
        else:
            self.prepare_data(data)
            pred = []
            for local_X, _ in self.data_generator(data, shuffle=False):
                data_local = local_X.to(self.device)
                output = self.forward_pass(data_local, cur_batch_size=len(local_X))
                pred.append(self.loss_stack.predict(output).detach().cpu())
            return torch.cat(pred, dim=0)

    def monitor(self, dataset):
        self.prepare_data(dataset)

        # Prepare a list for each monitor to hold the batches
        results = [[] for _ in self.monitors]
        for local_X, local_y in self.data_generator(dataset, shuffle=False):
            for m in self.monitors:
                m.reset()

            output = self.forward_pass(
                local_X, record=True, cur_batch_size=len(local_X)
            )

            for k, mon in enumerate(self.monitors):
                results[k].append(mon.get_data())

        return [torch.cat(res, dim=0) for res in results]

    def monitor_backward(self, dataset):
        """
        Allows monitoring of gradients with GradientMonitor
            - If there are no GradientMonitors, this runs the usual `monitor` method
            - Returns both normal monitor output and backward monitor output
        """

        # if there is a gradient monitor
        if any([isinstance(m, monitors.GradientMonitor) for m in self.monitors]):
            self.prepare_data(dataset)

            # Set monitors to record gradients
            gradient_monitors = [
                m for m in self.monitors if isinstance(m, monitors.GradientMonitor)
            ]
            for gm in gradient_monitors:
                gm.set_hook()

            # Prepare a list for each monitor to hold the batches
            results = [[] for _ in self.monitors]
            for local_X, local_y in self.data_generator(dataset, shuffle=False):
                for m in self.monitors:
                    m.reset()

                # forward pass
                output = self.forward_pass(
                    local_X, record=True, cur_batch_size=len(local_X)
                )

                # compute loss
                total_loss = self.get_total_loss(output, local_y)

                # Use autograd to compute the backward pass.
                self.optimizer_instance.zero_grad()
                total_loss.backward()

                # do not call an optimizer step as that would update the weights!

                # Retrieve data from monitors
                for k, mon in enumerate(self.monitors):
                    results[k].append(mon.get_data())

            # Turn gradient recording off
            for gm in gradient_monitors:
                gm.remove_hook()

            return [torch.cat(res, dim=0) for res in results]

        else:
            return self.monitor(dataset)

    def record_group_outputs(self, group, x_input):
        res = []
        # we don't care about the labels, but want to use the generator
        fake_labels = torch.zeros(len(x_input))
        self.prepare_data((x_input, fake_labels))
        for local_X, _ in self.data_generator((x_input, fake_labels)):
            output = self.forward_pass(local_X)
            res.append(group.get_out_sequence())
        return torch.cat(res, dim=0)

    def evaluate_ensemble(
        self, dataset, test_dataset, nb_repeats=5, nb_epochs=10, callbacks=None
    ):
        """Fits the model nb_repeats times to the data and returns evaluation results.

        Args:
            dataset: Training dataset
            test_dataset: Testing data
            nb_repeats: Number of repeats to retrain the model (default=5)
            nb_epochs: Train for x epochs (default=20)
            callbacks: A list with callbacks (functions which will be called as f(self) whose return value
                       is stored in a list and returned as third return value

        Returns:
            List of learning training histories curves
            and a list of test scores and if callbacks is not None an additional
            list with the callback results
        """
        results = []
        test_scores = []
        callback_returns = []
        for k in range(nb_repeats):
            print("Repeat %i/%i" % (k + 1, nb_repeats))
            self.reconfigure()
            self.fit(dataset, nb_epochs=nb_epochs, verbose=(k == 0))
            score = self.evaluate(test_dataset)
            results.append(np.array(self.hist))
            test_scores.append(score)
            if callbacks is not None:
                callback_returns.append([callback(self) for callback in callbacks])

        if callbacks is not None:
            return results, test_scores, callback_returns
        else:
            return results, test_scores
    
    def count_parameters(self):
        # TODO: check if this counts MaskedTensor parameters correctly
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary"""

        print("\n# Model summary")
        print("\n## Groups")
        for group in self.groups:
            if group.name is None or group.name == "":
                print("no name, %s" % (group.shape,))
            else:
                print("%s, %s" % (group.name, group.shape))

        print("\n## Connections")
        for con in self.connections:
            print(con)

        print("\n## Trainable Parameters")

        print("Total number of trainable parameters: %i" % self.count_parameters())
        print(
            "Number of parameter objects:",
            len([p for p in self.parameters() if p.requires_grad]),
        )

    def __str__(self):
        self.summary()
        return ""

    def to(self, device):
        self.device = device
        super().to(device)
        self.configure_objects()

        return self
