from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data


import numpy as np
from functools import partial
import sys, os, time, itertools

from networks import *
from losses import *

try:
    from tqdm import tqdm, trange
except ImportError:

    def tqdm(x):
        return x

    def trange(x):
        return x

try:
    import distributed
    HAS_DISTRIBUTED = True
except ImportError:
    distributed = None
    HAS_DISTRIBUTED = False


def _in_dask_worker():
    if HAS_DISTRIBUTED:
        from distributed.utils import thread_state
        return hasattr(thread_state, "execution_state")
    else:
        return False


logger = logging.getLogger("mmd_gradient_flow")


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = (
            "cuda:" + str(args.device)
            if torch.cuda.is_available() and args.device > -1
            else "cpu"
        )

        self.dtype = get_dtype(args)
        self.log_dir = os.path.join(
            args.log_dir,
            args.log_name
            + "_loss_"
            + args.loss
            + "_noise_level_"
            + str(args.noise_level),
        )

        logger.handlers[:] = []
        if args.log_in_file:
            if not os.path.isdir(self.log_dir):
                os.mkdir(self.log_dir)

            self.log_file = open(
                os.path.join(self.log_dir, "log.txt"), "w", buffering=1
            )
            logger.addHandler(logging.FileHandler(self.log_file))
        else:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(getattr(logging, args.log_level))

        logger.debug("==> Building model..")
        self.build_model()

    def build_model(self):
        torch.manual_seed(self.args.seed)
        if not self.args.with_noise:
            self.args.noise_level = 0.0
        self.teacherNet = get_net(
            self.args, self.dtype, self.device, "teacher"
        )
        self.student = get_net(self.args, self.dtype, self.device, "student")
        self.data_train = get_data_gen(
            self.teacherNet, self.args, self.dtype, self.device
        )
        self.data_valid = get_data_gen(
            self.teacherNet, self.args, self.dtype, self.device
        )

        self.loss = self.get_loss()

        self.optimizer = self.get_optimizer(self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=50, verbose=True, factor=0.9
        )
        self.records = {
            "lr": [],
            "noise_level": [],
            "train_loss": [],
            "val_loss": [],
        }

        self.metadata = vars(self.args)

    def get_loss(self):
        if self.args.loss == "mmd_noise_injection":
            return MMD(
                self.student,
                self.args.with_noise,
                self.args.inject_noise_in_prediction,
            )
        elif self.args.loss == "mmd_diffusion":
            return MMD_Diffusion(self.student)
        elif self.args.loss == "sobolev":
            return Sobolev(self.student)

    def get_optimizer(self, lr):
        if self.args.optimizer == "SGD":
            return optim.SGD(self.student.parameters(), lr=lr)

    def init_student(self, mean, std):
        weights_init_student = partial(
            weights_init, {"mean": mean, "std": std}
        )
        self.student.apply(weights_init_student)

    def train(self, start_epoch=0, total_iters=0):
        logger.debug("Starting Training Loop...")
        start_time = time.time()
        best_valid_loss = np.inf
        r = range(start_epoch, start_epoch + self.args.total_epochs)
        if getattr(logging, self.args.log_level) > logging.INFO:
            if not _in_dask_worker():
                r = tqdm(r)
        for epoch in r:
            total_iters, train_loss = train_epoch(
                epoch,
                total_iters,
                self.loss,
                self.data_train,
                self.optimizer,
                "train",
                device=self.device,
            )
            total_iters, valid_loss = train_epoch(
                epoch,
                total_iters,
                self.loss,
                self.data_valid,
                self.optimizer,
                "valid",
                device=self.device,
            )

            self.records["train_loss"].append(train_loss)
            self.records["val_loss"].append(valid_loss)
            self.records["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.records["noise_level"].append(
                self.student.linear1.noise_level
            )

            if not np.isfinite(train_loss):
                break

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            if self.args.use_scheduler:
                self.scheduler.step(train_loss)
            if np.mod(epoch, self.args.noise_decay_freq) == 0 and epoch > 0:
                self.loss.student.update_noise_level()
            if np.mod(epoch, 10) == 0:
                if hasattr(r, "set_description"):
                    r.set_description(
                        f"train {train_loss:.3f}, valid: {valid_loss:.3f}"
                    )
                new_time = time.time()

                start_time = new_time
        return train_loss, valid_loss, best_valid_loss


def get_data_gen(net, args, dtype, device):
    params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}
    if args.input_data == "Spherical":
        teacher = SphericalTeacher(net, args.N_train, dtype, device)
    return data.DataLoader(teacher, **params)


def get_net(args, dtype, device, net_type):
    non_linearity = get_non_linearity(args.non_linearity)
    if net_type == "teacher":
        weights_init_net = partial(
            weights_init, {"mean": args.mean_teacher, "std": args.std_teacher}
        )
        if args.teacher_net == "OneHidden":
            Net = OneHiddenLayer(
                args.d_int, args.H, args.d_out, non_linearity, bias=args.bias,
            )
    if net_type == "student":
        weights_init_net = partial(
            weights_init, {"mean": args.mean_student, "std": args.std_student}
        )
        if args.student_net == "NoisyOneHidden":
            Net = NoisyOneHiddenLayer(
                args.d_int,
                args.H,
                args.d_out,
                args.num_particles,
                non_linearity,
                noise_level=args.noise_level,
                noise_decay=args.noise_decay,
                bias=args.bias,
            )

    Net.to(device)
    if args.dtype == "float64":
        Net.double()

    Net.apply(weights_init_net)
    return Net


def get_dtype(args):
    if args.dtype == "float32":
        return torch.float32
    else:
        return torch.float64


def weights_init(args, m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=args["mean"], std=args["std"])
        if m.bias:
            m.bias.data.normal_(mean=args["mean"], std=args["std"])


def train_epoch(
    epoch, total_iters, Loss, data_loader, optimizer, phase, device="cuda"
):

    # Training Loop
    # Lists to keep track of progress

    if phase == "train":
        Loss.student.train(True)  # Set model to training mode
    else:
        Loss.student.train(False)  # Set model to evaluate mode

    cum_loss = 0
    # For each epoch

    # For each batch in the dataloader
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if phase == "train":
            total_iters += 1
            Loss.student.zero_grad()
            loss = Loss(inputs, targets)
            # Calculate the gradients for this batch
            loss.backward()
            optimizer.step()
            loss = loss.item()
            cum_loss += loss

        elif phase == "valid":
            loss = Loss(inputs, targets).item()
            cum_loss += loss
    total_loss = cum_loss / (batch_idx + 1)
    if np.mod(epoch, 10) == 0:

        logger.info(
            "Epoch: "
            + str(epoch)
            + " | "
            + phase
            + " loss: "
            + str(round(total_loss, 5))
        )
    return total_iters, total_loss
