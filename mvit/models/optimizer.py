#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Optimizer."""

import json

import mvit.utils.lr_policy as lr_policy
import torch


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
            optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
            learning rate,  momentum, weight_decay, dampening, and etc.
    """
    if cfg.SOLVER.LAYER_DECAY > 0.0 and cfg.SOLVER.LAYER_DECAY < 1.0:
        optim_params = get_ld_param_groups(model, cfg)
    elif cfg.SOLVER.LAYER_DECAY == 1.0:
        optim_params = get_param_groups(model, cfg)
    else:
        raise ValueError(
            "Layer decay should be in (0, 1], but is {}".format(cfg.SOLVER.LAYER_DECAY)
        )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_param_groups(model, cfg):
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []
    skip = {}
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif any(k in name for k in skip) or (
                (len(p.shape) == 1 or name.endswith(".bias"))
                and cfg.SOLVER.ZERO_WD_1D_PARAM
            ):
                zero_parameters.append(p)
            else:
                non_bn_parameters.append(p)

    optim_params = [
        {"params": bn_parameters, "weight_decay": 0.0},
        {
            "params": non_bn_parameters,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        },
        {"params": zero_parameters, "weight_decay": 0.0},
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_parameters
    ) + len(zero_parameters) + len(
        no_grad_parameters
    ), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(list(model.parameters())),
    )
    print(
        "bn {}, non bn {}, zero {} no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )

    return optim_params


def get_ld_param_groups(model, cfg):
    def _get_layer_decay(name):
        layer_id = None
        if name in ("cls_token", "mask_token", "pos_embed"):
            layer_id = 0
        elif name.startswith("patch_embed"):
            layer_id = 0
        elif name.startswith("blocks"):
            layer_id = int(name.split(".")[1]) + 1
        else:
            layer_id = cfg.MVIT.DEPTH + 1
        layer_decay = cfg.SOLVER.LAYER_DECAY ** (cfg.MVIT.DEPTH + 1 - layer_id)
        return layer_id, layer_decay

    for m in model.modules():
        assert not isinstance(
            m, torch.nn.modules.batchnorm._NormBase
        ), "BN is not supported with layer decay"

    non_bn_parameters_count = 0
    zero_parameters_count = 0
    no_grad_parameters_count = 0
    parameter_group_names = {}
    parameter_group_vars = {}

    skip = {}
    if cfg.NUM_GPUS > 1:
        if hasattr(model.module, "no_weight_decay"):
            skip = model.module.no_weight_decay()
            # skip = {"module." + v for v in skip}
    else:
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            group_name = "no_grad"
            no_grad_parameters_count += 1
            continue
        name = name[len("module.") :] if name.startswith("module.") else name
        if name in skip or (
            (len(p.shape) == 1 or name.endswith(".bias"))
            and cfg.SOLVER.ZERO_WD_1D_PARAM
        ):
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = "layer_%d_%s" % (layer_id, "zero")
            weight_decay = 0.0
            zero_parameters_count += 1
        else:
            layer_id, layer_decay = _get_layer_decay(name)
            group_name = "layer_%d_%s" % (layer_id, "non_bn")
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            non_bn_parameters_count += 1

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": weight_decay,
                "params": [],
                "layer_decay": layer_decay,
            }
        parameter_group_names[group_name]["params"].append(name)
        parameter_group_vars[group_name]["params"].append(p)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    optim_params = list(parameter_group_vars.values())

    # Check all parameters will be passed into optimizer.
    assert (
        len(list(model.parameters()))
        == non_bn_parameters_count + zero_parameters_count + no_grad_parameters_count
    ), "parameter size does not match: {} + {} + {} != {}".format(
        non_bn_parameters_count,
        zero_parameters_count,
        no_grad_parameters_count,
        len(list(model.parameters())),
    )
    print(
        "non bn {}, zero {}, no grad {}".format(
            non_bn_parameters_count,
            zero_parameters_count,
            no_grad_parameters_count,
        )
    )

    return optim_params


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        ld = param_group["layer_decay"] if "layer_decay" in param_group else 1.0
        param_group["lr"] = new_lr * ld
