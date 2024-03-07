import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .fourier.heatmap import eval_fourier_heatmap  # noqa

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calc_errors(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """Calculate top-k errors over output from architecture (model).

    Args:
        output (torch.Tensor): Output tensor from architecture (model).
        target (torch.Tensor): Training target tensor.
        topk (Tuple[int, ...]): Tuple of int which you want to know error.

    Returns:
        List[torch.Tensor]: list of errors.

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(
            maxk, dim=1
        )  # return the k larget elements. top-k index: size (b, k).
        pred = pred.t()  # (k, b)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        errors = list()
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            wrong_k = batch_size - correct_k
            errors.append(wrong_k.mul_(100.0 / batch_size))

        return errors


def eval_mean_errors(
    arch: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...] = (1,),
) -> List[float]:
    """Evaluate top-k mean errors of the architecture over given dataloader.

    Args:
        arch (nn.Module): An architecture to be evaluated.
        loader (DataLoader): A dataloader.
        device (torch.device): A device used for calculation.
        topk (Tuple[int, ...]): Tuple of int which you want to know error.

    Returns:
        List[float]: list of mean errors.

    """
    arch = arch.to(device)
    err_dict: Dict[int, List[float]] = {k: list() for k in topk}

    for x, t in loader:
        x, t = x.to(device), t.to(device)

        output = arch(x)
        for k, err in zip(topk, calc_errors(output, t, topk=topk)):
            err_dict[k].append(err.item())

    return [sum(err_dict[k]) / len(err_dict[k]) for k in err_dict]


def get_first_batchnorm_output(model, x):
    outputs = []

    def hook(module, input, output):
        outputs.append(output)
        return None

    # 最初のBatchNorm層を見つけてフックを登録
    for _, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            handle = layer.register_forward_hook(hook)
            break

    # モデルを通じて入力を前進させ、フックを解除
    model(x)
    handle.remove()

    return outputs[0]  # 最初のBatchNorm層の出力を返す


def get_batchnorm_output(model, x):
    bn_outputs = []

    hooks = []

    def register_hook(module):
        if isinstance(module, nn.BatchNorm2d):
            hook = module.register_forward_hook(lambda module, input, output: bn_outputs.append(output))
            hooks.append(hook)

    # モデルの全BN層にフックを設定
    model.apply(register_hook)

    # モデルを通じて入力を前進させる
    model(x)

    # フックを解除
    for hook in hooks:
        hook.remove()

    return bn_outputs


def eval_batchnorm_similarity(
    arch: nn.Module,
    loader: DataLoader,
    original_bn_outputs,
    device: torch.device,
) -> List[float]:
    """フーリエノイズが加えられる前後のBN出力の類似度計算

    Args:
        arch (nn.Module): An architecture to be evaluated.
        loader (DataLoader): A dataloader.
        noised_transforms (List): A list of noise transform.
        device (torch.device): A device used for calculation.

    Returns:
        List[float]: list of mean errors.

    """
    arch = arch.to(device)
    arch.eval()
    similarity_dict: Dict[int, List[float]] = {k: list() for k in range(len(original_bn_outputs))}

    noised_bn_outputs = []
    for x, _ in loader:
        with torch.no_grad():
            x = x.to(device)
            outputs = get_batchnorm_output(arch, x)
            if not noised_bn_outputs:
                noised_bn_outputs = [[] for _ in range(len(outputs))]
            for j, output in enumerate(outputs):
                noised_bn_outputs[j].append(output)

    for k, (original_bn_output, noised_bn_output) in enumerate(zip(original_bn_outputs, noised_bn_outputs)):
        for original, noised in zip(original_bn_output, noised_bn_output):
            # cos_sim = (
            #     F.cosine_similarity(original.flatten(start_dim=2), noised.flatten(start_dim=2), dim=2)
            #     .mean(dim=1)
            #     .mean()
            #     .item()
            # )
            l2_diff = torch.norm(original - noised, p=2, dim=[1, 2, 3]).mean().item()
            similarity_dict[k].append(l2_diff)

    return [sum(similarity_dict[k]) / len(similarity_dict[k]) for k in similarity_dict]
