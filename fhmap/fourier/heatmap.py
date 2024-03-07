import copy
import pathlib
from collections import OrderedDict
from typing import Final, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

import fhmap
import fhmap.fourier as fourier
from fhmap.fourier.noise import AddFourierNoise


def create_fourier_heatmap_from_error_matrix(
    error_matrix: torch.Tensor,
) -> torch.Tensor:
    """Create Fourier Heat Map from error matrix (about quadrant 1 and 4).

    Note:
        Fourier Heat Map is symmetric about the origin.
        So by performing an inversion operation about the origin, Fourier Heat Map is created from error matrix.

    Args:
        error_matrix (torch.Tensor): The size of error matrix should be (H, H/2+1). Here, H is height of image.
                                     This error matrix shoud be about quadrant 1 and 4.

    Returns:
        torch.Tensor (torch.Tensor): Fourier Heat Map created from error matrix.

    """
    assert len(error_matrix.size()) == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    fhmap_rightside = error_matrix[1:, :-1]
    fhmap_leftside = torch.flip(fhmap_rightside, (0, 1))
    return torch.cat([fhmap_leftside[:, :-1], fhmap_rightside], dim=1)


def save_fourier_heatmap(
    fhmap: torch.Tensor, savedir: pathlib.Path, suffix: str = "", vmin: float = 0.0, vmax: float = 1.0
) -> None:
    """Save Fourier Heat Map as a png image.

    Args:
        fhmap (torch.Tensor): Fourier Heat Map.
        savedir (pathlib.Path): Path to the directory where the results will be saved.
        suffix (str, optional): Suffix which is attached to result file of Fourier Heat Map.

    """
    torch.save(fhmap, savedir / ("fhmap_data" + suffix + ".pth"))  # save raw data.
    sns.heatmap(
        fhmap.numpy(),
        vmin=vmin,
        vmax=vmax,
        cmap="jet",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.savefig(savedir / ("fhmap" + suffix + ".png"))
    plt.close("all")  # This is needed for continuous figure generation.


def insert_fourier_noise(transforms: List, basis: torch.Tensor) -> None:
    """Insert Fourier noise transform to given a list of transform by inplace operation.

    Note:
        If Normalize transform is included in the given list, Fourier noise transform is added to just before Normalize transform.
        If not, Fourier noise transform is added at the end of the list.

    Args:
        transforms (List): A list of transform.
        basis (torch.Tensor): 2D Fourier basis.

    """

    def _get_index_of_normalize(transforms: List) -> Optional[int]:
        for index, transform in enumerate(transforms):
            if isinstance(transform, torchvision.transforms.Normalize):
                return index

        return None

    normalize_index: Final = _get_index_of_normalize(transforms)
    insert_index: Final = normalize_index if normalize_index else -1

    transforms.insert(insert_index, AddFourierNoise(basis))


def eval_fourier_heatmap(
    input_size: int,
    ignore_edge_size: int,
    eps: float,
    arch: nn.Module,
    dataset: torchvision.datasets.VisionDataset,
    batch_size: int,
    device: torch.device,
    topk: Tuple[int, ...] = (1,),
    savedir: Optional[pathlib.Path] = None,
    logger: Optional[logging.Logger] = None,
) -> List[torch.Tensor]:
    """Evaluate Fourier Heat Map about given architecture and dataset.

    Args:
        input_size (int): A size of input image.
        ignore_edge_size (int): A size of the edge to ignore.
        eps (float): L2 norm size of Fourier basis.
        arch (nn.Module): An architecture to be evaluated.
        dataset (torchvision.datasets.VisionDataset): A dataset used for evaluation.
        batch_size (int): A size of batch.
        device (torch.device): A device used for calculation.
        topk (Tuple[int, ...], optional): Tuple of int which you want to know error.
        savedir (pathlib.Path, optional): Path to the directory where the results will be saved.

    Returns:
        List[torch.Tensor]: List of Fourier Heat Map.

    """
    if input_size % 2 != 0:
        raise ValueError("currently we only support even input size.")

    height: Final[int] = input_size
    width: Final[int] = height // 2 + 1
    fhmap_height: Final[int] = height - 2 * ignore_edge_size
    fhmap_width: Final[int] = width - ignore_edge_size

    if not isinstance(dataset.transform, torchvision.transforms.Compose):
        raise ValueError(
            f"type of dataset.transform should be torchvision.transforms.Compose, not {type(dataset.transform)}"
        )
    original_transforms: Final = dataset.transform.transforms

    error_matrix_dict = {k: torch.zeros(fhmap_height * fhmap_width).float() for k in topk}
    error_matrix_dict_bn = {k: torch.zeros(fhmap_height * fhmap_width).float() for k in topk}
    # bn_adapted_arch = copy.deepcopy(arch)
    # bn_adapted_arch = fhmap.AdaptiveBatchNorm.adapt_model(bn_adapted_arch, 0.0, device)
    # logger.info(f"Adapted model: {bn_adapted_arch}")
    # logger.info(f"Original model: {arch}")

    spectrums = fourier.get_spectrum(height, width, ignore_edge_size, ignore_edge_size)
    with tqdm(
        spectrums, ncols=160, total=fhmap_height * fhmap_width
    ) as pbar:  # without total progress par might not be shown.
        for i, spectrum in enumerate(pbar):  # Size of basis is [height, width]
            basis = fourier.spectrum_to_basis(spectrum, l2_normalize=True) * eps
            bn_adapted_arch = copy.deepcopy(arch)
            bn_adapted_arch = fhmap.AdaptiveBatchNorm.adapt_model(bn_adapted_arch, 0.0, device)

            # insert fourier noise by inpalce operation
            noised_transforms = copy.deepcopy(original_transforms)
            insert_fourier_noise(noised_transforms, basis)

            # overwrite torchvision.transforms.Compose.transforms
            dataset.transform.transforms = noised_transforms

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

            for k, (mean_err, mean_err_bn) in zip(
                topk, fhmap.eval_mean_errors(arch, bn_adapted_arch, loader, device, topk)
            ):
                error_matrix_dict[k][i] = mean_err
                error_matrix_dict_bn[k][i] = mean_err_bn

            # show result to pbar
            results = OrderedDict()
            for k, v in error_matrix_dict.items():
                results[f"err{k}"] = v[i].item()
            pbar.set_postfix(results)
            pbar.update()

    fourier_heatmaps: Final[List[torch.Tensor]] = [
        create_fourier_heatmap_from_error_matrix(error_matrix_dict[k].view(fhmap_height, fhmap_width)) for k in topk
    ]
    fourier_heatmaps_bn: Final[List[torch.Tensor]] = [
        create_fourier_heatmap_from_error_matrix(error_matrix_dict_bn[k].view(fhmap_height, fhmap_width)) for k in topk
    ]
    fourier_heatmaps_bn_original: Final[List[torch.Tensor]] = [
        create_fourier_heatmap_from_error_matrix(
            error_matrix_dict_bn[k].view(fhmap_height, fhmap_width)
            - error_matrix_dict[k].view(fhmap_height, fhmap_width)
        )
        for k in topk
    ]

    if savedir:
        for k, fourier_heatmap, fourier_heatmap_bn, fourier_heatmap_bn_original in zip(
            topk, fourier_heatmaps, fourier_heatmaps_bn, fourier_heatmaps_bn_original
        ):
            save_fourier_heatmap(fourier_heatmap / 100.0, savedir, f"_top{k}")
            save_fourier_heatmap(fourier_heatmap_bn / 100.0, savedir, f"_top{k}_bn")
            save_fourier_heatmap(
                fourier_heatmap_bn_original / 100.0, savedir, f"_top{k}_bn_original", vmin=-1.0, vmax=1.0
            )

    return fourier_heatmaps


class AdaptiveBatchNorm(nn.Module):
    """Use the source statistics as a prior on the target statistics"""

    @staticmethod
    def find_bns(parent, prior, device):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = AdaptiveBatchNorm(child, prior, device)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AdaptiveBatchNorm.find_bns(child, prior, device))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior, device):
        replace_mods = AdaptiveBatchNorm.find_bns(model, prior, device)
        # print(f"| Found {len(replace_mods)} modules to be replaced.")
        for parent, name, child in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior, device):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0).to(device)

        self.prior = prior

    def forward(self, input):
        self.norm(input)

        running_mean = self.prior * self.layer.running_mean + (1 - self.prior) * self.norm.running_mean
        running_var = self.prior * self.layer.running_var + (1 - self.prior) * self.norm.running_var

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )
