from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T
from typing import Any


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            if target is None:
                image = t(image)
            else:
                image, target = t(image, target)
        if target is None:
            return image
        return image, target


class WrapTvTensors(nn.Module):
    """Wrap image, boxes, and masks into torchvision v2 tv_tensors.

    This enables torchvision v2 transforms (e.g., RandomRotation, RandomAffine)
    to correctly apply geometric ops to targets.
    """

    def forward(self, image: Tensor, target: Optional[Dict[str, Any]] = None):
        try:
            from torchvision import tv_tensors
        except Exception:
            # If tv_tensors is not available, no-op
            return image, target

        # Determine image dims (C, H, W) works for both PIL and Tensor
        if isinstance(image, torch.Tensor):
            _, H, W = F.get_dimensions(image)
        else:
            # PIL Image path: let v2 handle it, but we still need canvas for boxes
            w, h = image.size
            H, W = h, w

        # Wrap image
        if not isinstance(image, torch.Tensor) or not isinstance(image, tv_tensors.Image):
            # Ensure tensor first
            if not isinstance(image, torch.Tensor):
                image = F.pil_to_tensor(image)
            image = tv_tensors.Image(image)

        # Wrap targets
        if target is not None:
            if "boxes" in target and not isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                target["boxes"] = tv_tensors.BoundingBoxes(
                    target["boxes"], format="XYXY", canvas_size=(H, W)
                )
            if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                target["masks"] = tv_tensors.Mask(target["masks"])  # (N, H, W)
        if target is None:
            return image
        return image, target


class UnwrapTvTensors(nn.Module):
    """Convert tv_tensors back to plain tensors for the model consumption."""

    def forward(self, image: Tensor, target: Optional[Dict[str, Any]] = None):
        try:
            from torchvision import tv_tensors
        except Exception:
            return image, target

        # Unwrap image
        if isinstance(image, torch.Tensor) and isinstance(image, tv_tensors.Image):
            image = torch.as_tensor(image)

        # Unwrap targets
        if target is not None:
            if "boxes" in target and isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                # Ensure float32 dtype
                target["boxes"] = torch.as_tensor(target["boxes"]).to(torch.float32)
            if "masks" in target and isinstance(target["masks"], tv_tensors.Mask):
                target["masks"] = torch.as_tensor(target["masks"]).to(torch.uint8)
        if target is None:
            return image
        return image, target


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        if target is None:
            return image
        return image, target


class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            if target is not None:
                _, height, _ = F.get_dimensions(image)
                target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-2)
        if target is None:
            return image
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        if target is None:
            return image
        return image, target


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if not self.scale:
            image = image.to(dtype=self.dtype)
        else:
            image = F.convert_image_dtype(image, self.dtype)
        if target is None:
            return image
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)

                # Crop image and masks
                image = F.crop(image, top, left, new_h, new_w)
                if target is not None and "masks" in target:
                    target["masks"] = F.crop(target["masks"][is_within_crop_area], top, left, new_h, new_w)

                # Preserve tv_tensors semantics and update canvas_size for boxes
                try:
                    from torchvision import tv_tensors
                    # Ensure image is tv_tensors.Image
                    if not isinstance(image, tv_tensors.Image):
                        image = tv_tensors.Image(image)
                    # Wrap masks back to tv_tensors.Mask
                    if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                        target["masks"] = tv_tensors.Mask(target["masks"])  # (N, H, W)
                    # Rebuild boxes with updated canvas_size
                    if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                        target["boxes"] = tv_tensors.BoundingBoxes(
                            target["boxes"], format="XYXY", canvas_size=(new_h, new_w)
                        )
                    else:
                        target["boxes"] = tv_tensors.BoundingBoxes(
                            target["boxes"], format="XYXY", canvas_size=(new_h, new_w)
                        )
                except Exception:
                    pass

                return image, target


class RandomZoomOut(nn.Module):
    def __init__(
        self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h):, :] = image[
                ..., :, (left + orig_w):
            ] = v

        if target is not None:
            # Shift boxes by the padding amounts
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top
            # Pad masks to match new canvas size
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], [left, top, right, bottom], 0, "constant")

            # Preserve tv_tensors semantics and update canvas_size for boxes
            try:
                from torchvision import tv_tensors
                new_canvas_h, new_canvas_w = canvas_height, canvas_width
                # Ensure image is tv_tensors.Image
                if not isinstance(image, tv_tensors.Image):
                    image = tv_tensors.Image(image)
                # Wrap masks back to tv_tensors.Mask
                if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                    target["masks"] = tv_tensors.Mask(target["masks"])  # (N, H, W)
                # Rebuild boxes with updated canvas_size
                if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_canvas_h, new_canvas_w)
                    )
                else:
                    # If plain tensor, make it BoundingBoxes so downstream v2 ops track geometry
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_canvas_h, new_canvas_w)
                    )
            except Exception:
                pass

        if target is None:
            return image
        return image, target


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        if target is None:
            return image
        return image, target


class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
    <https://arxiv.org/abs/2012.07177>`_.

    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias=True,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation, antialias=self.antialias)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"],
                    [new_height, new_width],
                    interpolation=InterpolationMode.NEAREST,
                    antialias=self.antialias,
                )
            # Preserve tv_tensors semantics and update canvas_size for boxes
            try:
                from torchvision import tv_tensors
                if not isinstance(image, tv_tensors.Image):
                    image = tv_tensors.Image(image)
                if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                    target["masks"] = tv_tensors.Mask(target["masks"])
                if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_height, new_width)
                    )
                else:
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_height, new_width)
                    )
            except Exception:
                pass

        if target is None:
            return image
        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")
            # Preserve tv_tensors semantics and update canvas_size for boxes
            try:
                from torchvision import tv_tensors
                # Determine new size from padded image
                _, new_h, new_w = F.get_dimensions(img)
                if not isinstance(img, tv_tensors.Image):
                    img = tv_tensors.Image(img)
                if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                    target["masks"] = tv_tensors.Mask(target["masks"])
                if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_h, new_w)
                    )
                else:
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_h, new_w)
                    )
            except Exception:
                pass

        if target is None:
            return img
        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                target["masks"] = F.crop(target["masks"][is_valid], top, left, height, width)

            # Preserve tv_tensors semantics and update canvas_size for boxes
            try:
                from torchvision import tv_tensors
                if not isinstance(img, tv_tensors.Image):
                    img = tv_tensors.Image(img)
                if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                    target["masks"] = tv_tensors.Mask(target["masks"])
                if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(height, width)
                    )
                else:
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(height, width)
                    )
            except Exception:
                pass

        if target is None:
            return img
        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        if target is None:
            return img
        return img, target


class RandomShortestSize(nn.Module):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )
            # Preserve tv_tensors semantics and update canvas_size for boxes
            try:
                from torchvision import tv_tensors
                if not isinstance(image, tv_tensors.Image):
                    image = tv_tensors.Image(image)
                if "masks" in target and not isinstance(target["masks"], tv_tensors.Mask):
                    target["masks"] = tv_tensors.Mask(target["masks"])
                if isinstance(target["boxes"], tv_tensors.BoundingBoxes):
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_height, new_width)
                    )
                else:
                    target["boxes"] = tv_tensors.BoundingBoxes(
                        target["boxes"], format="XYXY", canvas_size=(new_height, new_width)
                    )
            except Exception:
                pass

        if target is None:
            return image
        return image, target


def _copy_paste(
    image: torch.Tensor,
    target: Dict[str, Tensor],
    paste_image: torch.Tensor,
    paste_target: Dict[str, Tensor],
    blending: bool = True,
    resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, Dict[str, Tensor]]:

    # Random paste targets selection:
    num_masks = len(paste_target["masks"])

    if num_masks < 1:
        # Such degerante case with num_masks=0 can happen with LSJ
        # Let's just return (image, target)
        return image, target

    # We have to please torch script by explicitly specifying dtype as torch.long
    random_selection = torch.randint(0, num_masks, (num_masks,), device=paste_image.device)
    random_selection = torch.unique(random_selection).to(torch.long)

    paste_masks = paste_target["masks"][random_selection]
    paste_boxes = paste_target["boxes"][random_selection]
    paste_labels = paste_target["labels"][random_selection]

    masks = target["masks"]

    # We resize source and paste data if they have different sizes
    # This is something we introduced here as originally the algorithm works
    # on equal-sized data (for example, coming from LSJ data augmentations)
    size1 = image.shape[-2:]
    size2 = paste_image.shape[-2:]
    if size1 != size2:
        paste_image = F.resize(paste_image, size1, interpolation=resize_interpolation)
        paste_masks = F.resize(paste_masks, size1, interpolation=F.InterpolationMode.NEAREST)
        # resize bboxes:
        ratios = torch.tensor((size1[1] / size2[1], size1[0] / size2[0]), device=paste_boxes.device)
        paste_boxes = paste_boxes.view(-1, 2, 2).mul(ratios).view(paste_boxes.shape)

    paste_alpha_mask = paste_masks.sum(dim=0) > 0

    if blending:
        paste_alpha_mask = F.gaussian_blur(
            paste_alpha_mask.unsqueeze(0),
            kernel_size=(5, 5),
            sigma=[
                2.0,
            ],
        )

    # Copy-paste images:
    image = (image * (~paste_alpha_mask)) + (paste_image * paste_alpha_mask)

    # Copy-paste masks:
    masks = masks * (~paste_alpha_mask)
    non_all_zero_masks = masks.sum((-1, -2)) > 0
    masks = masks[non_all_zero_masks]

    # Do a shallow copy of the target dict
    out_target = {k: v for k, v in target.items()}

    out_target["masks"] = torch.cat([masks, paste_masks])

    # Copy-paste boxes and labels
    boxes = ops.masks_to_boxes(masks)
    out_target["boxes"] = torch.cat([boxes, paste_boxes])

    labels = target["labels"][non_all_zero_masks]
    out_target["labels"] = torch.cat([labels, paste_labels])

    # Update additional optional keys: area and iscrowd if exist
    if "area" in target:
        out_target["area"] = out_target["masks"].sum((-1, -2)).to(torch.float32)

    if "iscrowd" in target and "iscrowd" in paste_target:
        # target['iscrowd'] size can be differ from mask size (non_all_zero_masks)
        # For example, if previous transforms geometrically modifies masks/boxes/labels but
        # does not update "iscrowd"
        if len(target["iscrowd"]) == len(non_all_zero_masks):
            iscrowd = target["iscrowd"][non_all_zero_masks]
            paste_iscrowd = paste_target["iscrowd"][random_selection]
            out_target["iscrowd"] = torch.cat([iscrowd, paste_iscrowd])

    # Check for degenerated boxes and remove them
    boxes = out_target["boxes"]
    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
    if degenerate_boxes.any():
        valid_targets = ~degenerate_boxes.any(dim=1)

        out_target["boxes"] = boxes[valid_targets]
        out_target["masks"] = out_target["masks"][valid_targets]
        out_target["labels"] = out_target["labels"][valid_targets]

        if "area" in out_target:
            out_target["area"] = out_target["area"][valid_targets]
        if "iscrowd" in out_target and len(out_target["iscrowd"]) == len(valid_targets):
            out_target["iscrowd"] = out_target["iscrowd"][valid_targets]

    return image, out_target


class SimpleCopyPaste(torch.nn.Module):
    def __init__(self, blending=True, resize_interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.resize_interpolation = resize_interpolation
        self.blending = blending

    def forward(
        self, images: List[torch.Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Tensor]]]:
        torch._assert(
            isinstance(images, (list, tuple)) and all([isinstance(v, torch.Tensor) for v in images]),
            "images should be a list of tensors",
        )
        torch._assert(
            isinstance(targets, (list, tuple)) and len(images) == len(targets),
            "targets should be a list of the same size as images",
        )
        for target in targets:
            # Can not check for instance type dict with inside torch.jit.script
            # torch._assert(isinstance(target, dict), "targets item should be a dict")
            for k in ["masks", "boxes", "labels"]:
                torch._assert(k in target, f"Key {k} should be present in targets")
                torch._assert(isinstance(target[k], torch.Tensor), f"Value for the key {k} should be a tensor")

        # images = [t1, t2, ..., tN]
        # Let's define paste_images as shifted list of input images
        # paste_images = [t2, t3, ..., tN, t1]
        # FYI: in TF they mix data on the dataset level
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        output_images: List[torch.Tensor] = []
        output_targets: List[Dict[str, Tensor]] = []

        for image, target, paste_image, paste_target in zip(images, targets, images_rolled, targets_rolled):
            output_image, output_data = _copy_paste(
                image,
                target,
                paste_image,
                paste_target,
                blending=self.blending,
                resize_interpolation=self.resize_interpolation,
            )
            output_images.append(output_image)
            output_targets.append(output_data)

        return output_images, output_targets

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(blending={self.blending}, resize_interpolation={self.resize_interpolation})"
        return s
