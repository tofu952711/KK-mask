import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None


def _log(message):
    print(f"[MaskSmartCropPaste] {message}")


def _image_tensor_to_pil(image):
    image = image.detach().cpu().clamp(0, 1)
    if image.dim() == 4:
        image = image[0]
    array = (image.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _mask_tensor_to_pil(mask):
    mask = mask.detach().cpu().clamp(0, 1)
    if mask.dim() == 4:
        mask = mask[0]
    if mask.dim() == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask.mean(dim=-1)
    array = (mask.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array).convert("L")


def _pil_to_image_tensor(image):
    array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _pil_to_mask_tensor(mask):
    array = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _pad_image_tensors_to_match(images, masks):
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    padded_images = []
    padded_masks = []

    for image, mask in zip(images, masks):
        height = image.shape[1]
        width = image.shape[2]
        pad_bottom = max_height - height
        pad_right = max_width - width
        padded_images.append(F.pad(image.permute(0, 3, 1, 2), (0, pad_right, 0, pad_bottom)).permute(0, 2, 3, 1))
        padded_masks.append(F.pad(mask, (0, pad_right, 0, pad_bottom)))

    return padded_images, padded_masks


def _ensure_mask_batch(mask):
    if mask.dim() == 2:
        return mask.unsqueeze(0)
    if mask.dim() == 4 and mask.shape[-1] == 1:
        return mask[..., 0]
    if mask.dim() == 4:
        return mask.mean(dim=-1)
    if mask.dim() == 3:
        return mask
    raise ValueError(f"Unsupported MASK tensor shape: {tuple(mask.shape)}")


def _match_batch_index(tensor, index):
    return tensor[index] if index < tensor.shape[0] else tensor[-1]


def _round_up(value, multiple):
    if multiple <= 1:
        return int(math.ceil(value))
    return int(math.ceil(value / multiple) * multiple)


def _fit_box_to_canvas(cx, cy, width, height, canvas_width, canvas_height):
    width = min(int(width), canvas_width)
    height = min(int(height), canvas_height)
    x1 = int(round(cx - width / 2.0))
    y1 = int(round(cy - height / 2.0))
    x1 = max(0, min(x1, canvas_width - width))
    y1 = max(0, min(y1, canvas_height - height))
    return [x1, y1, x1 + width, y1 + height]


def _create_box_preview(size, mask_pil, crop_box, detected_box):
    preview = Image.new("RGB", size, (18, 18, 18))
    mask_rgb = Image.merge("RGB", (mask_pil, mask_pil, mask_pil))
    preview = Image.blend(preview, mask_rgb, 0.82)
    draw = ImageDraw.Draw(preview)
    if detected_box is not None:
        x1, y1, x2, y2 = detected_box
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 70, 70), width=3)
    x1, y1, x2, y2 = crop_box
    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(70, 255, 120), width=3)
    return preview


def _resize_image_batch(image, size):
    return F.interpolate(
        image.permute(0, 3, 1, 2),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def _resize_mask_batch(mask, size):
    return F.interpolate(
        mask.unsqueeze(1),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _blur_mask_batch(mask, blur_radius):
    if blur_radius <= 0:
        return mask

    kernel_size = blur_radius * 2 + 1
    x = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        dtype=torch.float32,
        device=mask.device,
    )
    sigma = max(blur_radius / 3.0, 0.001)
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    kernel = (gaussian_1d[:, None] * gaussian_1d[None, :]).unsqueeze(0).unsqueeze(0)
    return F.conv2d(mask.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1).clamp(0, 1)


def _expand_mask_batch(mask, expand_pixels):
    if expand_pixels == 0:
        return mask

    kernel_size = abs(expand_pixels) * 2 + 1
    data = mask.unsqueeze(1)
    if expand_pixels > 0:
        data = F.max_pool2d(data, kernel_size=kernel_size, stride=1, padding=abs(expand_pixels))
    else:
        data = 1.0 - F.max_pool2d(1.0 - data, kernel_size=kernel_size, stride=1, padding=abs(expand_pixels))
    return data.squeeze(1).clamp(0, 1)


def _plain_paste(background, layer, mask, crop_box, feathering, mask_expand):
    x1, y1, x2, y2 = [int(v) for v in crop_box]
    crop_height = y2 - y1
    crop_width = x2 - x1
    device = background.device
    layer = layer.to(device)
    mask = mask.to(device)
    batch_size = background.shape[0]

    if layer.shape[0] == 1 and batch_size > 1:
        layer = layer.repeat(batch_size, 1, 1, 1)
    elif layer.shape[0] != batch_size:
        batch_size = min(background.shape[0], layer.shape[0])
        background = background[:batch_size]
        layer = layer[:batch_size]

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.repeat(batch_size, 1, 1)
    elif mask.shape[0] != batch_size:
        mask = mask[:batch_size]

    if layer.shape[1:3] != (crop_height, crop_width):
        layer = _resize_image_batch(layer, (crop_height, crop_width))
    if mask.shape[1:3] != (crop_height, crop_width):
        mask = _resize_mask_batch(mask, (crop_height, crop_width))

    mask = _expand_mask_batch(mask, mask_expand)
    mask = _blur_mask_batch(mask, feathering)

    result = background.clone()
    alpha = mask.unsqueeze(-1)
    result[:, y1:y2, x1:x2, :] = result[:, y1:y2, x1:x2, :] * (1.0 - alpha) + layer * alpha

    full_mask = torch.zeros((batch_size, background.shape[1], background.shape[2]), dtype=torch.float32, device=device)
    full_mask[:, y1:y2, x1:x2] = mask.detach()
    return result.cpu().clamp(0, 1), full_mask.cpu().clamp(0, 1)


def _seamless_clone_one(background, layer, mask, crop_box, mode, feathering, mask_expand):
    if cv2 is None:
        raise RuntimeError("OpenCV cv2 is not available")

    x1, y1, x2, y2 = [int(v) for v in crop_box]
    crop_width = x2 - x1
    crop_height = y2 - y1

    bg_pil = _image_tensor_to_pil(background)
    layer_pil = _image_tensor_to_pil(layer)
    mask_pil = _mask_tensor_to_pil(mask)

    if layer_pil.size != (crop_width, crop_height):
        layer_pil = layer_pil.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
    if mask_pil.size != (crop_width, crop_height):
        mask_pil = mask_pil.resize((crop_width, crop_height), Image.Resampling.BILINEAR)

    if mask_expand > 0:
        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(mask_expand * 2 + 1))
    elif mask_expand < 0:
        mask_pil = mask_pil.filter(ImageFilter.MinFilter(abs(mask_expand) * 2 + 1))

    if feathering > 0:
        blend_mask = mask_pil.filter(ImageFilter.GaussianBlur(feathering))
    else:
        blend_mask = mask_pil

    hard_mask = np.asarray(mask_pil.point(lambda p: 255 if p > 8 else 0), dtype=np.uint8)
    if hard_mask.max() == 0:
        return bg_pil, Image.new("L", bg_pil.size, 0)

    src_canvas = np.asarray(bg_pil).copy()
    src_canvas[y1:y2, x1:x2, :] = np.asarray(layer_pil)
    dst = np.asarray(bg_pil)

    full_mask = np.zeros((bg_pil.height, bg_pil.width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = hard_mask
    center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
    clone_flag = cv2.MIXED_CLONE if mode == "poisson_mixed" else cv2.NORMAL_CLONE

    cloned_bgr = cv2.seamlessClone(
        cv2.cvtColor(src_canvas, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(dst, cv2.COLOR_RGB2BGR),
        full_mask,
        center,
        clone_flag,
    )
    cloned = Image.fromarray(cv2.cvtColor(cloned_bgr, cv2.COLOR_BGR2RGB))

    if feathering > 0:
        plain = bg_pil.copy()
        plain.paste(layer_pil, (x1, y1), blend_mask)
        seam_mask = Image.new("L", bg_pil.size, 0)
        seam_mask.paste(blend_mask, (x1, y1))
        cloned = Image.composite(cloned, plain, seam_mask)

    out_mask = Image.new("L", bg_pil.size, 0)
    out_mask.paste(mask_pil, (x1, y1))
    return cloned, out_mask


class MaskSmartCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 24, "min": 0, "max": 4096, "step": 1}),
                "target_aspect": (["auto", "square", "original", "custom"], {"default": "auto"}),
                "custom_width": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
                "custom_height": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
                "round_to_multiple": (["None", "8", "16", "32", "64", "128"], {"default": "8"}),
                "batch_mode": (["single_frame", "first_mask_reuse"], {"default": "first_mask_reuse"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "crop_box", "box_preview")
    FUNCTION = "crop"
    CATEGORY = "KK遮罩智能裁剪拼回"

    def _detect_box(self, mask_pil, threshold):
        arr = np.asarray(mask_pil, dtype=np.uint8)
        ys, xs = np.where(arr > int(threshold * 255.0))
        if xs.size == 0 or ys.size == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def _target_ratio(self, mode, custom_width, custom_height, canvas_width, canvas_height, box):
        if mode == "square":
            return 1.0
        if mode == "original":
            return canvas_width / max(canvas_height, 1)
        if mode == "custom":
            return custom_width / max(custom_height, 1)
        width = max(box[2] - box[0], 1)
        height = max(box[3] - box[1], 1)
        return width / height

    def _expand_box(self, box, canvas_width, canvas_height, padding, target_aspect, custom_width, custom_height, multiple):
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(canvas_width, x2 + padding)
        y2 = min(canvas_height, y2 + padding)

        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        ratio = self._target_ratio(target_aspect, custom_width, custom_height, canvas_width, canvas_height, [x1, y1, x2, y2])

        if width / height < ratio:
            width = height * ratio
        else:
            height = width / ratio

        if multiple is not None:
            width = _round_up(width, multiple)
            height = _round_up(height, multiple)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return _fit_box_to_canvas(cx, cy, width, height, canvas_width, canvas_height)

    def crop(self, image, mask, threshold, padding, target_aspect, custom_width, custom_height, round_to_multiple, batch_mode):
        mask = _ensure_mask_batch(mask)
        batch_size = image.shape[0]
        canvas_height, canvas_width = image.shape[1:3]
        multiple = None if round_to_multiple == "None" else int(round_to_multiple)

        crop_boxes = []
        detected_boxes = []
        if batch_mode == "first_mask_reuse":
            mask_pil = _mask_tensor_to_pil(mask[0])
            if mask_pil.size != (canvas_width, canvas_height):
                mask_pil = mask_pil.resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)
            detected = self._detect_box(mask_pil, threshold)
            if detected is None:
                detected = [0, 0, canvas_width, canvas_height]
            crop_box = self._expand_box(detected, canvas_width, canvas_height, padding, target_aspect, custom_width, custom_height, multiple)
            crop_boxes = [crop_box for _ in range(batch_size)]
            detected_boxes = [detected for _ in range(batch_size)]
        else:
            for i in range(batch_size):
                mask_i = _match_batch_index(mask, i)
                mask_pil = _mask_tensor_to_pil(mask_i)
                if mask_pil.size != (canvas_width, canvas_height):
                    mask_pil = mask_pil.resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)
                detected = self._detect_box(mask_pil, threshold)
                if detected is None:
                    detected = [0, 0, canvas_width, canvas_height]
                crop_boxes.append(self._expand_box(detected, canvas_width, canvas_height, padding, target_aspect, custom_width, custom_height, multiple))
                detected_boxes.append(detected)

        first_box = crop_boxes[0]
        cropped_images = []
        cropped_masks = []
        for i in range(batch_size):
            box = crop_boxes[i]
            img_pil = _image_tensor_to_pil(image[i])
            mask_i = _match_batch_index(mask, i)
            mask_pil = _mask_tensor_to_pil(mask_i)
            if mask_pil.size != img_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.BILINEAR)
            cropped_images.append(_pil_to_image_tensor(img_pil.crop(box)))
            cropped_masks.append(_pil_to_mask_tensor(mask_pil.crop(box)))

        preview_mask = _mask_tensor_to_pil(mask[0])
        if preview_mask.size != (canvas_width, canvas_height):
            preview_mask = preview_mask.resize((canvas_width, canvas_height), Image.Resampling.BILINEAR)
        preview = _create_box_preview((canvas_width, canvas_height), preview_mask, first_box, detected_boxes[0])

        cropped_images, cropped_masks = _pad_image_tensors_to_match(cropped_images, cropped_masks)

        _log(f"MaskSmartCrop crop_box={first_box}, batch={batch_size}")
        return (
            torch.cat(cropped_images, dim=0),
            torch.cat(cropped_masks, dim=0),
            first_box,
            _pil_to_image_tensor(preview),
        )


class MaskSeamlessPaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "cropped_image": ("IMAGE",),
                "crop_box": ("BOX",),
                "blend_mode": (["poisson_mixed", "poisson_normal", "feather"], {"default": "poisson_mixed"}),
                "feathering": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
            },
            "optional": {
                "cropped_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "paste_mask")
    FUNCTION = "paste"
    CATEGORY = "KK遮罩智能裁剪拼回"

    def paste(self, background_image, cropped_image, crop_box, blend_mode, feathering, mask_expand, cropped_mask=None):
        if cropped_mask is None:
            cropped_mask = torch.ones((cropped_image.shape[0], cropped_image.shape[1], cropped_image.shape[2]), dtype=torch.float32)
        else:
            cropped_mask = _ensure_mask_batch(cropped_mask).float()

        if blend_mode == "feather":
            return _plain_paste(background_image, cropped_image, cropped_mask, crop_box, feathering, mask_expand)

        batch_size = background_image.shape[0]
        if cropped_image.shape[0] != 1:
            batch_size = min(batch_size, cropped_image.shape[0])

        result_images = []
        result_masks = []
        try:
            for i in range(batch_size):
                bg = background_image[i]
                layer = cropped_image[i] if i < cropped_image.shape[0] else cropped_image[-1]
                mask = cropped_mask[i] if i < cropped_mask.shape[0] else cropped_mask[-1]
                image_pil, mask_pil = _seamless_clone_one(bg, layer, mask, crop_box, blend_mode, feathering, mask_expand)
                result_images.append(_pil_to_image_tensor(image_pil))
                result_masks.append(_pil_to_mask_tensor(mask_pil))
            _log(f"MaskSeamlessPaste {blend_mode}, batch={batch_size}")
            return torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0)
        except Exception as exc:
            _log(f"{blend_mode} failed, fallback to feather paste: {exc}")
            return _plain_paste(background_image, cropped_image, cropped_mask, crop_box, feathering, mask_expand)


NODE_CLASS_MAPPINGS = {
    "KKMaskSmartCrop": MaskSmartCrop,
    "KKMaskSeamlessPaste": MaskSeamlessPaste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KKMaskSmartCrop": "KK遮罩智能裁剪",
    "KKMaskSeamlessPaste": "KK遮罩无缝拼回",
}
