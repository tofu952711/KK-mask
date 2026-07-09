# KK-mask

KK-mask 是一个面向 ComfyUI 的遮罩裁剪与无缝拼回插件，适用于局部重绘、局部放大、局部修复、人物/物体区域处理等工作流。插件围绕“先根据遮罩精准裁剪，再将处理结果自然拼回原图”的流程设计，尤其适合不规则遮罩区域。

## 功能特性

- **遮罩智能裁剪**：以黑白遮罩中的白色区域作为目标，自动计算有效裁剪范围。
- **不规则区域保留**：面对人物、头发、服饰、物体边缘等不规则遮罩时，会最大限度保留白色目标区域。
- **矩形/正方形输出**：在包住不规则目标后，自动补入必要背景区域，使裁剪结果符合图像处理节点常用的矩形或正方形输入要求。
- **裁剪框可复用**：输出 `crop_box`，可在后续节点中准确还原裁剪区域的位置。
- **无缝拼回**：支持 OpenCV Poisson blending，将处理后的裁剪图自然融合回原图，降低边缘接缝。
- **羽化回退**：在不需要 Poisson 融合或 OpenCV 不可用时，可使用 feather 羽化拼回。
- **批处理支持**：支持基础 batch 输入，默认使用首帧遮罩复用裁剪框，适合图像序列和视频类工作流。

## 节点列表

### KK遮罩智能裁剪

根据遮罩白色区域裁剪图片，并输出用于拼回的位置信息。

**输入**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `image` | IMAGE | 原始图片 |
| `mask` | MASK | 黑白遮罩，白色为需要裁剪的区域 |
| `threshold` | FLOAT | 遮罩阈值，高于该值的像素会被视为有效区域 |
| `padding` | INT | 在有效区域四周额外保留的背景像素 |
| `target_aspect` | 选项 | 输出比例：`auto`、`square`、`original`、`custom` |
| `custom_width` | INT | 自定义比例宽度，仅 `custom` 模式使用 |
| `custom_height` | INT | 自定义比例高度，仅 `custom` 模式使用 |
| `round_to_multiple` | 选项 | 将裁剪尺寸向上取整到指定倍数，便于适配模型尺寸要求 |
| `batch_mode` | 选项 | 批处理模式：首帧遮罩复用或逐帧计算 |

**输出**

| 输出 | 类型 | 说明 |
| --- | --- | --- |
| `cropped_image` | IMAGE | 裁剪后的图片 |
| `cropped_mask` | MASK | 同步裁剪后的遮罩 |
| `crop_box` | BOX | 裁剪区域坐标，用于拼回 |
| `box_preview` | IMAGE | 裁剪框预览图 |

### KK遮罩无缝拼回

将裁剪图按照 `crop_box` 放回原图，并对边缘进行融合。

**输入**

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `background_image` | IMAGE | 原始背景图 |
| `cropped_image` | IMAGE | 处理后的裁剪图 |
| `crop_box` | BOX | 来自 `KK遮罩智能裁剪` 的裁剪框 |
| `blend_mode` | 选项 | 融合模式：`poisson_mixed`、`poisson_normal`、`feather` |
| `feathering` | INT | 羽化半径，可减轻边缘硬切 |
| `mask_expand` | INT | 对拼回遮罩进行扩张或收缩 |
| `cropped_mask` | MASK | 可选，裁剪后的遮罩 |

**输出**

| 输出 | 类型 | 说明 |
| --- | --- | --- |
| `image` | IMAGE | 拼回后的完整图片 |
| `paste_mask` | MASK | 最终用于拼回的完整尺寸遮罩 |

## 推荐工作流

1. 使用任意分割、手绘或遮罩生成节点得到黑白 mask。
2. 将原图和 mask 接入 `KK遮罩智能裁剪`。
3. 对 `cropped_image` 进行局部重绘、放大、修复或风格处理。
4. 将处理后的图片、`cropped_mask` 和 `crop_box` 接入 `KK遮罩无缝拼回`。
5. 根据接缝情况调整 `blend_mode`、`feathering` 和 `mask_expand`。

## 融合模式建议

- `poisson_mixed`：推荐默认选项，适合多数局部替换、重绘和修复场景。
- `poisson_normal`：适合希望保留源图局部色彩和结构的场景。
- `feather`：速度快、行为直观，适合边缘差异较小或需要稳定可控结果的场景。

## 安装方法

进入 ComfyUI 的 `custom_nodes` 目录后执行：

```bash
git clone https://github.com/tofu952711/KK-mask.git
```

安装依赖：

```bash
pip install -r KK-mask/requirements.txt
```

如果使用的是整合包，请使用整合包自带的 Python 环境执行安装命令。安装完成后重启 ComfyUI。

## 依赖

- ComfyUI
- torch
- numpy
- Pillow
- opencv-python

大多数 ComfyUI 环境已经自带 torch、numpy 和 Pillow；`opencv-python` 用于 Poisson seamless clone 融合。

## 技术说明

`KK遮罩智能裁剪` 会先根据遮罩阈值检测白色区域的外接范围，再根据 padding、目标比例和尺寸倍数约束扩展裁剪框。这样即使输入是不规则遮罩，也可以得到适合后续模型处理的规整图像。

`KK遮罩无缝拼回` 的 Poisson 模式基于 OpenCV `seamlessClone`，通过泊松融合减弱裁剪边缘和背景之间的亮度、颜色与纹理突变。插件同时保留 feather 模式，用于轻量、快速或更可控的拼接场景。

## 注意事项

- 如果遮罩为空，裁剪节点会退回到整图范围。
- `single_frame` 批处理模式会逐帧计算裁剪区域，但当前 `BOX` 输出只能表示一个裁剪框；视频工作流建议优先使用默认的 `first_mask_reuse`。
- Poisson 融合更适合边缘附近存在相似背景纹理的场景；如果源图和裁剪图差异过大，可尝试提高 `feathering` 或改用 `feather` 模式。

## License

MIT
