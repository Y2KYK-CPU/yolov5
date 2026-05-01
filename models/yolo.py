# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO 专用模块，包含目标检测、实例分割以及人体关键点检测的模型定义。

使用示例：
    $ python models/yolo.py --cfg yolov5s.yaml
    $ python models/yolo.py --cfg yolov5s-pose.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # 用于计算 FLOPs
except ImportError:
    thop = None


class Detect(nn.Module):
    """YOLOv5 检测头，用于处理输入张量并生成目标检测输出。"""

    stride = None  # 构建时计算的步长
    dynamic = False  # 强制重建网格
    export = False  # 导出模式

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """初始化 YOLOv5 检测层，指定类别数、锚框、通道数和 inplace 操作。"""
        super().__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚框的输出数量（类别 + 5 个坐标/置信度）
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 每层锚框数量
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化网格
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # 初始化锚框网格
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
        self.inplace = inplace  # 使用 inplace 操作（如切片赋值）

    def forward(self, x):
        """前向传播：处理输入，输出形状为 x(bs, 3, ny, nx, 85) 的检测结果。"""
        z = []  # 推理输出列表
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) -> x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理阶段
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # 分割模式（框 + 掩码）
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # 解码 xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # 解码 wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # 普通检测模式（仅框）
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # 解码 xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # 解码 wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """生成锚框网格，兼容 torch < 1.10 版本。"""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # 网格形状
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 兼容
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # 添加网格偏移，y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 分割检测头，继承自 Detect，增加了掩码原型层，用于实例分割任务。"""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """初始化 YOLOv5 分割头，支持掩码数量、原型数量和通道配置。"""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # 掩码数量
        self.npr = npr  # 原型数量
        self.no = 5 + nc + self.nm  # 每个锚框的输出数量
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
        self.proto = Proto(ch[0], self.npr, self.nm)  # 原型网络
        self.detect = Detect.forward

    def forward(self, x):
        """前向传播，返回检测结果和原型；根据训练/导出模式调整输出格式。"""
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class KeypointDetect(Detect):
    """YOLOv5 人体关键点检测头，继承自 Detect，在目标框预测基础上增加关键点坐标和可见性预测。

    每个锚框的输出格式：[x, y, w, h, obj, cls..., kx1, ky1, kv1, kx2, ky2, kv2, ..., kxN, kyN, kvN]
    其中 kxi/kyi 为关键点坐标，kvi 为关键点可见性（sigmoid 后输出）。
    """

    def __init__(self, nc=1, anchors=(), nkpt=17, ch=(), inplace=True):
        """初始化人体关键点检测头。

        参数：
            nc (int): 类别数量，人体关键点检测时通常为 1（仅检测人）
            anchors (tuple): 锚框列表
            nkpt (int): 关键点数量，COCO 数据集为 17 个关键点
            ch (tuple): 各输入层的通道数
            inplace (bool): 是否使用 inplace 操作
        """
        super().__init__(nc, anchors, ch, inplace)
        self.nkpt = nkpt  # 关键点数量（COCO 为 17）
        self.no = nc + 5 + nkpt * 3  # 每锚框输出数 = 类别 + 5(xywh+obj) + 关键点*(x,y,v)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层

    def forward(self, x):
        """前向传播，输出目标框及关键点坐标/可见性预测结果。"""
        z = []  # 推理输出列表
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积
            bs, _, ny, nx = x[i].shape  # 特征图尺寸
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推理阶段
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # 分割输出：坐标、尺寸、置信度/类别、关键点
                xy, wh, conf, kpts = x[i].split((2, 2, self.nc + 1, self.nkpt * 3), 4)
                xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # 解码中心坐标
                wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # 解码宽高

                # 解码关键点：x/y 使用网格偏移，可见性使用 sigmoid
                kpts[..., 0::3] = (kpts[..., 0::3].sigmoid() * 2 - 0.5 + self.grid[i][..., 0:1]) * self.stride[i]  # kx
                kpts[..., 1::3] = (kpts[..., 1::3].sigmoid() * 2 - 0.5 + self.grid[i][..., 1:2]) * self.stride[i]  # ky
                kpts[..., 2::3] = kpts[..., 2::3].sigmoid()  # 可见性置信度

                y = torch.cat((xy, wh, conf.sigmoid(), kpts), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)


class BaseModel(nn.Module):
    """YOLOv5 基础模型类，提供单尺度推理、训练及特征可视化功能。"""

    def forward(self, x, profile=False, visualize=False):
        """执行单尺度推理或训练的前向传播，支持性能分析和特征可视化选项。"""
        return self._forward_once(x, profile, visualize)  # 单尺度推理/训练

    def _forward_once(self, x, profile=False, visualize=False):
        """执行一次完整的前向传播，支持逐层性能分析和特征可视化。"""
        y, dt = [], []  # 输出列表、时间列表
        for m in self.model:
            if m.f != -1:  # 如果不是从上一层获取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从之前的层获取输入
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 执行当前层
            y.append(x if m.i in self.save else None)  # 保存该层输出（供后续层使用）
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """分析单层的性能，计算 GFLOPs、执行时间和参数量。"""
        c = m == self.model[-1]  # 是否为最后一层（inplace 修复时需要复制输入）
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """将 Conv2d() 与 BatchNorm2d() 层融合以提升推理速度。"""
        LOGGER.info("正在融合层... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                delattr(m, "bn")  # 移除批归一化层
                m.forward = m.forward_fuse  # 更新前向传播方法
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """打印模型信息，参数包括详细程度和图像尺寸，例如 info(verbose=True, img_size=640)。"""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """对模型张量应用 to()、cpu()、cuda()、half() 等变换（不包含参数和已注册的缓冲区）。"""
        self = super()._apply(fn)
        m = self.model[-1]  # 最后一层（Detect/KeypointDetect）
        if isinstance(m, (Detect, Segment, KeypointDetect)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 目标检测模型类，支持自定义配置和锚框设置。"""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """初始化 YOLOv5 检测模型，参数包括配置文件路径、输入通道数、类别数量和自定义锚框。"""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # 直接使用字典配置
        else:  # 从 yaml 文件加载
            import yaml  # 供 torch hub 使用

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # 加载模型配置字典

        # 定义模型结构
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 输入通道数
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"用 nc={nc} 覆盖 model.yaml 中的 nc={self.yaml['nc']}")
            self.yaml["nc"] = nc  # 覆盖 yaml 中的类别数
        if anchors:
            LOGGER.info(f"用 anchors={anchors} 覆盖 model.yaml 中的锚框设置")
            self.yaml["anchors"] = round(anchors)  # 覆盖 yaml 中的锚框
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 构建模型和保存列表
        self.names = [str(i) for i in range(self.yaml["nc"])]  # 默认类别名称
        self.inplace = self.yaml.get("inplace", True)

        # 构建步长和锚框
        m = self.model[-1]  # Detect() 层
        if isinstance(m, (Detect, Segment, KeypointDetect)):

            def _forward(x):
                """将输入 x 通过模型前向传播，返回处理后的输出。"""
                if isinstance(m, Segment):
                    return self.forward(x)[0]
                elif isinstance(m, KeypointDetect):
                    return self.forward(x)[0]
                else:
                    return self.forward(x)

            s = 256  # 最小步长的 2 倍
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # 前向推断步长
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # 仅执行一次

        # 初始化权重和偏置
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """执行单尺度或增强推理，可选性能分析或特征可视化。"""
        if augment:
            return self._forward_augment(x)  # 增强推理
        return self._forward_once(x, profile, visualize)  # 单尺度推理/训练

    def _forward_augment(self, x):
        """对不同尺度和翻转进行增强推理，返回合并后的检测结果。"""
        img_size = x.shape[-2:]  # 图像高度和宽度
        s = [1, 0.83, 0.67]  # 尺度列表
        f = [None, 3, None]  # 翻转方式（2=上下翻转，3=左右翻转）
        y = []  # 输出列表
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # 前向推断
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 裁剪增强尾部
        return torch.cat(y, 1), None  # 增强推理结果

    def _descale_pred(self, p, flips, scale, img_size):
        """反缩放增强推理的预测结果，调整翻转和图像尺寸的影响。"""
        if self.inplace:
            p[..., :4] /= scale  # 反缩放
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # 反上下翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # 反左右翻转
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # 反缩放
            if flips == 2:
                y = img_size[0] - y  # 反上下翻转
            elif flips == 3:
                x = img_size[1] - x  # 反左右翻转
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """裁剪 YOLOv5 增强推理的尾部张量，根据网格点和层数调整首尾张量。"""
        nl = self.model[-1].nl  # 检测层数量（P3-P5）
        g = sum(4**x for x in range(nl))  # 网格点总数
        e = 1  # 排除层数
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # 索引
        y[0] = y[0][:, :-i]  # 大目标层
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][:, i:]  # 小目标层
        return y

    def _initialize_biases(self, cf=None):
        """初始化 YOLOv5 Detect() 模块的偏置，可选使用类别频率 cf。

        参考 https://arxiv.org/abs/1708.02002 第 3.3 节。
        """
        m = self.model[-1]  # Detect() 模块
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)  # conv.bias(255) -> (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 目标置信度偏置（640 图像中 8 个目标）
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # 类别偏置
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # 保留 YOLOv5 'Model' 类名以保持向后兼容性


class SegmentationModel(DetectionModel):
    """YOLOv5 实例分割模型，继承自 DetectionModel，支持分割任务的自定义配置。"""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """初始化 YOLOv5 分割模型，参数 cfg 为配置文件，ch 为通道数，nc 为类别数，anchors 为锚框列表。"""
        super().__init__(cfg, ch, nc, anchors)


class PoseModel(DetectionModel):
    """YOLOv5 人体关键点检测（姿态估计）模型，继承自 DetectionModel，使用 KeypointDetect 检测头。"""

    def __init__(self, cfg="yolov5s-pose.yaml", ch=3, nc=None, anchors=None):
        """初始化 YOLOv5 姿态估计模型，参数 cfg 为配置文件路径，ch 为输入通道数，nc 为类别数，anchors 为自定义锚框。"""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 图像分类模型，可从配置文件或检测模型初始化。"""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """初始化 YOLOv5 分类模型，参数 cfg 为配置文件，nc 为类别数，cutoff 为截断层索引。"""
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """从 YOLOv5 检测模型构建分类模型，在 cutoff 层处截断并添加分类头。"""
        if isinstance(model, DetectMultiBackend):
            model = model.model  # 解包 DetectMultiBackend
        model.model = model.model[:cutoff]  # 主干网络部分
        m = model.model[-1]  # 最后一层
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # 输入通道数
        c = Classify(ch, nc)  # 分类头
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # 索引、来源、类型
        model.model[-1] = c  # 替换最后一层
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """从指定的 *.yaml 配置文件构建 YOLOv5 分类模型。"""
        self.model = None


def parse_model(d, ch):
    """从字典 d 解析 YOLOv5 模型，根据输入通道 ch 和模型架构配置各层。"""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # 重新定义默认激活函数，例如 Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印激活函数信息
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 每层锚框数量
    no = na * (nc + 5)  # 输出数量 = 锚框数 * (类别数 + 5)

    # 获取关键点数量（用于姿态估计模型）
    nkpt = d.get("nkpt", 0)

    layers, save, c2 = [], [], ch[-1]  # 层列表、保存列表、输出通道数
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 将字符串转换为对象
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # 将字符串参数转换为值

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 深度缩放
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 如果不是输出层
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # 重复次数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 锚框数量为整数时
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is KeypointDetect:
            # 处理关键点检测头的参数
            args.append([ch[x] for x in f])  # 追加各输入层通道数
            if isinstance(args[1], int):  # 锚框数量为整数时
                args[1] = [list(range(args[1] * 2))] * len(f)
            # args 格式：[nc, anchors, nkpt, ch]
            if len(args) < 3 or args[2] == 0:
                args.insert(2, nkpt)  # 插入关键点数量
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 构建模块
        t = str(m)[8:-2].replace("__main__.", "")  # 模块类型字符串
        np = sum(x.numel() for x in m_.parameters())  # 参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 附加索引、来源索引、类型、参数量
        LOGGER.info(f"{i:>3}{f!s:>18}{n_:>3}{np:10.0f}  {t:<40}{args!s:<30}")  # 打印层信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 追加到保存列表
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="模型配置文件路径（model.yaml）")
    parser.add_argument("--batch-size", type=int, default=1, help="所有 GPU 的总批次大小")
    parser.add_argument("--device", default="", help="CUDA 设备编号，例如 0 或 0,1,2,3 或 cpu")
    parser.add_argument("--profile", action="store_true", help="分析模型推理速度")
    parser.add_argument("--line-profile", action="store_true", help="逐层分析模型推理速度")
    parser.add_argument("--test", action="store_true", help="测试所有 yolo*.yaml 配置")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # 检查 YAML 文件
    print_args(vars(opt))
    device = select_device(opt.device)

    # 创建模型
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # 可选操作
    if opt.line_profile:  # 逐层性能分析
        model(im, profile=True)

    elif opt.profile:  # 整体前向-反向性能分析
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # 测试所有模型
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"{cfg} 中存在错误：{e}")

    else:  # 输出融合后模型摘要
        model.fuse()
