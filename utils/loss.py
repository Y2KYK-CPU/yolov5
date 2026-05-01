# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""损失函数模块，包含目标检测、实例分割以及人体关键点检测的损失计算。"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # 用于关键点坐标损失计算

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """返回标签平滑的 BCE 目标值，用于减少过拟合；正样本：1.0 - 0.5*eps，负样本：0.5*eps。
    参考：https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """改进的 BCEWithLogitsLoss，通过 alpha 平滑减少 YOLOv5 训练中漏标签的影响。"""

    def __init__(self, alpha=0.05):
        """初始化改进的 BCEWithLogitsLoss，alpha 为平滑参数，用于减少漏标签影响。"""
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # 必须使用 nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """计算 YOLOv5 的改进 BCE 损失，减少漏标签影响，接受预测张量和目标张量，返回均值损失。"""
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # 从 logits 计算概率
        dx = pred - true  # 仅减少漏标签的影响
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss：通过 gamma 和 alpha 参数调整 BCEWithLogitsLoss，解决类别不平衡问题。"""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """初始化 FocalLoss，指定损失函数、gamma 和 alpha 参数；将损失 reduction 修改为 'none'。"""
        super().__init__()
        self.loss_fcn = loss_fcn  # 必须使用 nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # 对每个元素单独应用 FL

    def forward(self, pred, true):
        """计算预测值与目标值之间的 Focal Loss，使用改进的 BCEWithLogitsLoss。"""
        loss = self.loss_fcn(pred, true)

        # TF 实现：https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # 从 logits 计算概率
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Quality Focal Loss：通过基于预测置信度的调制因子解决类别不平衡问题。"""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """初始化 Quality Focal Loss，指定损失函数、gamma 和 alpha 参数；将 reduction 修改为 'none'。"""
        super().__init__()
        self.loss_fcn = loss_fcn  # 必须使用 nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # 对每个元素单独应用 FL

    def forward(self, pred, true):
        """计算预测值与目标值之间的 Quality Focal Loss，通过 gamma 和 alpha 调整权重。"""
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # 从 logits 计算概率
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """计算 YOLOv5 模型预测的总损失，包括分类损失、边界框损失和目标置信度损失。"""

    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        """初始化 ComputeLoss，根据 autobalance 选项决定是否自动平衡各层损失权重。"""
        device = next(model.parameters()).device  # 获取模型所在设备
        h = model.hyp  # 超参数

        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # 类别标签平滑（参考 https://arxiv.org/pdf/1902.04103.pdf 公式 3）
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # 正/负样本 BCE 目标值

        # Focal loss
        g = h["fl_gamma"]  # Focal Loss 的 gamma 参数
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() 模块
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7 各层损失权重
        self.ssi = list(m.stride).index(16) if autobalance else 0  # 步长 16 对应的层索引
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # 锚框数量
        self.nc = m.nc  # 类别数量
        self.nl = m.nl  # 检测层数量
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # 预测值，目标值
        """执行前向传播，计算分类损失、边界框损失和目标置信度损失。"""
        lcls = torch.zeros(1, device=self.device)  # 分类损失
        lbox = torch.zeros(1, device=self.device)  # 边界框损失
        lobj = torch.zeros(1, device=self.device)  # 目标置信度损失
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # 构建目标

        # 计算各层损失
        for i, pi in enumerate(p):  # 层索引，该层预测值
            b, a, gj, gi = indices[i]  # 图像索引、锚框索引、网格y/x
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # 目标置信度

            if n := b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # 提取预测的子集

                # 边界框回归
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测的边界框
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # 与目标框的 IoU
                lbox += (1.0 - iou).mean()  # IoU 损失

                # 目标置信度
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # IoU 比率作为目标置信度

                # 分类损失（仅在多类别时计算）
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)  # 目标值
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE 分类损失

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # 目标置信度损失
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # 批次大小

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """从输入目标（图像索引、类别、x、y、w、h）构建模型训练目标，返回类别、框、索引和锚框。"""
        na, nt = self.na, targets.shape[0]  # 锚框数量，目标数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # 归一化到网格空间的增益
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # 锚框索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # 追加锚框索引

        g = 0.5  # 偏移量
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m 方向
                ],
                device=self.device,
            ).float()
            * g
        )  # 偏移量列表

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy 网格增益

            # 将目标匹配到锚框
            t = targets * gain  # shape(3,n,7)
            if nt:
                # 锚框匹配：比较宽高比
                r = t[..., 4:6] / anchors[:, None]  # 宽高比
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # 筛选条件
                t = t[j]  # 过滤

                # 计算偏移量
                gxy = t[:, 2:4]  # 网格 xy
                gxi = gain[[2, 3]] - gxy  # 反向网格 xy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 解析目标
            bc, gxy, gwh, a = t.chunk(4, 1)  # (图像索引, 类别), 网格 xy, 网格 wh, 锚框索引
            a, (b, c) = a.long().view(-1), bc.long().T  # 锚框索引、图像索引、类别
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # 网格索引

            # 追加结果
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 相对网格的框坐标
            anch.append(anchors[a])  # 对应锚框
            tcls.append(c)  # 类别

        return tcls, tbox, indices, anch


class ComputeKeypointLoss:
    """计算人体关键点检测模型的总损失，包括边界框损失、目标置信度损失、分类损失以及关键点位置和可见性损失。"""

    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        """初始化关键点损失计算器。

        参数：
            model: 含有 KeypointDetect 头的 YOLOv5 姿态估计模型
            autobalance (bool): 是否自动平衡各检测层的损失权重
        """
        from models.yolo import KeypointDetect  # 避免循环导入

        device = next(model.parameters()).device  # 获取模型所在设备
        h = model.hyp  # 超参数

        # 定义各损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))
        BCEkptv = nn.BCEWithLogitsLoss()  # 关键点可见性损失

        # 类别标签平滑
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

        # Focal Loss
        g = h["fl_gamma"]
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # KeypointDetect 模块
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.BCEkptv = BCEkptv
        self.gr = 1.0
        self.hyp = h
        self.autobalance = autobalance
        self.na = m.na      # 锚框数量
        self.nc = m.nc      # 类别数量
        self.nl = m.nl      # 检测层数量
        self.nkpt = m.nkpt  # 关键点数量
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):
        """计算关键点检测的总损失。

        参数：
            p: 模型预测值列表，每个元素形状为 (bs, na, ny, nx, nc+5+nkpt*3)
            targets: 目标张量，形状为 (N, 6+nkpt*3)，列为
                     [图像索引, 类别, cx, cy, w, h, kx1, ky1, kv1, ..., kxN, kyN, kvN]

        返回：
            total_loss * batch_size: 加权总损失
            loss_items: 各损失分量的张量，用于日志记录
        """
        lcls = torch.zeros(1, device=self.device)  # 分类损失
        lbox = torch.zeros(1, device=self.device)  # 边界框损失
        lobj = torch.zeros(1, device=self.device)  # 目标置信度损失
        lkpt = torch.zeros(1, device=self.device)  # 关键点坐标损失
        lkptv = torch.zeros(1, device=self.device)  # 关键点可见性损失

        tcls, tbox, tkpts, indices, anchors = self.build_targets(p, targets)

        # 逐层计算损失
        for i, pi in enumerate(p):  # 层索引，该层预测值
            b, a, gj, gi = indices[i]  # 图像索引、锚框索引、网格y/x
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            if n := b.shape[0]:
                # 提取各部分预测值：坐标、尺寸、置信度、类别、关键点
                pxy, pwh, _, pcls, pkpts = pi[b, a, gj, gi].split(
                    (2, 2, 1, self.nc, self.nkpt * 3), 1
                )

                # 边界框回归
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # 目标置信度
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                # 分类损失（多类别时才计算）
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

                # 关键点损失
                # tkpts[i] 形状：(n, nkpt*3)，每三个值为 (kx, ky, kv)
                gt_kpts = tkpts[i]  # (n, nkpt*3)
                kv = gt_kpts[:, 2::3]  # 目标可见性掩码：shape (n, nkpt)
                # 仅对 visibility=2（真正可见）的关键点计算坐标损失，避免遮挡关键点引入噪声
                visible_mask = kv == 2

                if visible_mask.any():
                    # 预测关键点 x 坐标（归一化到网格单元）
                    pkx = pkpts[:, 0::3].sigmoid() * 2 - 0.5  # 形状 (n, nkpt)
                    # 预测关键点 y 坐标
                    pky = pkpts[:, 1::3].sigmoid() * 2 - 0.5  # 形状 (n, nkpt)
                    # 目标关键点 x/y（已归一化到网格单元）
                    gkx = gt_kpts[:, 0::3] - gi.float().unsqueeze(1)  # 相对网格偏移 x
                    gky = gt_kpts[:, 1::3] - gj.float().unsqueeze(1)  # 相对网格偏移 y

                    # 仅对可见关键点计算坐标损失（使用 MSE）
                    lkpt += (
                        F.mse_loss(pkx[visible_mask], gkx[visible_mask], reduction="mean")
                        + F.mse_loss(pky[visible_mask], gky[visible_mask], reduction="mean")
                    )

                # 关键点可见性损失（对所有已标注的关键点计算，v=1 或 v=2）
                pkv = pkpts[:, 2::3]  # 预测可见性 logits，形状 (n, nkpt)
                labeled_mask = kv >= 0  # 所有已标注的关键点（不含未标注的 -1）
                if labeled_mask.any():
                    # 可见性目标：0=不可见或遮挡(v=1)，1=真正可见(v=2)
                    tkv = (kv[labeled_mask] == 2).float()
                    lkptv += self.BCEkptv(pkv[labeled_mask], tkv)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 应用超参数权重
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lkpt *= self.hyp.get("kpt", 0.1)   # 关键点坐标损失权重（默认 0.1）
        lkptv *= self.hyp.get("kptv", 0.1)  # 关键点可见性损失权重（默认 0.1）

        bs = tobj.shape[0]  # 批次大小
        total_loss = (lbox + lobj + lcls + lkpt + lkptv) * bs
        return total_loss, torch.cat((lbox, lobj, lcls, lkpt, lkptv)).detach()

    def build_targets(self, p, targets):
        """从输入目标构建关键点模型的训练目标。

        参数：
            p: 模型各层预测列表
            targets: 目标张量，格式为 (N, 6 + nkpt*3)，各列：
                     [img_idx, cls, cx, cy, w, h, kx1, ky1, kv1, ..., kxN, kyN, kvN]

        返回：
            tcls: 各层目标类别列表
            tbox: 各层目标边界框（网格坐标）列表
            tkpts: 各层目标关键点（网格坐标 + 可见性）列表
            indices: 各层 (图像索引, 锚框索引, 网格y, 网格x) 元组列表
            anch: 各层对应锚框列表
        """
        na, nt = self.na, targets.shape[0]
        tcls, tbox, tkpts, indices, anch = [], [], [], [], []

        # gain: [1, 1, gx, gy, gx, gy, ...]  先为 6 + nkpt*3，之后按层设置网格增益
        gain = torch.ones(6 + self.nkpt * 3, device=self.device)

        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # 追加锚框索引

        g = 0.5  # 偏移量
        off = (
            torch.tensor(
                [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                device=self.device,
            ).float()
            * g
        )

        for i in range(self.nl):
            anchors_i, shape = self.anchors[i], p[i].shape
            # 设置 xy 和 wh 的网格增益
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            # 设置关键点的网格增益（奇数列为 x，偶数列为 y，v 列不缩放）
            for k in range(self.nkpt):
                gain[6 + k * 3] = shape[3]  # kx 增益（网格宽）
                gain[6 + k * 3 + 1] = shape[2]  # ky 增益（网格高）
                # gain[6 + k*3 + 2] = 1  # kv 不缩放

            t = targets * gain  # 将归一化目标映射到网格坐标
            if nt:
                # 锚框匹配
                r = t[..., 4:6] / anchors_i[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]

                # 网格偏移
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 解析目标
            # 格式：[img_idx, cls, cx, cy, w, h, kx1, ky1, kv1, ..., anchor_idx]
            bc = t[:, :2]         # (img_idx, cls)
            gxy = t[:, 2:4]       # 网格 xy（中心）
            gwh = t[:, 4:6]       # 网格 wh
            kpts_t = t[:, 6:-1]   # 关键点目标（nkpt*3）
            a = t[:, -1]          # 锚框索引

            a = a.long().view(-1)
            b, c = bc.long().T    # 图像索引，类别
            gij = (gxy - offsets).long()
            gi, gj = gij.T        # 网格 x、y 索引

            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 相对网格的框坐标
            tkpts.append(kpts_t)  # 关键点目标（含网格坐标）
            anch.append(anchors_i[a])
            tcls.append(c)

        return tcls, tbox, tkpts, indices, anch
