# -*- coding: utf-8 -*-
"""
Construct module for mutual-information based registration.

__author__ = Xinzhe Luo

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MI(nn.Module):
    """
    Mutual information module.

    """
    def __init__(self, dimension, num_bins=64, sample_rate=1, kernel_sigma=1, eps=1e-8, **kwargs):
        super(MI, self).__init__()
        self.dimension = dimension
        self.num_bins = num_bins
        self.sample_rate = sample_rate
        self.kernel_sigma = kernel_sigma
        self._kernel_radius = math.ceil(2 * self.kernel_sigma)
        self.eps = eps
        self.kwargs = kwargs
        self.bk_threshold = self.kwargs.pop('bk_threshold', float('-inf'))
        self.normalized = self.kwargs.pop('normalized', False)
        if self.dimension == 2:
            self.scale_mode = 'bicubic'
        elif self.dimension == 3:
            self.scale_mode = 'trilinear'
        else:
            raise NotImplementedError

    def forward(self, source, target, mask=None, **kwargs):
        """
        Compute mutual information by Parzen window estimation.

        :param source: tensor of shape [B, 1, *vol_shape]
        :param target: tensor of shape [B, 1, *vol_shape]
        :param mask: tensor of shape [B, 1, *vol_shape]
        :return:
        """
        scale = kwargs.pop('scale', 0)
        num_bins = kwargs.pop('num_bins', self.num_bins)
        assert source.shape == target.shape
        if mask is None:
            mask = torch.ones_like(source)

        image_mask = mask.to(torch.bool) & (source > self.bk_threshold) & (target > self.bk_threshold)

        if scale > 0:
            source = F.interpolate(source, scale_factor=2 ** (- scale), mode=self.scale_mode)
            target = F.interpolate(target, scale_factor=2 ** (- scale), mode=self.scale_mode)
            image_mask = F.interpolate(image_mask.to(target.dtype), scale_factor=2 ** (- scale),
                                       mode='nearest').to(torch.bool)

        B = source.shape[0]

        masked_source = [torch.masked_select(source[i], mask=image_mask[i]) for i in range(B)]
        masked_target = [torch.masked_select(target[i], mask=image_mask[i]) for i in range(B)]

        sample_mask = torch.rand_like(masked_source[0]).le(self.sample_rate)
        sampled_source = [torch.masked_select(masked_source[i], mask=sample_mask) for i in range(B)]
        sampled_target = [torch.masked_select(masked_target[i], mask=sample_mask) for i in range(B)]

        source_max_v = torch.stack([s.amax().detach() for s in sampled_source])
        source_min_v = torch.stack([s.amin().detach() for s in sampled_source])
        target_max_v = torch.stack([t.amax().detach() for t in sampled_target])
        target_min_v = torch.stack([t.amin().detach() for t in sampled_target])
        source_bin_width = (source_max_v - source_min_v) / num_bins
        source_pad_min_v = source_min_v - source_bin_width * self._kernel_radius
        target_bin_width = (target_max_v - target_min_v) / num_bins
        target_pad_min_v = target_min_v - target_bin_width * self._kernel_radius
        bin_center = torch.arange(num_bins + 2 * self._kernel_radius, dtype=source.dtype, device=source.device)

        source_bin_pos = [(sampled_source[i] - source_pad_min_v[i]) / source_bin_width[i] for i in range(B)]
        target_bin_pos = [(sampled_target[i] - target_pad_min_v[i]) / target_bin_width[i] for i in range(B)]
        source_bin_idx = [p.floor().clamp(min=self._kernel_radius,
                                          max=self._kernel_radius + num_bins - 1).detach() for p in source_bin_pos]
        target_bin_idx = [p.floor().clamp(min=self._kernel_radius,
                                          max=self._kernel_radius + num_bins - 1).detach() for p in target_bin_pos]

        source_min_win_idx = [(i - self._kernel_radius + 1).to(torch.int64) for i in source_bin_idx]
        target_min_win_idx = [(i - self._kernel_radius + 1).to(torch.int64) for i in target_bin_idx]
        source_win_idx = [torch.stack([(smwi + r) for r in range(self._kernel_radius * 2)])
                          for smwi in source_min_win_idx]
        target_win_idx = [torch.stack([(tmwi + r) for r in range(self._kernel_radius * 2)])
                          for tmwi in target_min_win_idx]

        source_win_bin_center = [torch.gather(bin_center.unsqueeze(1).repeat(1, source_win_idx[i].size(1)),
                                              dim=0, index=source_win_idx[i])
                                 for i in range(B)]
        target_win_bin_center = [torch.gather(bin_center.unsqueeze(1).repeat(1, target_win_idx[i].size(1)),
                                              dim=0, index=target_win_idx[i])
                                 for i in range(B)]

        source_win_weight = [self._bspline_kernel(source_bin_pos[i].unsqueeze(0) - source_win_bin_center[i])
                             for i in range(B)]
        target_win_weight = [self._bspline_kernel(target_bin_pos[i].unsqueeze(0) - target_win_bin_center[i])
                             for i in range(B)]

        source_bin_weight = torch.stack([torch.stack([torch.sum(source_win_idx[i].eq(idx) * source_win_weight[i], dim=0)
                                                      for idx in range(num_bins + self._kernel_radius * 2)], dim=0)
                                         for i in range(B)])

        target_bin_weight = torch.stack([torch.stack([torch.sum(target_win_idx[i].eq(idx) * target_win_weight[i], dim=0)
                                                      for idx in range(num_bins + self._kernel_radius * 2)], dim=0)
                                         for i in range(B)]) 
        source_hist = source_bin_weight.sum(-1)
        target_hist = target_bin_weight.sum(-1)
        joint_hist = torch.bmm(source_bin_weight, target_bin_weight.transpose(1, 2))

        source_density = source_hist / source_hist.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        target_density = target_hist / target_hist.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        joint_density = joint_hist / joint_hist.sum(dim=(1, 2), keepdim=True).clamp(min=self.eps)

        return source_density, target_density, joint_density

    def mi(self, source, target, mask=None, **kwargs):
        """
        (Normalized) mutual information

        :param source:
        :param target:
        :param mask:
        :return:
        """
        source_density, target_density, joint_density = self.forward(source, target, mask, **kwargs)
        source_entropy = - torch.sum(source_density * source_density.clamp(min=self.eps).log(), dim=-1)
        target_entropy = - torch.sum(target_density * target_density.clamp(min=self.eps).log(), dim=-1)
        joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2))
        if self.normalized:
            return torch.mean((source_entropy + target_entropy) / joint_entropy)
        else:
            return torch.mean(source_entropy + target_entropy - joint_entropy)

    def je(self, source, target, mask=None, **kwargs):
        """
        Joint entropy H(S, T).

        :param source:
        :param target:
        :param mask:
        :return:
        """
        _, _, joint_density = self.forward(source, target, mask, **kwargs)
        joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
        return joint_entropy

    def ce(self, source, target, mask=None, **kwargs):
        """
        Conditional entropy H(S | T) = H(S, T) - H(T).

        :param source:
        :param target:
        :param mask:
        :return:
        """
        _, target_density, joint_density = self.forward(source, target, mask, **kwargs)
        target_entropy = - torch.sum(target_density * target_density.clamp(min=self.eps).log(), dim=-1).mean()
        joint_entropy = - torch.sum(joint_density * joint_density.clamp(min=self.eps).log(), dim=(1, 2)).mean()
        return joint_entropy - target_entropy

    def _bspline_kernel(self, d):
        d /= self.kernel_sigma
        return torch.where(d.abs() < 1.,
                           (3. * d.abs() ** 3 - 6. * d.abs() ** 2 + 4.) / 6.,
                           torch.where(d.abs() < 2.,
                                       (2. - d.abs()) ** 3 / 6.,
                                       torch.zeros_like(d))
                           )
