import torch

from ..utils import multi_apply
from mmdet.core.bbox.transforms import edge2delta


def cube_target_v2(pos_proposals_list,
                pos_assigned_gt_inds_list,
                gt_cubes_list,
                cfg,
                target_means=[.0, .0],
                target_stds=[1.0, 1.0],
                concat=True):
    vedges_flag, vedges = multi_apply(
        cube_target_single_v2,
        pos_proposals_list,
        pos_assigned_gt_inds_list,
        gt_cubes_list,
        cfg=cfg,
        target_means=target_means,
        target_stds=target_stds,
    )
    if concat:
        vedges_flag = torch.cat(vedges_flag, 0)
        vedges = torch.cat(vedges, 0)
    return vedges_flag, vedges


def cube_target_single_v2(pos_proposals, pos_assigned_gt_inds, gt_cubes, cfg,
                       target_means=[.0, .0,],
                       target_stds=[1.0, 1.0]
                       ):
    num_pos = pos_proposals.size(0)
    vedges_flag = pos_proposals.new_zeros(num_pos)
    vedges = pos_proposals.new_zeros(num_pos, 8)

    if num_pos > 0:

        pos_gt_cubes = gt_cubes[pos_assigned_gt_inds]
        vedges_flag = pos_gt_cubes[:, :4]
        vedges = edge2delta(pos_proposals, pos_gt_cubes[:, 4:12],
                            means=target_means, stds=target_stds)
    return vedges_flag, vedges


def whl_target_v1(pos_proposals_list,
                pos_assigned_gt_inds_list,
                gt_whl_list,
                cfg,
                target_means=[.0, .0],
                target_stds=[1.0, 1.0],
                concat=True):

    vol_flag, vol = multi_apply(
        whl_target_single_v2,
        pos_proposals_list,
        pos_assigned_gt_inds_list,
        gt_whl_list,
        cfg=cfg,
        target_means=target_means,
        target_stds=target_stds,
    )

    if concat:
        vol = torch.cat(vol, 0)

    return vol


def whl_target_single_v2(pos_proposals, pos_assigned_gt_inds, gt_whl,cfg,
                       target_means=[.0, .0,],
                       target_stds=[1.0, 1.0]):
    num_pos = pos_proposals.size(0)
    vol_flag = pos_proposals.new_zeros(num_pos)
    vol = pos_proposals.new_zeros(num_pos, 3)

    if num_pos > 0:
        pos_gt_whl = gt_whl[pos_assigned_gt_inds]
        vol_flag = pos_gt_whl[:, :1]
        vol = pos_gt_whl
    return vol_flag, vol

