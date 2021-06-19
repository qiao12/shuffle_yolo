import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmcv.runner import (auto_fp16,  force_fp32)

from mmdet.core.bbox.transforms import  delta2edges
from mmdet.core.cube.cube_target import cube_target_v2


from mmcv.cnn import ConvModule
from ..builder import build_loss,HEADS



@HEADS.register_module
class CCCubeHead(nn.Module):
    def __init__(self,
                 num_convs=0,
                 num_fcs=2,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_output=12,
                 target_means=[0., 0.],
                 target_stds=[1.0, 1.0],
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_reg=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(CCCubeHead, self).__init__()
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.conv_out_channesl = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_output = num_output
        self.target_means = target_means
        self.target_stds = target_stds
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)

        in_channels *= self.roi_feat_area

        # add convs and fcs
        self.convs, self.fcs, last_layer_dim = \
            self._add_conv_fc(
                self.num_convs, self.num_fcs, self.in_channels)

        self.output_fc = nn.Linear(last_layer_dim, self.num_output)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_fc(self,
                     num_branch_convs,
                     num_branch_fcs,
                     in_channels):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        for module_list in [self.fcs, self.output_fc]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        # convs
        if self.num_convs > 0:
            for conv in self.convs:
                x = conv(x)

        if self.num_fcs > 0:
            x = x.flatten(1)
            for fc in self.fcs:
                x = self.relu(fc(x))

        outputs = self.output_fc(x)
        return outputs

    def get_target(self, sampling_results, gt_cubes,
                   rcnn_train_cfg):

        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        cube_targets = cube_target_v2(pos_proposals, pos_assigned_gt_inds,
                                      gt_cubes, rcnn_train_cfg,
                                      target_means=self.target_means,
                                      target_stds=self.target_stds)
        return cube_targets

    def get_target_v2(self, sampling_results, gt_cubes,
                      rcnn_train_cfg):

        pos_proposals = [res.vehicle_bboxes(max_label=3) for res in sampling_results]
        pos_assigned_gt_inds = [
            res.vehicle_assigned_gt_inds(max_label=3) for res in sampling_results
        ]

        cube_targets = cube_target_v2(pos_proposals, pos_assigned_gt_inds,
                                      gt_cubes, rcnn_train_cfg)

        return cube_targets

    @force_fp32(apply_to=('cube_pred',))
    def loss(self,
             cube_pred,
             vedges_flag, vedges,
             reduction_override=None):
        losses = dict()
        num_pos = cube_pred.size(0)

        vedges_flag_weights = cube_pred.new_zeros(num_pos, 4)
        vedges_weights = cube_pred.new_zeros(num_pos, 8)

        # 计算有效的cube
        # NumPos * 4
        vedges_flag = (vedges_flag > 0).cuda()

        # NumPos * 1, where valid cube is 1
        valid_cube_inds = torch.Tensor([item.any() for item in vedges_flag]). \
            cuda(device=cube_pred.device).bool()
        vedges_flag_weights[valid_cube_inds, :] = 1

        detail_vedges_flag = vedges_flag.reshape((-1, 4, 1)).repeat(1, 1, 2).reshape(-1, 8)
        vedges_weights[detail_vedges_flag != 0] = 1

        vedges_flag_pred = cube_pred[:, :4]
        vedges_pred = cube_pred[:, 4:12]

        # valid_avg_factor = torch.nonzero(valid_cube_inds).numel()
        # avg_factor = valid_avg_factor if valid_avg_factor > 0 else None
        avg_factor = None
        if vedges_flag.numel() > 0:
            losses['loss_cube_cls'] = self.loss_cls(
                vedges_flag_pred,
                vedges_flag.long(),
                vedges_flag_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override
            )

        if vedges.numel() > 0:
            losses['loss_cube_vedge'] = self.loss_reg(
                vedges_pred,
                vedges,
                vedges_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override
            )
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_cubes(self,
                      cube_pred,
                      rois,
                      scale_factor,
                      rescale=False,
                      cfg=None):
        vedges_flag_pred = cube_pred[:, :4]
        vedges_pred = cube_pred[:, 4:]

        vedges_score = F.sigmoid(vedges_flag_pred)
        vedges = delta2edges(rois[:, 1:], vedges_pred,
                             self.target_means, self.target_stds)

        if rescale:
            if isinstance(scale_factor, float):
                vedges /= scale_factor
            else:
                scale_factor = torch.from_numpy(scale_factor).to(vedges.device)
                vedges = (vedges.view(vedges.size(0), -1, 4) /
                          scale_factor).view(vedges.size()[0], -1)

        # return vedges_score, vedges
        return torch.cat([vedges_score, vedges], dim=1)
