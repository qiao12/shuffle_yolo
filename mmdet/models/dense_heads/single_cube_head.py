import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core.bbox import bbox_overlaps
from mmcv.cnn import xavier_init, constant_init, ConvModule


from mmdet.core.bbox.transforms import  delta2edges,edge2delta
from mmdet.core.cube.cube_target import cube_target_v2

from mmcv.runner import  force_fp32
from mmdet.core import (multi_apply,multiclass_nms)
from mmdet.core.post_processing.bbox_nms import multiclass_nms_cube

from ..builder import HEADS, build_loss

from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
@HEADS.register_module
class SingleCubeHead(BaseDenseHead, BBoxTestMixin):
    def __init__(self,
                 num_classes,
                 in_channels,
                 cube_num_output,
                 mlvl_sizes=[[(10, 13), (16, 30), (33, 23)],
                             [(30, 61), (62, 45), (59, 119)],
                             [(116, 90), (156, 198), (373, 326)]],
                 mlvl_strides=[8, 16, 32],
                 iou_calculator = dict(type="BboxOverlaps2D"),
                 ignore_iou_thr=0.5,
                 eps=1e-6,
                 background_label=None,
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 target_means=[0., 0.],
                 target_stds=[1.0, 1.0],
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, reduction="sum", loss_weight=1.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, reduction="sum", loss_weight=1.0),
                 loss_center=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, reduction="sum", loss_weight=1.0),    
                 loss_scale=dict(
                     type='MSELoss', reduction="sum", loss_weight=1.0),
                 loss_cube_cls=dict(
                     type='CrossEntropyLoss',use_sigmoid=True,reduction="sum",loss_weight=1.0),
                 loss_cube_reg=dict(
                     type='SmoothL1Loss', beta=1.0, reduction="sum",loss_weight=1.0),        
                 **kwargs
                 ):
        super(SingleCubeHead, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(mlvl_sizes, list)
        assert isinstance(mlvl_strides, list)
        assert len(mlvl_strides) == len(in_channels) == len(mlvl_strides)
        assert loss_cls.get('use_sigmoid', False)
        assert loss_obj.get('use_sigmoid', False)

        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.cube_num_output = cube_num_output

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self.out_channels = num_classes + 5 #num_classes + xywh + objectness
        self.ignore_iou_thr = ignore_iou_thr

        self.loss_cls = build_loss(loss_cls)
        self.loss_obj = build_loss(loss_obj)
        self.loss_center = build_loss(loss_center)
        self.loss_scale = build_loss(loss_scale)
        self.loss_cube_cls = build_loss(loss_cube_cls)
        self.loss_cube_reg = build_loss(loss_cube_reg)

        self.mlvl_anchors = self._generate_mlvl_anchors(mlvl_sizes)
        self.mlvl_strides = mlvl_strides

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.eps = eps
        self._init_layers()
    

    @staticmethod
    def lite_block(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, inp, groups=inp, kernel_size=3,
                      padding=1, stride=1, dilation=1, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, kernel_size=1, padding=0, stride=1, bias=True)
        )


    def _init_layers(self):
        """Initialize layers of the head."""
        num_anchors = [anchor.size(0) for anchor in self.mlvl_anchors]
        final_convs = []
        cube_convs = []
        for i in range(self.num_levels):
            output_dim = num_anchors[i] * self.out_channels
            cube_output_dim = num_anchors[i] * self.cube_num_output
            final_convs.append(
                self.lite_block(
                    self.in_channels[i], output_dim
                )
                # ConvModule(self.in_channels[i], output_dim,
                #            kernel_size=1, stride=1, padding=0,
                #            norm_cfg=None, act_cfg=None)
            )
            cube_convs.append(
                self.lite_block(
                    self.in_channels[i], cube_output_dim
                )
                # ConvModule(self.in_channels[i], cube_output_dim,
                #            kernel_size=1, stride=1, padding=0,
                #            norm_cfg=None, act_cfg=None)
            )
        self.final_convs = nn.ModuleList(final_convs)
        self.cube_convs = nn.ModuleList(cube_convs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, feats):
        preds = []
        cube_preds = []
        for feat, final_conv, cube_conv in zip(feats, self.final_convs, self.cube_convs):
            preds.append(final_conv(feat))
            cube_preds.append(cube_conv(feat))
        return (tuple(preds), tuple(cube_preds), )


    def _generate_mlvl_anchors(self, mlvl_sizes, device="cuda"):
        mlvl_anchors = [torch.Tensor(size).to(device) for size in mlvl_sizes]
        return mlvl_anchors

    def _generate_mlvl_grids(self, featmap_sizes, device="cuda"):
        num_levels = len(featmap_sizes)
        mlvl_grids = []

        for i in range(num_levels):
            feat_h, feat_w = featmap_sizes[i]
            grid_x = torch.arange(feat_w)
            grid_y = torch.arange(feat_h)
            grid_xx = grid_x.repeat(len(grid_y))
            grid_yy = grid_y.reshape(-1, 1).repeat(1, len(grid_x)).view(-1)     

            mlvl_grids.append(torch.stack([grid_xx, grid_yy], dim=-1).to(device))    

        return mlvl_grids  

    @force_fp32(apply_to=('mlvl_preds', ))
    def get_bboxes(self,
                   mlvl_preds,
                   mlvl_cube_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        num_levels = len(mlvl_preds)
        device = mlvl_preds[0].device
        featmap_sizes = [mlvl_preds[i].shape[-2:] for i in range(num_levels)]
        mlvl_grids = self._generate_mlvl_grids(featmap_sizes, device=device)
        
        result_list = []

        for img_id in range(len(img_metas)):
            # mlvl pred for each image
            single_mlvl_preds = [
                mlvl_preds[i][img_id].detach() for i in range(num_levels)
            ]
            single_mlvl_cube_preds = [
                mlvl_cube_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(single_mlvl_preds, single_mlvl_cube_preds, self.mlvl_anchors, mlvl_grids, 
                                                self.mlvl_strides,
                                                img_shape, scale_factor, cfg, rescale=rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           mlvl_preds,
                           mlvl_cube_preds,
                           mlvl_anchors,
                           mlvl_grids,
                           mlvl_strides,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False
                           ):
        """Transform outputs for a single batch item into labeled boxes.

        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_preds) == len(mlvl_anchors) == len(mlvl_grids) == len(mlvl_cube_preds)
        mlvl_scores = []
        mlvl_bboxes = []
        mlvl_objs = []
        mlvl_vedge_scores = []
        mlvl_vedges = []

        for pred, cube_pred, anchor, grid, stride in zip(mlvl_preds, mlvl_cube_preds, mlvl_anchors, mlvl_grids, mlvl_strides):
            feat_h, feat_w = pred.size()[-2:]
            num_anchors = anchor.size(0)
            # preds is num_anchors * (4 + 1 + num_classes) * h * w
            pred = pred.view(num_anchors, -1, feat_h, feat_w).permute(2, 3, 0, 1)
            cube_pred = cube_pred.view(num_anchors, -1, feat_h, feat_w).permute(2, 3, 0, 1)
            # print(cube_pred.shape)
            # print(grid.shape)
            # print(anchor.shape)
            
            grid_anchors = torch.cat((grid[:, None, :] * stride - anchor[None, :, :] / 2 + stride / 2,
                grid[:, None, :] * stride + anchor[None, :, :] / 2 + stride / 2), dim=-1).view(-1,4)
            # print(grid_anchors.shape)
            # bboxes cubes
            xy_pred = (torch.sigmoid(pred[..., :2]).view(-1, 2) + grid.repeat(1, num_anchors).view(-1, 2)) * stride
            wh_pred = (torch.exp(pred[..., 2:4])* anchor.expand_as(pred[..., 2:4])).reshape(-1, 2)
            bbox_pred = torch.cat([xy_pred - wh_pred / 2, xy_pred + wh_pred / 2], dim=1).view(-1, 4)
            cls_score = torch.sigmoid(pred[..., 5:]).view(-1, self.num_classes)
            obj_score = torch.sigmoid(pred[..., 4]).view(-1)

            vedges_flag_pred = cube_pred[..., :4]
            # print(cube_pred[..., 4:].shape)
            vedges_pred = cube_pred[..., 4:].contiguous().view(-1, 8)
            vedges_score = torch.sigmoid(vedges_flag_pred)
            vedges = delta2edges(grid_anchors, vedges_pred, self.target_means, self.target_stds)
            if rescale:
                if isinstance(scale_factor, float):
                    vedges /= scale_factor
                else:
                    scale_factor = torch.from_numpy(scale_factor).to(vedges.device)
                    vedges = (vedges.view(vedges.size(0), -1, 4) /
                            scale_factor).view(vedges.size()[0], -1)

            vedges_score = vedges_score.view(-1, 4)
            vedges = vedges.view(-1, 8)

            # Filtering out all predictions with conf < conf_thr
            obj_thr = cfg.get('conf_thr', -1)
            obj_inds = obj_score.ge(obj_thr).nonzero().flatten()
            bbox_pred = bbox_pred[obj_inds, :]
            cls_score = cls_score[obj_inds, :]
            obj_score = obj_score[obj_inds]
            vedges_score = vedges_score[obj_inds, :]
            vedges = vedges[obj_inds, :]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and obj_score.shape[0] > nms_pre:
                _, topk_inds = obj_score.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]
                obj_score = obj_score[topk_inds]
                vedges_score = vedges_score[topk_inds, :]
                vedges = vedges[topk_inds, :]

            mlvl_scores.append(cls_score)
            mlvl_bboxes.append(bbox_pred)
            mlvl_objs.append(obj_score)
            mlvl_vedge_scores.append(vedges_score)
            mlvl_vedges.append(vedges)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_objs = torch.cat(mlvl_objs)
        mlvl_vedge_scores = torch.cat(mlvl_vedge_scores)
        mlvl_vedges = torch.cat(mlvl_vedges)
        mlvl_cubes = torch.cat([mlvl_vedge_scores, mlvl_vedges], dim=1)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        det_bboxes, det_labels, det_cubes = multiclass_nms_cube(mlvl_bboxes, mlvl_scores,
                                                                mlvl_cubes,
                                                                cfg.score_thr,
                                                                cfg.nms,
                                                                cfg.max_per_img,
                                                                score_factors=mlvl_objs)
        return det_bboxes, det_labels, det_cubes


    def _get_target_single(self,
                           mlvl_preds,
                           mlvl_cube_preds,
                           mlvl_grids,
                           mlvl_anchors,
                           mlvl_strides,
                           image_meta,
                           gt_bboxes,
                           gt_cubes,
                           gt_labels):
        """ Get target of each image

        Args:
            mlvl_preds (list[Tensor]): List of multi level bboxs predictions in single image
            mlvl_cube_preds (list[Tensor]): List of multi level cubes predictions in single image
            mlvl_grids (list[Tensor]): List of multi level grids 
            mlvl_anchors (list[Tensor]): List of multi level anchors 
            mlvl_strides (list[int/float]): List of multi level strides
            image_meta (dict): Image meta infomation including input image shape
            gt_bboxes (Tensor): Ground truth bboxes in each image
            gt_cubes (Tensor): Ground truth cubes in each image
            gt_labels (Tensor): Ground truth labels in each image

        Returns:
            bbox_targets List[Tensor]): 
            cube_targets List[Tensor]):
            reg_weights (List[Tensor]):
            cube_reg_weights List[Tensor]):
            assigned_gt_inds (List[Tensor]):  -1 ignored, 0 negtive, >1 positive 
            assigned_labels (List[Tensor]): -1 ignored, >=0 label(0-based)
        """
        assert isinstance(mlvl_anchors, (tuple, list))
        assert isinstance(mlvl_preds, (tuple, list))
        assert isinstance(mlvl_grids, (tuple, list))
        assert len(mlvl_preds) == len(mlvl_anchors) == len(mlvl_grids) == self.num_levels

        device = gt_bboxes.device

        # Origin input image width and height
        pad_shape = image_meta['pad_shape']
        pad_h, pad_w, _ = pad_shape

        mlvl_featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_preds]

        # multi level anchors 
        mlvl_anchors_num = torch.Tensor([anchors.size(0) for anchors in mlvl_anchors]).long().to(device)#[3, 3, 3]
        mlvl_anchors_cusum = torch.cumsum(mlvl_anchors_num, dim=0).to(device)#[3, 6, 9]
        mlvl_anchors_cusum_ = torch.cat([torch.Tensor([0]).long().to(device), mlvl_anchors_cusum])#[0, 3, 6, 9]

        # multi level grids
        mlvl_grids_num = torch.Tensor([grids.size(0) for grids in mlvl_grids]).long().to(device)

        num_gts = gt_bboxes.size(0)

        # concat all level anchors to a single tensor
        flat_anchors = torch.cat(mlvl_anchors)

        # caclulate scale overlaps between anchors and gt_bboxes
        gt_cxy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / 2
        gt_wh = gt_bboxes[:, 2:4] - gt_bboxes[:, :2]
        gt_xywh = torch.cat([gt_cxy, gt_wh], dim=1)



        pesudo_gt_bboxes = torch.cat([-0.5 * gt_wh, 0.5*gt_wh], dim=1)
        pesudo_anchors = torch.cat([-0.5 * flat_anchors, 0.5 * flat_anchors], dim=1)



        overlaps = bbox_overlaps(pesudo_gt_bboxes, pesudo_anchors)

        # return results
        assigned_gt_inds = []

        bbox_targets = []

        vedges_flag = []

        vedges = []

        reg_weights = []

        vedges_flag_weights = []

        vedges_weights = []

        assigned_labels = []

        if num_gts == 0:
            for level_idx in range(self.num_levels):
                grids_num_level = mlvl_grids_num[level_idx]
                anchors_num_level = mlvl_anchors_num[level_idx]
                assigned_gt_inds_level = overlaps.new_full((grids_num_level, anchors_num_level), 0, dtype=torch.long)
                bbox_targets_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
                vedges_flag_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
                vedges_level = overlaps.new_full((grids_num_level, anchors_num_level, 8), 0)
                reg_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 2), 0)
                vedges_flag_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
                vedges_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 8), 0)
                assigned_labels_level = overlaps.new_full((grids_num_level, anchors_num_level), -1, dtype=torch.long)

                assigned_gt_inds.append(assigned_gt_inds_level)
                bbox_targets.append(bbox_targets_level)
                vedges_flag.append(vedges_flag_level)
                vedges.append(vedges_level)
                reg_weights.append(reg_weights_level)
                vedges_flag_weights.append(vedges_flag_weights_level)
                vedges_weights.append(vedges_weights_level)
                assigned_labels.append(assigned_labels_level) 

            return bbox_targets, reg_weights, vedges_flag, vedges, vedges_flag_weights, vedges_weights, assigned_gt_inds, assigned_labels

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        _, gt_argmax_overlaps = overlaps.max(dim=1)
        argmax_level = torch.stack([torch.nonzero(mlvl_anchors_cusum > argmax, as_tuple=False)[0][0] for argmax in gt_argmax_overlaps])
        gt_inds = torch.arange(0, num_gts, dtype=torch.long).to(device)

        # calculate assigner for each level
        for level_idx in range(self.num_levels):
            stride = mlvl_strides[level_idx]
            anchors = mlvl_anchors[level_idx]
            grids = mlvl_grids[level_idx]
            feat_h, feat_w = mlvl_featmap_sizes[level_idx]
            grid_anchors = torch.cat((grids[:, None, :] * stride - anchors[None, :, :] / 2 + stride / 2,
                            grids[:, None, :] * stride + anchors[None, :, :] / 2 + stride / 2), dim=-1).view(-1, 4)

            grids_num_level = mlvl_grids_num[level_idx]
            anchors_num_level = mlvl_anchors_num[level_idx]

            # initialize assigned gt inds by assume all sample is negtive
            assigned_gt_inds_level = overlaps.new_full((grids_num_level, anchors_num_level), 
                                                       0, 
                                                       dtype=torch.long)

            # initialize bbox_targets
            # initialize reg_weights
            # initialize cube_targets
            # initialize cube_reg_weights
            bbox_targets_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
            reg_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 2), 0)
            assigned_labels_level = overlaps.new_full((grids_num_level, anchors_num_level), -1, dtype=torch.long)
            vedges_flag_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
            vedges_level = overlaps.new_full((grids_num_level, anchors_num_level, 8), 0)
            vedges_flag_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
            vedges_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 8), 0)
            valid_cube_inds = overlaps.new_full((grids_num_level, anchors_num_level), 0, dtype=torch.bool)

            # whether to ignore the sample which is overlaped with groud truth bboxes
            if self.ignore_iou_thr > 0:
                # anchors = mlvl_anchors[level_idx]
                # grids = mlvl_grids[level_idx]
                # grid_anchors = torch.cat((grids[:, None, :] * stride - anchors[None, :, :] / 2 + stride / 2,
                #                           grids[:, None, :] * stride + anchors[None, :, :] / 2 + stride / 2), dim=-1).view(-1, 4)
                ovelaps_level = bbox_overlaps(gt_bboxes, grid_anchors)
                
                # for each anchor, which gt best overlaps with it
                # for each anchor, the max iou of all gts
                max_overlaps, _ = ovelaps_level.max(dim=0)
                assigned_gt_inds_level = assigned_gt_inds_level.view(-1)

                # assigne gt inds with -1 when max overlaps between sample and gt bboxes > igore_iou_thr
                assigned_gt_inds_level[max_overlaps > self.ignore_iou_thr] = -1
                assigned_gt_inds_level = assigned_gt_inds_level.view(grids_num_level, anchors_num_level)

            # assinged gt inds 
            matched_gt_inds = torch.nonzero(argmax_level == level_idx, as_tuple=False).squeeze(1)
            if matched_gt_inds.numel() > 0:
                matched_anchor_inds = gt_argmax_overlaps[matched_gt_inds] - mlvl_anchors_cusum_[level_idx]
                matched_gt_xywhs = gt_xywh[matched_gt_inds]
                matched_gt_locx = (matched_gt_xywhs[:, 0] / stride).clamp(min=0).long()
                matched_gt_locy = (matched_gt_xywhs[:, 1] / stride).clamp(min=0).long()
                matched_grid_index = matched_gt_locy * feat_w + matched_gt_locx
                assigned_gt_inds_level[matched_grid_index, matched_anchor_inds] = gt_inds[matched_gt_inds] + 1
                bbox_targets_level[matched_grid_index, matched_anchor_inds, 0] = (matched_gt_xywhs[:, 0] / stride - matched_gt_locx).clamp(self.eps, 1 - self.eps)
                bbox_targets_level[matched_grid_index, matched_anchor_inds, 1] = (matched_gt_xywhs[:, 1] / stride - matched_gt_locy).clamp(self.eps, 1 - self.eps)
                matched_gt_bbox_wh = matched_gt_xywhs[:, 2:4]
                matched_anchor = mlvl_anchors[level_idx][matched_anchor_inds]
                bbox_targets_level[matched_grid_index, matched_anchor_inds, 2:4] = torch.log((matched_gt_bbox_wh / matched_anchor).clamp(min=self.eps))
                reg_weights_level[matched_grid_index, matched_anchor_inds, 0] = 2.0 - matched_gt_bbox_wh.prod(1) / pad_w / pad_h
                reg_weights_level[matched_grid_index, matched_anchor_inds, 1] = 2.0 - matched_gt_bbox_wh.prod(1) / pad_w / pad_h
                assigned_labels_level[matched_grid_index, matched_anchor_inds] = gt_labels[matched_gt_inds]


                grid_anchors_level = torch.cat((grids[:, None, :] * stride - anchors[None, :, :] / 2 + stride / 2,
                                grids[:, None, :] * stride + anchors[None, :, :] / 2 + stride / 2), dim=-1)
                pos_proposals = grid_anchors_level[matched_grid_index, matched_anchor_inds]
                matched_gt_cubes = gt_cubes[matched_gt_inds]
                vedges_flag_level[matched_grid_index, matched_anchor_inds] = matched_gt_cubes[:, :4]
                vedges_level[matched_grid_index, matched_anchor_inds] = edge2delta(pos_proposals, matched_gt_cubes[:, 4:12],
                                    means=self.target_means, stds=self.target_stds)

                # 计算有效的cube
                # NumPos * 4
                vedges_flag_level_valid = (vedges_flag_level[matched_grid_index, matched_anchor_inds] > 0).cuda()

                # # NumPos * 1, where valid cube is 1
                valid_cube_inds[matched_grid_index, matched_anchor_inds] = torch.Tensor([item.any() for item in vedges_flag_level_valid]). \
                    cuda(device=device).bool()
                
                vedges_flag_weights_level[valid_cube_inds, :] = 1

                detail_vedges_flag = vedges_flag_level.reshape((-1, 3, 4, 1)).repeat(1, 1, 1, 2).reshape(-1, 3, 8)
                vedges_weights_level[detail_vedges_flag != 0] = 1

            vedges_flag.append(vedges_flag_level)
            vedges.append(vedges_level)
            vedges_flag_weights.append(vedges_flag_weights_level)
            vedges_weights.append(vedges_weights_level)
            assigned_gt_inds.append(assigned_gt_inds_level)
            bbox_targets.append(bbox_targets_level)
            reg_weights.append(reg_weights_level)
            assigned_labels.append(assigned_labels_level) 

        return bbox_targets, reg_weights, vedges_flag, vedges, vedges_flag_weights, vedges_weights, assigned_gt_inds, assigned_labels

    def get_targets(self, 
                    mlvl_preds_list,
                    mlvl_cube_preds_list,
                    mlvl_grids,
                    mlvl_anchors,
                    mlvl_strides,
                    image_metas,
                    gt_bboxes_list,
                    gt_cubes_list,
                    gt_labels_list):
        """
        Args:
            mlvl_preds_list (list[list[Tensor]]): List of multi level predictions in batched images
            mlvl_grids (list[Tensor]): List of multi level grids 
            mlvl_anchors (list[Tensor]): List of multi level anchors 
            mlvl_strides (list[Tuple]): List of multi level strides
            image_metas (list[dict]): List of image meta infomation in batched images
            gt_bboxes_list (list[Tensor]): List of ground truth bboxes in batched image
            gt_labels_list (list[Tensor]): List of ground truth labels in batched image
        Returns:

        """
        num_imgs = len(image_metas)
        mlvl_grids_list = [mlvl_grids] * num_imgs
        mlvl_anchors_list = [mlvl_anchors] * num_imgs
        mlvl_strides_list = [mlvl_strides] * num_imgs

        (all_bbox_targets, all_reg_weights, 
        all_vedges_flag, all_vedges, all_vedges_flag_weights, all_vedges_weights,
        all_assigned_gt_inds, all_assigned_labels) = multi_apply(
            self._get_target_single,
            mlvl_preds_list,
            mlvl_cube_preds_list,
            mlvl_grids_list,
            mlvl_anchors_list,
            mlvl_strides_list,
            image_metas,
            gt_bboxes_list,
            gt_cubes_list,
            gt_labels_list)
        # if concat:
        #     all_vedges_flag = torch.cat(all_vedges_flag, 0)
        #     all_vedges = torch.cat(all_vedges, 0)

        return all_bbox_targets, all_reg_weights, all_vedges_flag, all_vedges, all_vedges_flag_weights, all_vedges_weights, all_assigned_gt_inds, all_assigned_labels

    @force_fp32(apply_to=('preds_list', 'cube_preds_list',))
    def loss(self,
             preds_list,
             cube_preds_list,
             gt_bboxes_list,
             gt_cubes_list,
             gt_labels_list,
             img_metas,
             train_cfg,
             gt_bboxes_ignore=None):
        """ Calculate loss of YOLO

        Args:
            preds_list (list[Tensor]): List of predicted results in multiple feature maps
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image
            gt_label_list (list[Tensor]): Ground truth labels of each box
            img_metas (list[dict]): Meta info of each image
            gt_bboxes_ignore (list[Tensor]): Ground truth labels of each box.
        """
        device = preds_list[0].device
        num_levels = len(preds_list)

        featmap_sizes = [featmap.size()[-2:] for featmap in preds_list]
        mlvl_grids = self._generate_mlvl_grids(featmap_sizes, device=device)
        mlvl_anchors_num = [anchors.size(0) for anchors in self.mlvl_anchors]

        mlvl_preds_list = []
        mlvl_cube_preds_list = []
        for img_id in range(len(img_metas)):
            mlvl_preds_list.append([preds_list[level][img_id] for level in range(num_levels)])
            mlvl_cube_preds_list.append([cube_preds_list[level][img_id] for level in range(num_levels)])



        all_bbox_targets, all_reg_weights, all_vedges_flag, all_vedges, \
        all_vedges_flag_weights, all_vedges_weights, all_assigned_gt_inds, all_assigned_labels = \
            self.get_targets(mlvl_preds_list, mlvl_cube_preds_list, mlvl_grids, self.mlvl_anchors, \
            self.mlvl_strides, img_metas, gt_bboxes_list, gt_cubes_list, gt_labels_list)

        ft = torch.cuda.FloatTensor if preds_list[0].is_cuda else torch.Tensor
        lcls, lcenter, lscale, lobj, lvedges_flag, lvedges = ft([0]), ft([0]), ft([0]), ft([0]), ft([0]), ft([0])

        for mlvl_preds, mlvl_bbox_targets, mlvl_reg_weights, mlvl_cube_preds, mlvl_assigned_gt_inds, \
            mlvl_vedges_flag, mlvl_vedges, mlvl_vedges_flag_weights, mlvl_vedges_weights, mlvl_assigned_labels in \
            zip(mlvl_preds_list, all_bbox_targets, all_reg_weights, mlvl_cube_preds_list, all_assigned_gt_inds, \
            all_vedges_flag, all_vedges, all_vedges_flag_weights, all_vedges_weights, all_assigned_labels):

            for level_idx in range(self.num_levels):
                preds = mlvl_preds[level_idx].view(mlvl_anchors_num[level_idx], self.out_channels, -1).permute(2, 0, 1)
                cube_preds = mlvl_cube_preds[level_idx].view(mlvl_anchors_num[level_idx], self.cube_num_output, -1).permute(2, 0, 1)
                bbox_targets = mlvl_bbox_targets[level_idx]
                vedges_flag = mlvl_vedges_flag[level_idx]
                vedges = mlvl_vedges[level_idx]
                reg_weights = mlvl_reg_weights[level_idx]
                vedges_flag_weights = mlvl_vedges_flag_weights[level_idx]
                vedges_weights = mlvl_vedges_weights[level_idx]
                assigned_gt_inds = mlvl_assigned_gt_inds[level_idx]
                assigned_labels = mlvl_assigned_labels[level_idx]

                preds_cxy = preds[..., :2]
                preds_wh = preds[..., 2:4]
                preds_obj = preds[..., 4]
                preds_cls = preds[..., 5:]
                vedges_flag_pred = cube_preds[:, :, :4]
                vedges_pred = cube_preds[:, :, 4:12]
                pos_inds = assigned_gt_inds[assigned_gt_inds > 0]
                pos_nums = pos_inds.numel()
                if pos_nums > 0:
                    lcenter += self.loss_center(preds_cxy, bbox_targets[..., :2], weight=reg_weights, avg_factor=None)
                    lscale += self.loss_scale(preds_wh, bbox_targets[..., 2:4], weight=reg_weights, avg_factor=None)
                    lvedges_flag += self.loss_cube_cls(vedges_flag_pred, vedges_flag, weight=vedges_flag_weights, \
                                    avg_factor=None)
                    lvedges += self.loss_cube_reg(vedges_pred, vedges, weight=vedges_weights, avg_factor=None)
                    # construct classification target, and expand binary label into multi label
                    cls_weights = torch.zeros_like(assigned_labels, dtype=torch.long)
                    cls_weights[assigned_labels > -1] = 1
                    cls_weights = cls_weights[..., None].expand_as(preds_cls)
                    cls_targets = assigned_labels.new_full(preds_cls.size(), 0, dtype=torch.long)
                    inds = torch.nonzero(assigned_labels > 0, as_tuple=False)
                    cls_targets[inds[:, 0], inds[:, 1],  assigned_labels[inds[:, 0], inds[:, 1]] - 1] = 1
                    lcls += self.loss_cls(preds_cls, cls_targets, weight=cls_weights, avg_factor=None)

                obj_weights = torch.zeros_like(preds_obj)
                obj_weights[assigned_gt_inds != -1] = 1
                obj_targets = assigned_gt_inds.clamp(min=0, max=1)
                lobj += self.loss_obj(preds_obj, obj_targets, weight=obj_weights, avg_factor=None)
        return dict(loss_center=lcenter, loss_scale=lscale, loss_object=lobj, loss_cls=lcls, loss_cube_cls=lvedges_flag, loss_cube_reg=lvedges)
