import numpy as np



from mmdet.datasets.builder import DATASETS

from mmdet.datasets.coco import CocoDataset



@DATASETS.register_module()
class CalmCarDataset(CocoDataset):
    CLASSES = None

    @staticmethod
    def transfer_category(category: str, classes: list):
        """根据标注文件 和 CLASSES进行类别合并
        :param category:
        :return: category_id
        """
        category_id = None
        if category in classes:
            category_id = classes.index(category)

        return category_id

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] == 1280 and img_info['height'] == 1080:
                self.flag[i] = 0
            elif img_info['width'] == 1920 and img_info['height'] == 1208:
                self.flag[i] = 1
            else:
                self.flag[i] = 2

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox, 3d annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_cube_ann = []
        gt_whl = []
        gt_xyz = []
        gt_yaw = []
	

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

            ann_3d = ann.get('3d', None)
            # vertical_edge labels 4 float
            # vertical edge 4 float
            # bottom edge 4 float
            # side edge 4 float
            # whether 3d valid
            if ann_3d is None:
                gt_cube_ann.append(np.zeros((16,), dtype=np.float))
            else:
                ann_3d_array = np.zeros((16,), dtype=np.float)
                ann_3d_array[0] = int(ann_3d['left_front'] != None)
                ann_3d_array[1] = int(ann_3d['left_behind'] != None)
                ann_3d_array[2] = int(ann_3d['right_front'] != None)
                ann_3d_array[3] = int(ann_3d['right_behind'] != None)
                ann_3d_array[4:6] = 0 if ann_3d['left_front'] == None else ann_3d['left_front']
                ann_3d_array[6:8] = 0 if ann_3d['left_behind'] == None else ann_3d['left_behind']
                ann_3d_array[8:10] = 0 if ann_3d['right_front'] == None else ann_3d['right_front']
                ann_3d_array[10:12] = 0 if ann_3d['right_behind'] == None else ann_3d['right_behind']
                gt_cube_ann.append(ann_3d_array)
            
            ann_real_3d = ann.get("real_3d",None)
            if ann_real_3d is None:
                gt_whl.append(np.zeros((3,),dtype=np.float))
                gt_xyz.append(np.zeros((3,),dtype=np.float))
                gt_yaw.append(np.ones((1,), dtype=np.float)*5)
		
            else:
                ann_real_3d_array = np.zeros((3,), dtype=np.float)
                ann_pos_3d_array = np.zeros((3,), dtype=np.float)
                ann_gt_yaw = np.zeros((1,), dtype=np.float)
                ann_real_3d_array[0] = 0 if ann_real_3d["l"] == None else ann_real_3d["l"]
                ann_real_3d_array[1] = 0 if ann_real_3d["w"] == None else ann_real_3d["w"]
                ann_real_3d_array[2] = 0 if ann_real_3d["h"] == None else ann_real_3d["h"]
                ann_pos_3d_array[0] = 0 if ann_real_3d["x"] == None else ann_real_3d["x"]
                ann_pos_3d_array[1] = 0 if ann_real_3d["y"] == None else ann_real_3d["y"]
                ann_pos_3d_array[2] = 0 if ann_real_3d["z"] == None else ann_real_3d["z"]
                ann_gt_yaw[0] = 0 if ann_real_3d["yaw"] == None else ann_real_3d["yaw"]
                gt_whl.append(ann_real_3d_array)
                gt_xyz.append(ann_pos_3d_array) 
                gt_yaw.append(ann_gt_yaw)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if gt_cube_ann:
            gt_cube_ann = np.array(gt_cube_ann, dtype=np.float32)
        else:
            gt_cube_ann = np.zeros((0, 16), dtype=np.float32)
        
        if gt_whl:
            gt_whl = np.array(gt_whl, dtype=np.float32)
        else:
            gt_whl = np.zeros((0,3), dtype=np.float32)

        if gt_xyz:
            gt_xyz = np.array(gt_xyz, dtype=np.float32)
        else:
            gt_xyz = np.zeros((0,3), dtype=np.float32)

        if gt_yaw:
            gt_yaw = np.array(gt_yaw, dtype=np.float32)
        else:
            gt_yaw = np.zeros((0, 1), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            cubes=gt_cube_ann,
            masks=gt_masks_ann,
            seg_map=seg_map,
            whl=gt_whl,
            xyz=gt_xyz,
            yaw=gt_yaw)

        return ann

