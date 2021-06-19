from .calmcar import CalmCarDataset
from mmdet.datasets.registry import DATASETS
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class CalmCarBSDV1Datasetzwz(CalmCarDataset):

    CLASSES = (
                'bicycle',
                'br_ball',
                'br_cones',
                'br_railing',
                'br_waterhorse',
                'bus',
                'car',
                'cyclist',
                'engineer_van',
                'fire_hydrant',
                'flat_car',
                'garbage_can',
                'other_barricade',
                'parkingspacelock',
                'person',
                'speed_bump',
                'stop_rod',
                'stop_sign',
                'tricycle',
                'truck',
                'warning_plate'
                )
                

    @staticmethod
    def transfer_category(category: str, classes: list):
        """根据标注文件 和 CLASSES进行类别合并
        目前有两种标注方式: 带目标状态的标注方式和不带目标状态的标注方式
        :param category:
        :return: category_id
        """
        MAPS = dict(
            bicycle=['bicycle'],
            br_ball=['br_Ball'],
            br_cones=['br_Cones'],
            br_railing=['br_Railing'],
            br_waterhorse=['br_WaterHorse'],
            bus=['bus'],
            car=['car'],
            cyclist=['cyclist'],
            engineer_van=['Engineer_Van'],
            fire_hydrant=['fire_Hydrant'],
            flat_car=['flat_Car'],
            garbage_can=['garbage_Can'],
            other_barricade=['other_Barricade'],
            parkingspacelock=['parkingSpaceLock'],
            person=['person'],
            speed_bump=['speed_Bump'],
            stop_rod=['stop_rod'],
            stop_sign=['stop_Sign'],
            tricycle=['tricycle'],
            truck=['truck'],
            warning_plate=['warning_Plate']

        )
        category_id = None
        for key, value in MAPS.items():
            if category in value:
                category_id = classes.index(key)

        return category_id

