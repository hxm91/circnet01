# infer 
# to get prediction of all sequences
# nohup python infer.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset -l ./logs/2024-10-17-18:35/infer -m ./logs/2024-10-17-18:35/ -s True --gpu 0 > ./logs/2024-10-17-18:35/1025/inferout.log 2>&1 & 
# python infer.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset -l ./logs/2024-9-18-20:50/infer -m ./logs/2024-9-18-20:50/ -s True --gpu 0
## evaluate
# python evaluate_iou.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset -p logs/2024-10-17-18:35/infer
# python infer.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset -l ./logs/2024-9-12-18:03/infer -m ./logs/2024-9-12-18:03/ -s True --gpu 0 


python infer.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset -l ./logs/2025-5-02-13:03/debug -m ./logs/2025-5-02-13:03/ -s True --gpu 0 
# python evaluate_iou.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset -p logs/2025-5-02-13:03/infer