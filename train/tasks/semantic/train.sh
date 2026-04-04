# python train.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset_for_debug --arch_cfg config/arch/arch_cfg.yaml --data_cfg config/labels/semantic-kitti.yaml / --gpu 0 
# nohup python train.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset  --arch_cfg config/arch/arch_cfg.yaml --data_cfg config/labels/semantic-kitti.yaml / --pretrained logs/2025-3-26-01:01/ --gpu 0 > ./logs/20241224/out.log 2>&1 & 
# nohup python train.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset  --arch_cfg config/arch/arch_cfg.yaml --data_cfg config/labels/semantic-kitti.yaml /  --gpu 0 > ./logs/20241224/out.log 2>&1 & 
# python train.py -d /mnt/nas/Daten/kitti_daten/KITTI/segmentation/dataset  --arch_cfg config/arch/arch_cfg.yaml --data_cfg config/labels/semantic-kitti.yaml /  --gpu 0

# training bis etwa 120 epoches
python train.py -d /mnt/c/Users/hxm/Documents/projects/phd/daten/KITTI/segmentation/dataset  --arch_cfg config/arch/arch_cfg.yaml --data_cfg config/labels/semantic-kitti.yaml /  --gpu 0 > ./logs/20260402/out.log 

# training_debug with debaug_dataset
# python train.py -d /mnt/c/Users/hxm/Documents/projects/phd/code/dataset_for_debug/  --arch_cfg config/arch/arch_cfg.yaml --data_cfg config/labels/semantic-kitti.yaml /  --gpu 0 > ./logs/20260402/out_200_finetune_on_accum_1.log 