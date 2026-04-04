### istall conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
### install
conda create -n FPSNet python=3.10
conda activate FPSNet
cd /ROOT/
python -m pip install numpy torchvision matplotlib tensorflow scipy torch vispy opencv_python opencv_contrib_python pillow PyYAML wandb
apt-get install -yqq  build-essential ninja-build \
  python3-dev python3-pip apt-utils curl git cmake unzip autoconf autogen \
  libtool mlocate zlib1g-dev python3-numpy python3-wheel wget \
  software-properties-common openjdk-8-jdk libpng-dev  \
  libxft-dev ffmpeg python3-pyqt5.qtopengl
### Dataset
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/

### check tensorboard
%tensorboard --logdir ./logs/2024-07-23-13:45/tb_iter --port 2123
###### path must be abs-path
tensorboard --logdir=/mnt/nas/code/Solingen_MOT/range/fpsnet/train/tasks/semantic/logs/2024-7-23-13:45/tb_epoch --port 2124

##### check tensorboard
https://github.com/tensorflow/tensorboard/blob/master/README.md#my-tensorboard-isnt-showing-any-data-whats-wrong
![alt text](image.png)
##### val result from 20240805
![alt text](image-1.png)



### train model

### test validieren

1. sh test.sh 
set log_path, model_path
2. beachten die Daten  Struktur
└── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
3. apt install zip , falls kein zip vorhanden
4. cd /to/ziel_path
5. zip -r test.zip sequences/   , -r表示递归所有子目录，否则之压缩一个空文件夹
6. sh subtest.sh 校验文件结构
7. codelab 提交


