# SA3D-YOLO

SA3D 코드를 개선하여 인식률을 높이기 위한 코드를 개발하고 있습니다.

<br>

# Installation

```bash
git clone https://github.com/franzkafka779/SA3D_YOLO.git
```

```bash
cd SA3D_YOLO;
```

Then install the dependencies:
```bash
conda create -n sa3d python=3.10
conda activate sa3d_yolo
pip install -r requirements.txt
```

Install SAM:
```bash
mkdir dependencies; cd dependencies
git clone https://github.com/facebookresearch/segment-anything.git 
cd segment-anything; pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Installing Grounding-DINO
```bash
cd ..
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/; pip install -e .
mkdir weights; cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Prepare Data

이곳에서 데이터, 라벨링 GT, YOLO pt 파일을 다운로드 받을 수 있습니다. 
https://drive.google.com/drive/folders/1WeVuaVRklUjtP9QMRl5RY5P26Rf9Lx22?usp=drive_link

data 파일은 압축을 해제하여 SA3D_YOLO에 세팅 \
YOLO pt파일은 다운받아 SA3D_YOLO에 위치 \
Labeling 파일은 SA3D_YOLO보다 한 단계 상위에 위치하게 함 

## Usage
- Train NeRF
  ```bash
  python run.py --config=configs/nerf_unbounded/Set13.py --stop_at=20000 --render_video --i_weights=10000
  ```
- Run SA3D in GUI
  ```bash
  python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set12.py --segment --sp_name=_gui --num_prompts=20 --render_opt=train --save_ckpt
  ```
- Render and Save Fly-through Videos
  ```bash
  python run_seg_gui.py --config=configs/nerf_unbounded/seg_Set1.py --segment --sp_name=_gui --num_prompts=20 --render_only --render_opt=video --dump_images --seg_type seg_img seg_density
  ```
