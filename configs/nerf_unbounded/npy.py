import numpy as np

# 파일 경로 수정
file_path = 'C:/Users/jungy/Desktop/SA3D/SegmentAnythingin3D_YOLO/configs/nerf_unbounded/data/360_v2/Set1/poses_bounds.npy'

# 파일 로드
data = np.load(file_path)

# 데이터 구조 확인
print(data.shape)
print(data.dtype)
