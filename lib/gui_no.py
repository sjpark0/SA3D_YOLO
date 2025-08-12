import logging
from collections import deque

# logging.basicConfig(level=logging.WARNING)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import imageio
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import dash
from dash import Dash, Input, Output, dcc, html, State
from dash.exceptions import PreventUpdate
from .self_prompting import grounding_dino_prompt

# by seok
from ultralytics import YOLO
from torchvision.utils import save_image
from lib import sam3d
from . import utils
import PIL
import torchvision.transforms as transforms

# by young
from scipy.spatial.transform import Rotation as R

# model_yolo = YOLO('yolov8n.pt')
model_yolo = YOLO('yolov8x-seg.pt')
best_view_idx = None  # 전역 변수로 선언

def compute_pose_distance(pose1, pose2, w=0.5):
    pose1_cpu = pose1.cpu().numpy() if isinstance(pose1, torch.Tensor) else pose1
    pose2_cpu = pose2.cpu().numpy() if isinstance(pose2, torch.Tensor) else pose2
    r1, r2 = R.from_matrix(pose1_cpu[:3, :3]), R.from_matrix(pose2_cpu[:3, :3])
    angle_diff = (r1.inv() * r2).magnitude()
    trans_diff = np.linalg.norm(pose1_cpu[:3, 3] - pose2_cpu[:3, 3])
    
    return trans_diff + angle_diff

def mark_image(_img, points):
    assert(len(points) > 0)
    img = _img.copy()
    r = 10
    mark_color = np.array([255, 0, 0]).reshape(1, 1, 3)
    for i in range(len(points)):
        point = points[i]
        img[point[1]-r:point[1]+r+1, point[0]-r:point[0]+r+1] = mark_color
    return img

def draw_figure(fig, title, animation_frame=None):
    fig = px.imshow(fig, animation_frame=animation_frame)
    if animation_frame is not None:
        # fig.update_layout(sliders = [{'visible': False}])
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 33
    fig.update_layout(title_text=title, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def find_view_with_max_objects(data_dict, yolo_model):
    """
    모든 객체가 가장 잘 보이는 뷰를 선택하는 함수
    Args:
        data_dict: 학습 데이터 (images, poses, etc.)
        yolo_model: YOLO 모델 객체
    Returns:
        best_view_idx: 가장 많은 객체가 감지된 뷰의 인덱스
        max_instances: 해당 뷰에서 감지된 총 객체 수
    """
    #max_instances = 0
    #best_view_idx = 0

    #for idx in range(len(data_dict['i_train'])):
    #    img = data_dict['images'][idx, :, :, :].numpy()
    #    img = utils.to8b(img)
    #    h, w, c = img.shape

        # YOLO로 객체 감지
    #    results = yolo_model.predict(source=img, imgsz=(h, w))
    #    num_instances = len(results[0].boxes)  # 감지된 객체 수

    #    if num_instances > max_instances:
    #        max_instances = num_instances
    #        best_view_idx = idx

    #return best_view_idx, max_instances
    
    max_score = -float('inf')
    best_view_idx = 0

    for idx in range(len(data_dict['i_train'])):
        img = data_dict['images'][idx].numpy()
        img = utils.to8b(img)
        h, w, c = img.shape
        
        # YOLO 추론 (이미지 크기 자동 조정 방지)
        results = yolo_model.predict(
            source=img, 
            imgsz=(h, w),  # 원본 해상도 유지
            classes=0,  # person 클래스만 검출
            stream=False  # 단일 이미지 처리
        )
        
        # 신뢰도 및 객체 수 계산
        if len(results[0].boxes) == 0:
            continue  # 객체 미검출 시 건너뜀
            
        confidences = results[0].boxes.conf.cpu().numpy()
        num_instances = len(results[0].boxes)
        
        # 종합 점수 계산 (신뢰도 70% + 객체 수 30%)
        conf_sum = np.sum(confidences)
        score = (conf_sum * 0.7) + (num_instances * 0.3)

        # 최대 점수 갱신
        if score > max_score:
            max_score = score
            best_view_idx = idx

    return best_view_idx, max_score
    
class Sam3dNoGUI:
    def __init__(self, Seg3d, debug=False):
        ctx = {
            'num_clicks': 0, 
            'click': [], 
            'cur_img': None, 
            'btn_clear': 0, 
            'btn_text': 0, 
            'prompt_type': 'point',
            'show_rgb': False
            }
        self.ctx = ctx
        self.Seg3d = Seg3d
        self.debug = debug

        self.train_idx = 0

    def run(self):
        global best_view_idx  # 전역 변수 사용
        init_rgb = self.Seg3d.init_model()
        
        # 모든 객체가 잘 보이는 뷰를 선택
        best_view_idx, max_instances = find_view_with_max_objects(self.Seg3d.data_dict, model_yolo)

        logging.info(f"Best view with max objects: {best_view_idx} (Instances: {max_instances})")

        # 선택된 뷰에서 초기 이미지 설정
        init_rgb = self.Seg3d.data_dict['images'][best_view_idx, :, :, :].numpy()
        init_rgb = utils.to8b(init_rgb)
        self.Seg3d.predictor.set_image(init_rgb)
        
        self.ctx['cur_img'] = init_rgb
        self.run_app(sam_pred=self.Seg3d.predictor, ctx=self.ctx, init_rgb=init_rgb)


    def run_app(self, sam_pred, ctx, init_rgb):
        h, w, c = ctx['cur_img'].shape
        results = model_yolo.predict(source=ctx['cur_img'], imgsz=(h,w), classes=0)
        h2, w2 = results[0].masks.data[0].shape
        m = torch.zeros([h, w])
        for j, img in enumerate(results[0].masks.data):
            # save_image(img, f'result_{j}.png')
            m += img[(h2-h)//2:(h2+h)//2,:]

        # first person only
        # p1 0, p2 2, p3 5, p4 1, p5 4, p6 3
        idx_select = 0
        m = torch.zeros([h, w])
        img = results[0].masks.data[idx_select]
        m = img[(h2-h)//2:(h2+h)//2,:]
        self.Seg3d.confidences.append(results[0].boxes.conf[idx_select])

        masks = torch.zeros([c, h, w]).cpu().numpy()
        masks[0:3,:,:] = m.type(torch.bool).cpu().numpy()
        
        self.ctx['btn_text'] += 1
        self.ctx['text'] = None                    
        ctx['masks'] = masks
        ctx['select_mask_id'] = 0

        all_view_indices = list(range(len(self.Seg3d.data_dict['i_train'])))
        print("Available view indices:", all_view_indices)
        print("Best view index:", best_view_idx)

        # # 1. 실제 뷰 ID 리스트
        # all_view_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

        # print("valid_views:", all_view_ids)  # valid_views 리스트 출력
        # print("best_view_idx:", best_view_idx)  # best_view_idx 값 출력

        # 포즈 거리 기반 정렬 적용
        sorted_indices = [best_view_idx]
        available_indices = list(range(len(all_view_indices)))
        available_indices.remove(best_view_idx)

        # 히스토리 윈도우 초기화
        history_window = deque(maxlen=4)
        history_window.append(best_view_idx)

        # 포즈 거리 기반 정렬
        poses = self.Seg3d.data_dict['poses'][self.Seg3d.data_dict['i_train']]

        while available_indices:
            # 가중치 계산
            weighted_distances = {}
            for candidate in available_indices:
                total = 0.0
                weight_sum = 0.0
                for k, prev_idx in enumerate(reversed(history_window)):
                    weight = 0.5 ** (k+1)
                    distance = compute_pose_distance(poses[prev_idx], poses[candidate])
                    total += weight * distance
                    weight_sum += weight
                weighted_distances[candidate] = total / weight_sum  # 정규화
            # 최소 가중 거리 선택
            next_idx = min(weighted_distances.items(), key=lambda x: x[1])[0]
            # 히스토리 업데이트
            sorted_indices.append(next_idx)
            available_indices.remove(next_idx)
            history_window.append(next_idx)

        # 첫 번째 학습을 위한 정렬된 인덱스 저장
        i_train_sorted_1 = sorted_indices.copy()
        self.Seg3d.data_dict['i_train'] = i_train_sorted_1

        # start_idx = best_view_idx
        
        # # 3. best_view_idx부터 끝까지 순차적으로 진행
        # view_order = all_view_ids[start_idx:] + all_view_ids[:start_idx]

        # print("view_order:", view_order)  # best_view_idx 값 출력

        # view_id_to_index = {view_id: index for index, view_id in enumerate(all_view_ids)}
        # index_to_view_id = {index: view_id for index, view_id in enumerate(all_view_ids)}
        
        # i_train_sorted_1 = [view_id_to_index[view_id] for view_id in view_order]

        # self.Seg3d.data_dict['i_train'] = i_train_sorted_1  # 학습 순서 갱신

        # render_poses, HW, Ks 갱신
        self.Seg3d.update_render_poses()

        self.train_idx = 0
        # `train_step` 호출
        self.Seg3d.train_step(self.train_idx, sam_mask=ctx['masks'][ctx['select_mask_id']])
        self.train_idx += 1  # train_idx 증가

        while True:
            rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
            self.train_idx += 1
            self.ctx['fig_seg_rgb'] = rgb
            self.ctx['fig_sam_mask'] = sam_prompt
            self.ctx['show_rgb'] = True
            logging.info(f"Updated fig_seg_rgb and fig_sam_mask at train_idx {self.train_idx}")
            
            if is_finished:
                break

        # 학습순서선정/학습수행 플래그 설정
        self.Seg3d.vsgflag = True  # 할당 연산자

        # 첫 번째 학습에서 사용한 인덱스 저장
        first_train_indices = i_train_sorted_1.copy()
        
        # by seok: sort view list according to the confidence value
        confidences = self.Seg3d.confidences
        max_conf_idx = confidences.index(max(confidences))
        sorted_indices = [max_conf_idx]
        available_indices = list(range(len(confidences)))
        available_indices.remove(max_conf_idx)

        history_window = deque(maxlen=4)  # 최근 4개 뷰 추적
        history_window.append(max_conf_idx)

        # by young : sort view list according to the pose distances
        # 포즈 거리 기반으로 정렬된 순서에 따라 학습 진행
        poses = self.Seg3d.data_dict['poses'][self.Seg3d.data_dict['i_train']]

        while available_indices:
            # 가중치 계산
            weighted_distances = {}
            for candidate in available_indices:
                total = 0.0
                weight_sum = 0.0
                for k, prev_idx in enumerate(reversed(history_window)):
                    weight = 0.5 ** (k+1)
                    distance = compute_pose_distance(poses[prev_idx], poses[candidate])
                    total += weight * distance
                    weight_sum += weight
                weighted_distances[candidate] = total / weight_sum  # 정규화
            # 최소 가중 거리 선택
            next_idx = min(weighted_distances.items(), key=lambda x: x[1])[0]
            # 히스토리 업데이트
            sorted_indices.append(next_idx)
            available_indices.remove(next_idx)
            history_window.append(next_idx)

        # 두 번째 학습을 위한 정렬된 인덱스
        i_train_sorted_2 = sorted_indices

        # 직접 인덱스 사용 (매핑 없이)
        self.Seg3d.data_dict['i_train'] = i_train_sorted_2

        print(f"Selected Instance Confidence List: {self.Seg3d.confidences}")
        print(f"Sorted Confidence Order: {[self.Seg3d.confidences[i] for i in sorted_indices]}")
        print(f"Initial view (highest confidence): {max_conf_idx}")
        print(f"New training order: {sorted_indices}")
        
        # render_poses, HW, Ks 갱신
        self.Seg3d.update_render_poses()

        img = self.Seg3d.data_dict['images'][i_train_sorted_2[0], :, :, :].numpy()
        img = utils.to8b(img)
        h, w, c = img.shape
        results = model_yolo.predict(source=img, imgsz=(h, w), classes=0)
        h2, w2 = results[0].masks.data[0].shape
        m = torch.zeros([h, w])
        idx_select = self.Seg3d.idx_selected[i_train_sorted_2[0]]
        mask_img = results[0].masks.data[idx_select]
        m = mask_img[(h2 - h) // 2 : (h2 + h) // 2, :]

        # 마스크 이미지 저장
        # save_image(m, f'yolo_0.png')
        masks = torch.zeros([c, h, w]).cpu().numpy()
        masks[0:3, :, :] = m.type(torch.bool).cpu().numpy()

        # train_idx 초기화
        self.train_idx = 0
        # 학습 진행
        self.Seg3d.train_step(self.train_idx, sam_mask=masks)
        self.train_idx += 1

        # cross-view training
        while True:
            rgb, sam_prompt, is_finished = self.Seg3d.train_step(self.train_idx)
            self.train_idx += 1
            self.ctx['fig_seg_rgb'] = rgb
            self.ctx['fig_sam_mask'] = sam_prompt
            self.ctx['show_rgb'] = True
            logging.info(f"Updated fig_seg_rgb and fig_sam_mask at train_idx {self.train_idx}")
            if is_finished:
                break

        # 학습 체크포인트 저장
        self.Seg3d.save_ckpt()

        # by seok get rendered mask
        # 결과 마스크 계산 및 IoU 측정
        # f = open("iou.txt", "w")
        # avg_IoU = 0
        # render_poses, HW, Ks = sam3d.fetch_seg_poses(self.Seg3d.args.seg_poses, self.Seg3d.data_dict)
        # for idx in range(len(render_poses)):
        #     rgb, depth, bgmap, seg_m, dual_seg_m = self.Seg3d.render_view(idx, [render_poses, HW, Ks])
        #     tmp_rendered_mask = seg_m.detach().cpu().clone()
        #     tmp_rendered_mask[tmp_rendered_mask < 0] = 0
        #     tmp_rendered_mask[tmp_rendered_mask != 0] = 1
        #     # imageio.imwrite(f"mask_rendered_{idx:02d}.png", tmp_rendered_mask)

        #     # Ground Truth 마스크 로드 및 IoU 계산
        #     tf = transforms.ToTensor()
        #     current_view_id = sorted_view_ids[idx]
        #     m = tf(PIL.Image.open(f'../Labeling/Set11/masks/{current_view_id:02d}_p5.png'))  # 수정
        #     m_numpy = m.cpu().numpy()
        #     if m_numpy.ndim == 3 and m_numpy.shape[0] == 1:  # 단일 채널 텐서
        #         m_numpy = m_numpy.squeeze(axis=0)  # 2D 배열로 변환
        #     elif m_numpy.ndim == 2:
        #         pass  # 이미 2D 배열
        #     else:
        #         raise ValueError("Input tensor has unsupported dimensions for saving.")
        #     imageio.imwrite(f"GT_sorted_{idx:02d}.png", m_numpy)
        #     tmp_IoU = utils.cal_IoU(m.cpu(), tmp_rendered_mask.squeeze())
        #     avg_IoU += tmp_IoU
        #     f.write(f"{tmp_IoU}\n")
        # avg_IoU /= len(render_poses)
        # f.write(f"{avg_IoU}\n")
        # f.close()

        # 학습 결과를 반환
        #masked_rgb, seged_rgb = self.Seg3d.render_test()
        #fig_masked_rgb = draw_figure(masked_rgb, 'Masked RGB', animation_frame=0)
        #fig_seged_rgb = draw_figure(seged_rgb, 'Seged RGB', animation_frame=0)

        #return html.Div("Train Stage Finished! Press Ctrl+C to Exit!"), fig_masked_rgb, fig_seged_rgb
        
        

        
    
if __name__ == '__main__':
    from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
    class Sam_predictor():
        def __init__(self, device):
            sam_checkpoint = "./dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
            self.predictor = SamPredictor(self.sam)
            print('sam inited!')
            # pass

        def forward(self, points, multimask_output=True, return_logits=False):
            # self.predictor.set_image(image)
            # input_point = np.array([[x, y], [x + 1, y + 1]]) # TODO, add interactive mode
            input_point = points
            input_label = np.ones(len(input_point))

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=multimask_output,
                return_logits=return_logits
            )
            return masks
        
    image = cv2.cvtColor(cv2.imread('data/nerf_llff_data/fern/images_4/image000.png'), cv2.COLOR_BGR2RGB)
    sam_pred = Sam_predictor(torch.device('cuda'))
    sam_pred.predictor.set_image(image)
    video = np.stack(imageio.mimread('logs/llff/fern/render_train_coarse_segmentation_gui/video.rgbseg_gui.mp4'))
    gui = Sam3dGUINo(None, debug=True)
    gui.ctx['cur_img'] = image
    gui.ctx['video'] = video
    gui.run_app(sam_pred.predictor, gui.ctx, image)

