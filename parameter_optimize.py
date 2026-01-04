import optuna
import cv2
import json
import numpy as np
import os
import glob
import sys

# [중요] 수정된 모듈 이름으로 import (파일명을 roi_using_ver1.py로 가정)
try:
    from roi_using_ver1 import FastHorizonDetector
except ImportError:
    # 만약 파일명이 roi_using.py라면 아래 주석 해제 후 사용
    from roi_using import FastHorizonDetector 
    # print("roi_using_ver1 not found, trying roi_using...")

# ==========================================
# 1. 설정
# ==========================================
# [설정] 데이터셋 경로를 사용자 환경에 맞게 수정하세요.
DATA_DIR = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o\island_x\select(200)"
SAMPLING_STEP = 5 

# ==========================================
# 2. 유틸리티: Polyline -> Dense Points 변환
# ==========================================
def get_dense_points_from_polyline(points, step=5):
    """
    GT(정답)가 폴리라인(여러 점)일 경우, 이를 촘촘한 점들의 집합으로 변환
    """
    dense_points = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        # x좌표 기준 정렬
        if x1 > x2:
            start_x, end_x = x2, x1
            start_y, end_y = y2, y1
        else:
            start_x, end_x = x1, x2
            start_y, end_y = y1, y2
            
        dist_x = end_x - start_x
        if dist_x == 0: continue
            
        xs = np.arange(start_x, end_x, step)
        if len(xs) == 0: continue
        
        slope = (end_y - start_y) / dist_x
        ys = start_y + slope * (xs - start_x)
        
        for x, y in zip(xs, ys):
            dense_points.append((x, y))
            
    dense_points.append(tuple(points[-1]))
    return dense_points

# ==========================================
# 3. RMSE 계산 함수
# ==========================================
def load_gt_points_from_json(json_path):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except: return None

    for shape in data.get('shapes', []):
        if shape.get('label') == 'horizon':
            return shape.get('points')
    return None

def calculate_rmse(pred_result, dense_gt_points):
    """
    pred_result: (a, b, full_line_pts)
    dense_gt_points: [(x, y), ...]
    """
    if pred_result is None:
        return 1000.0

    # roi_using_ver1은 (a, b, points)를 반환함
    # RMSE 계산은 전체적인 추세선(a, b)과의 거리를 측정하여 평가 (간단하고 강건함)
    a_pred, b_pred, _ = pred_result
    
    error_sum = 0.0
    count = 0
    
    for (gx, gy) in dense_gt_points:
        # 예측된 직선(y = ax + b) 위에서의 y값
        py = a_pred * gx + b_pred
        error_sum += (py - gy) ** 2
        count += 1
    
    if count == 0:
        return 1000.0
        
    mse = error_sum / count
    return np.sqrt(mse)

# ==========================================
# 4. Optuna Objective (변수 변경됨)
# ==========================================
def objective(trial):
    # -----------------------------------------------------------
    # [최적화 대상 파라미터 4가지]
    # -----------------------------------------------------------
    # 1. 길이 가중치 (0.0 ~ 1.0)
    score_weight_len = trial.suggest_float('score_weight_len', 0.0, 1.0)
    
    # 2. 편차 가중치 (0.0 ~ 1.0)
    # 두 가중치의 합이 1.0이 되도록 강제할 수도 있지만, 여기선 독립적으로 탐색하게 둠
    score_weight_std = trial.suggest_float('score_weight_std', 0.0, 1.0)
    
    # 3. 상위 N개 컨투어 후보 (3 ~ 10 정도가 적당)
    top_n_contours = trial.suggest_int('top_n_contours', 2, 5)
    
    # 4. 각도 표준편차 임계값 (5.0 ~ 50.0)
    angle_std_thresh = trial.suggest_float('angle_std_thresh', 10.0, 50.0)
    
    # 고정 파라미터 (이전 최적화 값이나 기본값 사용)
    fixed_params = {
        'num_regions': 4,        # 예시값
        'roi_resize_factor': 0.2023,
        'canny_thresh1': 76,
        'canny_thresh2': 147,
        'edge_sum_thresh': 148,
        'scales': (1, 2, 3)
    }

    # 파라미터 통합
    params = fixed_params.copy()
    params.update({
        'score_weight_len': score_weight_len,
        'score_weight_std': score_weight_std,
        'top_n_contours': top_n_contours,
        'angle_std_thresh': angle_std_thresh
    })
    
    # Detector 초기화
    detector = FastHorizonDetector(**params)
    
    total_rmse = 0.0
    valid_count = 0
    
    # 이미지 로드
    img_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.JPG"))
    
    # 데이터가 너무 많으면 랜덤 샘플링 (속도 향상용, 필요시 주석 해제)
    # import random
    # random.shuffle(img_paths)
    # img_paths = img_paths[:50] 

    for img_path in img_paths:
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + ".json"
        
        # GT 로드
        raw_points = load_gt_points_from_json(json_path)
        if raw_points is None or len(raw_points) < 2:
            continue
            
        # Dense Points 생성
        dense_points = get_dense_points_from_polyline(raw_points, step=SAMPLING_STEP)
        
        # 이미지 읽기
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: continue
        
        # 예측
        result = detector.detect(img)
        
        # RMSE 계산 (result가 None이면 1000.0 반환됨)
        error = calculate_rmse(result, dense_points)
        total_rmse += error
        valid_count += 1

    if valid_count == 0:
        return 1000.0

    return total_rmse / valid_count

if __name__ == "__main__":
    # n_trials 횟수는 시간 여유에 따라 조절 (50~100회 추천)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("\n=========================================")
    print("Optimization Finished")
    print("=========================================")
    print("Best RMSE:", study.best_value)
    print("Best Params:", study.best_params)
    
    # 결과 해석 팁 출력
    best = study.best_params
    print("\n[Result Interpretation]")
    print(f"- Length Weight: {best['score_weight_len']:.2f}")
    print(f"- StdDev Weight: {best['score_weight_std']:.2f}")
    if best['score_weight_len'] > best['score_weight_std']:
        print("-> '길이(Length)'가 더 중요한 요소로 작용했습니다.")
    else:
        print("-> '직선성(Low StdDev)'이 더 중요한 요소로 작용했습니다.")