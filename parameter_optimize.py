import optuna
import cv2
import json
import numpy as np
import os
import glob
from roi_using_old import FastHorizonDetector  # 사용자 파일

# ==========================================
# 1. 설정
# ==========================================
DATA_DIR = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o\island_x\select(200)"
# 샘플링 간격 (작을수록 정밀하지만 느려짐. 5~10 정도면 충분)
SAMPLING_STEP = 5 

# ==========================================
# 2. 핵심: Polyline을 촘촘한 점들로 변환하는 함수
# ==========================================
def get_dense_points_from_polyline(points, step=5):
    """
    드문드문 있는 polyline 점들을 이어서, step 간격마다 점을 생성함.
    points: [[x1, y1], [x2, y2], ...]
    return: [(x, y), (x+step, y'), ...]
    """
    dense_points = []
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        # x좌표 기준으로 정렬 (항상 왼쪽 -> 오른쪽으로 계산)
        if x1 > x2:
            start_x, end_x = x2, x1
            start_y, end_y = y2, y1
        else:
            start_x, end_x = x1, x2
            start_y, end_y = y1, y2
            
        dist_x = end_x - start_x
        
        # 수직선에 가까우면(x차이가 0이면) 건너뜀 (y=ax+b로 비교 불가)
        if dist_x == 0:
            continue
            
        # 두 점 사이를 step 간격으로 채움
        # 예: x=100에서 x=200까지 5씩 증가하며 y값 계산
        xs = np.arange(start_x, end_x, step)
        if len(xs) == 0: continue
        
        # 선형 보간 (Linear Interpolation) 공식: y = y1 + slope * (x - x1)
        slope = (end_y - start_y) / dist_x
        ys = start_y + slope * (xs - start_x)
        
        # 리스트에 추가
        for x, y in zip(xs, ys):
            dense_points.append((x, y))
            
    # 마지막 점도 포함
    dense_points.append(tuple(points[-1]))
    
    return dense_points

# ==========================================
# 3. RMSE 계산 및 로드 함수
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

def calculate_rmse(pred_line, dense_gt_points):
    """
    pred_line: 예측된 직선 (a, b)
    dense_gt_points: 촘촘하게 채워진 GT 점들 [(x, y), ...]
    """
    if pred_line is None:
        return 1000.0

    a_pred, b_pred, _ = pred_line
    
    error_sum = 0.0
    count = 0
    
    for (gx, gy) in dense_gt_points:
        # 예측된 y값
        py = a_pred * gx + b_pred
        
        # 오차 제곱 누적
        error_sum += (py - gy) ** 2
        count += 1
    
    if count == 0:
        return 1000.0
        
    mse = error_sum / count
    return np.sqrt(mse)

# ==========================================
# 4. Optuna Objective
# ==========================================
def objective(trial):
    params = {
        'num_regions': trial.suggest_int('num_regions', 3, 15),
        'roi_resize_factor': trial.suggest_float('roi_resize_factor', 0.1, 0.5),
        'canny_thresh1': trial.suggest_int('canny_thresh1', 10, 100),
        'canny_thresh2': trial.suggest_int('canny_thresh2', 100, 250),
        'edge_sum_thresh': trial.suggest_int('edge_sum_thresh', 50, 200),
        'hough_threshold': trial.suggest_int('hough_threshold', 30, 100),
        'scales': (1, 2, 3)
    }
    
    if params['canny_thresh1'] >= params['canny_thresh2']:
        params['canny_thresh1'] = params['canny_thresh2'] - 1

    detector = FastHorizonDetector(**params)
    
    total_rmse = 0.0
    valid_count = 0
    
    # 이미지 찾기
    img_paths = glob.glob(os.path.join(DATA_DIR, "*.jpg")) + glob.glob(os.path.join(DATA_DIR, "*.JPG"))
    
    # 속도를 위해 데이터가 많으면 일부만 랜덤 샘플링해서 튜닝 가능
    # import random
    # random.shuffle(img_paths)
    # img_paths = img_paths[:50] 

    for img_path in img_paths:
        base_name = os.path.splitext(img_path)[0]
        json_path = base_name + ".json"
        
        # 1. JSON에서 점 가져오기
        raw_points = load_gt_points_from_json(json_path)
        if raw_points is None or len(raw_points) < 2:
            continue
            
        # 2. [중요] 점들을 촘촘하게 보간하기
        dense_points = get_dense_points_from_polyline(raw_points, step=SAMPLING_STEP)
        
        # 3. 이미지 로드 및 예측
        stream = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        if img is None: continue
        
        result, _ = detector.detect(img)
        
        # 4. 오차 계산
        error = calculate_rmse(result, dense_points)
        total_rmse += error
        valid_count += 1

    if valid_count == 0:
        return 1000.0

    return total_rmse / valid_count

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best Params:", study.best_params)
    print("Best RMSE:", study.best_value)