import cv2
import numpy as np
import os
import glob
import math

# custom_canny 모듈 import (같은 폴더에 위치해야 함)
try:
    from custom_canny import GradientCannyDetector
except ImportError:
    print("Warning: custom_canny.py not found. Using default Canny.")
    GradientCannyDetector = None

class FastHorizonDetector:
    def __init__(
            self,
            num_regions=4,
            roi_resize_factor=0.2023,
            scales=(1, 2, 3),  # 기존 Multi-scale 유지
            canny_thresh1=76,
            canny_thresh2=147,
            edge_sum_thresh=148, # Multi-scale 합산 임계값
            # [New] 조건 필터링용 변수
            top_n_contours=5,       # 길이 상위 N개 후보 검사
            angle_std_thresh=20.0,  # Angle 표준편차 허용치 (낮을수록 일직선/일정한 패턴)
            target_angle=90.0       # (옵션) 수평선이 가져야 할 이상적인 Gradient 각도 (보통 수직방향 90/270)
    ):
        self.num_regions = num_regions
        self.roi_resize_factor = roi_resize_factor
        self.scales = scales
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.edge_sum_thresh = edge_sum_thresh
        
        # 필터링 파라미터
        self.top_n_contours = top_n_contours
        self.angle_std_thresh = angle_std_thresh
        self.target_angle = target_angle

        # Custom Detector 초기화
        if GradientCannyDetector:
            # 내부 blur_ksize는 5로 설정하되, 외부에서 medianBlur된 이미지가 들어오므로 
            # custom_canny 내부의 GaussianBlur 영향은 미미하거나 보정 역할을 함
            self.canny_detector = GradientCannyDetector(ksize=3, blur_ksize=5)
        else:
            self.canny_detector = None

    # -----------------------------
    # Public API
    # -----------------------------
    def detect(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]

        # 1. ROI 검출 (기존 로직)
        roi_y0, roi_y1 = self._detect_roi_vertical_range(img_bgr)
        if roi_y0 >= roi_y1: # 예외 처리
            roi_y0 = int(h * 0.3)
            roi_y1 = int(h * 0.7)

        roi_img = img_bgr[roi_y0:roi_y1, :]

        # 2. [Modified] Multi-scale Edge & Angle Map 생성
        edge_combined, angle_combined = self._multi_scale_edge_map(roi_img)
        
        if edge_combined is None:
            return None

        # 3. [Modified] 조건(길이, Angle 편차)을 반영한 수평선 추정
        result = self._estimate_horizon_geometric(edge_combined, angle_combined, roi_y0, w, h)
        return result

    # -----------------------------
    # 1. ROI detection (기존 유지)
    # -----------------------------
    def _detect_roi_vertical_range(self, img_bgr):
        # (기존 코드와 로직 동일)
        h, w = img_bgr.shape[:2]
        scale = self.roi_resize_factor
        if h < 50 or w < 50: return 0, h

        small_h = max(1, int(h * scale))
        small_w = max(1, int(w * scale))
        img_small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

        N = self.num_regions
        if N < 2: return 0, h
        step = small_h // (N + 1)
        region_height = step * 2
        regions = []
        for i in range(N):
            y0 = i * step
            y1 = y0 + region_height
            if y0 >= small_h: break
            y1 = min(y1, small_h)
            if y1 - y0 < 5: continue
            regions.append((y0, y1))
        if len(regions) < 2: return 0, h

        means, covs = [], []
        eps = 1e-6
        for (y0, y1) in regions:
            region = img_small[y0:y1, :]
            pixels = region.reshape(-1, 3).astype(np.float32)
            means.append(np.mean(pixels, axis=0))
            cov = np.cov(pixels, rowvar=False)
            if cov.shape == (): cov = np.eye(3, dtype=np.float32)
            covs.append(cov + eps * np.eye(3, dtype=np.float32))

        best_dist = -1.0
        best_pair = (0, 1)
        for i in range(len(regions) - 1):
            m1, m2 = means[i], means[i + 1]
            S1, S2 = covs[i], covs[i + 1]
            diff = (m1 - m2).reshape(3, 1)
            S = 0.5 * (S1 + S2)
            try:
                S_inv = np.linalg.inv(S)
            except:
                S_inv = np.linalg.pinv(S)
            D = float(diff.T @ S_inv @ diff)
            if D > best_dist:
                best_dist = D
                best_pair = (i, i + 1)

        y0_orig = int(regions[best_pair[0]][0] / scale)
        y1_orig = int(regions[best_pair[1]][1] / scale)
        return max(0, min(y0_orig, h - 1)), max(0, min(y1_orig, h))

    # -----------------------------
    # 2. Multi-scale edge detection (수정됨: Custom Canny + Angle)
    # -----------------------------
    def _multi_scale_edge_map(self, roi_img_bgr):
        if roi_img_bgr is None or roi_img_bgr.size == 0:
            return None, None

        # 입력은 BGR이지만 MedianBlur 등을 위해 Gray 변환
        gray = cv2.cvtColor(roi_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        edge_sum = np.zeros((h, w), dtype=np.uint16)
        
        # Angle 값을 저장할 맵 (초기값 -1)
        # 여러 스케일 중 Edge가 검출된 곳의 Angle을 기록합니다.
        # 나중 스케일(더 큰 Blur)의 Angle이 덮어쓰는 구조로 하여 노이즈를 줄입니다.
        full_angle_map = np.zeros((h, w), dtype=np.float32) - 1

        for s in self.scales:
            # [기존 로직 유지] Median Blur 커널 크기 계산
            ksize = 10 * s + 1
            if ksize < 3: ksize = 3
            if ksize % 2 == 0: ksize += 1

            smoothed = cv2.medianBlur(gray, ksize)

            # [수정] cv2.Canny -> custom_canny.detect
            if self.canny_detector:
                # smoothed 이미지를 입력으로 넣음
                # custom_canny 내부적으로 GaussianBlur가 한번 더 돌지만, 
                # 노이즈 억제 측면에서 큰 문제 없음.
                edges, _, angles = self.canny_detector.detect(smoothed, self.canny_thresh1, self.canny_thresh2)
            else:
                # fallback (custom_canny 없을 때)
                edges = cv2.Canny(smoothed, self.canny_thresh1, self.canny_thresh2)
                angles = np.zeros_like(gray, dtype=np.float32)

            # Edge 누적
            mask = (edges > 0)
            edge_sum += mask.astype(np.uint16) * 255
            
            # Angle 정보 저장 (Edge가 있는 위치만 업데이트)
            # 스케일이 커질수록(루프 뒤쪽) 더 굵직한 구조의 Angle이 남게 됨
            if self.canny_detector:
                full_angle_map[mask] = angles[mask]

        # [기존 로직 유지] Thresholding
        edge_combined = np.zeros_like(gray, dtype=np.uint8)
        valid_mask = (edge_sum >= self.edge_sum_thresh)
        edge_combined[valid_mask] = 255
        
        # 최종 Edge가 살아남은 곳의 Angle만 남김 (나머지는 0 처리)
        final_angle_map = np.zeros_like(full_angle_map)
        final_angle_map[valid_mask] = full_angle_map[valid_mask]

        return edge_combined, final_angle_map

    # -----------------------------
    # 3. Geometric Horizon Estimation (수정됨: 조건 1,2,3 적용)
    # -----------------------------
    def _estimate_horizon_geometric(self, edge_img, angle_map, roi_y0, img_width, img_height):
        # [조건 1] Contour 검출
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        # [조건 2] 길이 기준 정렬 후 상위 N개 추출
        # 윤슬 등 작은 노이즈들은 길이가 짧아 여기서 대부분 탈락
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        candidates = contours[:self.top_n_contours]

        best_cnt = None
        min_std_score = float('inf') # 편차는 작을수록 좋음

        for cnt in candidates:
            # 너무 짧은 선은 제외
            if len(cnt) < 20: 
                continue

            # [조건 3] Angle 편차 확인
            std_dev = self._calculate_angle_std(cnt, angle_map)
            
            # 디버깅용 출력 (필요시 주석 해제)
            # print(f"Contour len: {len(cnt)}, Angle Std: {std_dev:.2f}")

            # 편차가 임계값보다 작아야 함 (직선/완만한 곡선 형태)
            if std_dev < self.angle_std_thresh:
                # 여기서 추가로 평균 각도(Mean Angle)를 체크하여 수평선(90도 근처)인지 볼 수도 있음.
                # 우선은 편차가 가장 작은(가장 매끄러운) 선을 선택
                if std_dev < min_std_score:
                    min_std_score = std_dev
                    best_cnt = cnt

        # 만약 조건을 만족하는 것이 하나도 없다면?
        # -> fallback: 그냥 가장 긴 놈을 쓰거나, None 리턴
        if best_cnt is None:
            # 안전장치: 편차 조건이 너무 빡빡해서 실패했다면 그냥 1순위 길이 사용 (옵션)
            if candidates and len(candidates[0]) > 50:
                best_cnt = candidates[0]
            else:
                return None

        # 이하: 양 끝점 연장 및 직선 방정식 도출 (기존 로직)
        pts = best_cnt.squeeze()
        if pts.ndim == 1: pts = pts.reshape(-1, 2)
        pts = pts[pts[:, 0].argsort()] # X축 정렬

        p0 = pts[0]
        p1 = pts[1] # 시작점 부근 방향
        pn = pts[-1]
        pn_1 = pts[-2] # 끝점 부근 방향

        # 조금 더 안정적인 기울기 추정을 위해 양 끝 10% 지점 등을 쓸 수도 있으나,
        # 여기선 기존 로직대로 바로 옆 점을 사용 (직선성이 보장된다면 OK)
        left_y = self._extrapolate_y(p1, p0, 0)
        right_y = self._extrapolate_y(pn_1, pn, img_width - 1)

        left_y_global = np.clip(left_y + roi_y0, 0, img_height)
        right_y_global = np.clip(right_y + roi_y0, 0, img_height)

        dx = (img_width - 1)
        dy = right_y_global - left_y_global
        
        if dx == 0: return None
        a = dy / dx
        b = left_y_global

        x1, y1 = 0, int(round(left_y_global))
        x2, y2 = img_width - 1, int(round(right_y_global))

        return float(a), float(b), (x1, y1, x2, y2)

    def _calculate_angle_std(self, cnt, angle_map):
        """
        Contour 픽셀들의 Angle 표준편차를 계산합니다.
        0도와 360도가 붙어있는 Circular 성질을 고려합니다.
        """
        angles = []
        h, w = angle_map.shape
        
        # 샘플링 스텝 (속도 최적화)
        step = 1
        for i in range(0, len(cnt), step):
            p = cnt[i][0]
            x, y = p[0], p[1]
            if 0 <= x < w and 0 <= y < h:
                a = angle_map[y, x]
                # Angle이 유효한 값인지 체크 (배경 0 또는 초기값 -1 제외)
                if a >= 0:
                    angles.append(a)
        
        if not angles:
            return 999.0

        angles = np.array(angles)

        # 각도 평균 계산 (벡터 합 이용)
        rads = np.deg2rad(angles)
        sin_sum = np.sum(np.sin(rads))
        cos_sum = np.sum(np.cos(rads))
        mean_rad = np.arctan2(sin_sum, cos_sum)
        mean_deg = np.degrees(mean_rad)
        if mean_deg < 0: mean_deg += 360

        # 각도 차이 계산 (최단 거리: 359도와 1도의 차이는 2도)
        diffs = np.abs(angles - mean_deg)
        diffs = np.minimum(diffs, 360 - diffs)

        # 표준편차
        std_dev = np.sqrt(np.mean(diffs ** 2))
        return std_dev

    def _extrapolate_y(self, p_start, p_end, target_x):
        x1, y1 = p_start
        x2, y2 = p_end
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0: return y2
        slope = dy / dx
        return slope * (target_x - x2) + y2

if __name__ == "__main__":
    print("Updated roi_using.py with Multi-scale Custom Canny and Angle Filtering.")