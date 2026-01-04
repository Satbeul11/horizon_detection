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
            scales=(1, 2, 3),
            canny_thresh1=76,
            canny_thresh2=147,
            edge_sum_thresh=148,
            top_n_contours=5,
            angle_std_thresh=20.0,
            target_angle=90.0
    ):
        self.num_regions = num_regions
        self.roi_resize_factor = roi_resize_factor
        self.scales = scales
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.edge_sum_thresh = edge_sum_thresh
        
        self.top_n_contours = top_n_contours
        self.angle_std_thresh = angle_std_thresh
        self.target_angle = target_angle

        if GradientCannyDetector:
            self.canny_detector = GradientCannyDetector(ksize=3, blur_ksize=5)
        else:
            self.canny_detector = None

    def detect(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]

        # 1. ROI 검출
        roi_y0, roi_y1 = self._detect_roi_vertical_range(img_bgr)
        if roi_y0 >= roi_y1:
            roi_y0 = int(h * 0.3)
            roi_y1 = int(h * 0.7)

        roi_img = img_bgr[roi_y0:roi_y1, :]

        # 2. Multi-scale Edge & Angle Map 생성
        edge_combined, angle_combined = self._multi_scale_edge_map(roi_img)
        
        if edge_combined is None:
            return None

        # 3. [Modified] 굴곡과 직선이 혼합된 최종 수평선 추출
        # 반환값: (기울기근사, y절편근사, 전체_좌표_배열)
        result = self._estimate_horizon_geometric(edge_combined, angle_combined, roi_y0, w, h)
        return result

    def _detect_roi_vertical_range(self, img_bgr):
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

    def _multi_scale_edge_map(self, roi_img_bgr):
        if roi_img_bgr is None or roi_img_bgr.size == 0:
            return None, None

        gray = cv2.cvtColor(roi_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        edge_sum = np.zeros((h, w), dtype=np.uint16)
        full_angle_map = np.zeros((h, w), dtype=np.float32) - 1

        for s in self.scales:
            ksize = 10 * s + 1
            if ksize < 3: ksize = 3
            if ksize % 2 == 0: ksize += 1

            smoothed = cv2.medianBlur(gray, ksize)

            if self.canny_detector:
                edges, _, angles = self.canny_detector.detect(smoothed, self.canny_thresh1, self.canny_thresh2)
            else:
                edges = cv2.Canny(smoothed, self.canny_thresh1, self.canny_thresh2)
                angles = np.zeros_like(gray, dtype=np.float32)

            mask = (edges > 0)
            edge_sum += mask.astype(np.uint16) * 255
            
            if self.canny_detector:
                full_angle_map[mask] = angles[mask]

        edge_combined = np.zeros_like(gray, dtype=np.uint8)
        valid_mask = (edge_sum >= self.edge_sum_thresh)
        edge_combined[valid_mask] = 255
        
        final_angle_map = np.zeros_like(full_angle_map)
        final_angle_map[valid_mask] = full_angle_map[valid_mask]

        return edge_combined, final_angle_map

    def _estimate_horizon_geometric(self, edge_img, angle_map, roi_y0, img_width, img_height):
        # 1. Contour 검출
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        # 2. 길이 기준 정렬
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        candidates = contours[:self.top_n_contours]

        best_cnt = None
        min_std_score = float('inf')

        for cnt in candidates:
            if len(cnt) < 20: continue

            # 3. Angle 편차 확인
            std_dev = self._calculate_angle_std(cnt, angle_map)
            
            if std_dev < self.angle_std_thresh:
                if std_dev < min_std_score:
                    min_std_score = std_dev
                    best_cnt = cnt

        if best_cnt is None:
            if candidates and len(candidates[0]) > 50:
                best_cnt = candidates[0]
            else:
                return None

        # --- [Modified] 굴곡이 유지된 중간 + 양 끝 직선 연장 ---
        
        # 1) Contour 좌표 정리 (X축 기준 정렬)
        pts = best_cnt.squeeze()
        if pts.ndim == 1: pts = pts.reshape(-1, 2)
        pts = pts[pts[:, 0].argsort()] 

        # 2) 전역 좌표(Global Coordinates)로 변환
        # ROI 내부 좌표이므로 roi_y0를 더해줘야 전체 이미지 좌표가 됨
        pts_global = pts.astype(np.float32)
        pts_global[:, 1] += roi_y0

        # 3) 왼쪽 확장 (직선)
        # 가장 왼쪽 두 점을 사용하여 기울기 계산 -> x=0까지 연장
        p0 = pts_global[0] # 가장 왼쪽 점
        p1 = pts_global[1] # 두 번째 왼쪽 점
        left_y_global = self._extrapolate_y(p1, p0, 0)
        
        # 4) 오른쪽 확장 (직선)
        # 가장 오른쪽 두 점을 사용하여 기울기 계산 -> x=W-1까지 연장
        pn = pts_global[-1]   # 가장 오른쪽 점
        pn_1 = pts_global[-2] # 두 번째 오른쪽 점
        right_y_global = self._extrapolate_y(pn_1, pn, img_width - 1)

        # 5) 점 합치기: [Left_Point] + [Contour_Points] + [Right_Point]
        left_pt = np.array([[0, left_y_global]], dtype=np.float32)
        right_pt = np.array([[img_width - 1, right_y_global]], dtype=np.float32)
        
        # 최종 좌표 배열 (N x 2)
        full_line_pts = np.concatenate((left_pt, pts_global, right_pt), axis=0)
        full_line_pts = full_line_pts.astype(np.int32) # 정수형 변환

        # 6) 반환값 구성
        # a, b는 전체를 대표하는 직선의 근사치(시작점과 끝점 기준)로 남겨둠 (호환성용)
        dy = right_y_global - left_y_global
        dx = img_width - 1
        a = dy / dx if dx != 0 else 0
        b = left_y_global

        # **중요**: 마지막 반환값이 이제 (x1,y1,x2,y2) 튜플이 아니라 
        # 전체 경로를 담은 'numpy array'입니다.
        return float(a), float(b), full_line_pts

    def _calculate_angle_std(self, cnt, angle_map):
        angles = []
        h, w = angle_map.shape
        step = 1
        for i in range(0, len(cnt), step):
            p = cnt[i][0]
            x, y = p[0], p[1]
            if 0 <= x < w and 0 <= y < h:
                a = angle_map[y, x]
                if a >= 0:
                    angles.append(a)
        
        if not angles:
            return 999.0

        angles = np.array(angles)
        rads = np.deg2rad(angles)
        sin_sum = np.sum(np.sin(rads))
        cos_sum = np.sum(np.cos(rads))
        mean_rad = np.arctan2(sin_sum, cos_sum)
        mean_deg = np.degrees(mean_rad)
        if mean_deg < 0: mean_deg += 360

        diffs = np.abs(angles - mean_deg)
        diffs = np.minimum(diffs, 360 - diffs)
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
    print("Updated roi_using.py: Returns full polyline (Linear Ext + Curved Middle).")