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
            target_angle=90.0,
            # [New] 가중치 파라미터 추가 (a, b)
            score_weight_len=0.5,  # a: 길이 가중치 (0.0 ~ 1.0)
            score_weight_std=0.5   # b: 편차(직선성) 가중치 (0.0 ~ 1.0)
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
        
        # 가중치 저장
        self.w_len = score_weight_len
        self.w_std = score_weight_std

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

        # 3. [Modified] 점수 기반(Weighted Score) 최종 수평선 추출
        result = self._estimate_horizon_geometric(edge_combined, angle_combined, roi_y0, w, h)
        return result

    def _detect_roi_vertical_range(self, img_bgr):
        # (기존 로직 유지)
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
        # (기존 로직 유지)
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

        # 2. 길이 기준 정렬 (Longest First)
        # 길이가 긴 녀석들이 우선 후보가 됩니다.
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        candidates = contours[:self.top_n_contours]
        
        if not candidates:
            return None

        # [Score Calculation Setup]
        # 점수 정규화를 위해 최대 길이(Max Length)를 구합니다.
        # 정렬되어 있으므로 첫 번째 녀석이 가장 깁니다.
        max_len = cv2.arcLength(candidates[0], False)
        if max_len == 0: return None

        best_cnt = None
        best_score = -float('inf') # 점수는 높을수록 좋음

        # 3. 후보군 점수 계산 Loop
        for cnt in candidates:
            length = cv2.arcLength(cnt, False)
            
            # 너무 짧은 선은 무조건 패스
            if length < 20: continue

            # Angle 편차 계산
            std_dev = self._calculate_angle_std(cnt, angle_map)
            
            # --- [핵심] 점수 계산 로직 ---
            # (A) 길이 점수 (0 ~ 1.0): 길수록 1.0에 가까움
            norm_len = length / max_len
            
            # (B) 편차 점수 (1.0 ~ 음수): 편차가 0이면 1.0, 임계값이면 0.0, 그보다 크면 마이너스(벌점)
            # 수식: 1 - (내_편차 / 허용_임계값)
            if self.angle_std_thresh > 0:
                norm_std = 1.0 - (std_dev / self.angle_std_thresh)
            else:
                norm_std = 0 # 예외처리

            # 최종 스코어 = (a * 길이점수) + (b * 편차점수)
            final_score = (self.w_len * norm_len) + (self.w_std * norm_std)

            # 디버깅용 (필요시 주석 해제)
            # print(f"Len:{length:.0f}, Std:{std_dev:.1f} -> L_sc:{norm_len:.2f}, S_sc:{norm_std:.2f} = Total:{final_score:.3f}")

            # 최대 점수 갱신
            if final_score > best_score:
                best_score = final_score
                best_cnt = cnt

        # 만약 적절한 후보가 없다면 (점수가 너무 낮거나 없으면), 그냥 가장 긴 놈 사용 (Fallback)
        if best_cnt is None:
            if candidates and len(candidates[0]) > 50:
                best_cnt = candidates[0]
            else:
                return None

        # --- 4. 최종 수평선 좌표 계산 (10% Slope Extrapolation) ---
        pts = best_cnt.squeeze()
        if pts.ndim == 1: pts = pts.reshape(-1, 2)
        pts = pts[pts[:, 0].argsort()] 

        pts_global = pts.astype(np.float32)
        pts_global[:, 1] += roi_y0

        # 양 끝 10% 데이터를 이용한 Trend Line 기울기
        num_pts = len(pts_global)
        subset_len = max(2, int(num_pts * 0.10))
        
        # 왼쪽 확장
        left_segment = pts_global[:subset_len]
        if len(left_segment) >= 2:
            m_left, _ = np.polyfit(left_segment[:, 0], left_segment[:, 1], 1)
        else:
            p0, p1 = pts_global[0], pts_global[1]
            dx = p1[0] - p0[0]
            m_left = (p1[1] - p0[1]) / dx if dx != 0 else 0
            
        p_start_ref = pts_global[0]
        left_y_global = m_left * (0 - p_start_ref[0]) + p_start_ref[1]

        # 오른쪽 확장
        right_segment = pts_global[-subset_len:]
        if len(right_segment) >= 2:
            m_right, _ = np.polyfit(right_segment[:, 0], right_segment[:, 1], 1)
        else:
            pn, pn_1 = pts_global[-1], pts_global[-2]
            dx = pn[0] - pn_1[0]
            m_right = (pn[1] - pn_1[1]) / dx if dx != 0 else 0
            
        p_end_ref = pts_global[-1]
        right_y_global = m_right * ((img_width - 1) - p_end_ref[0]) + p_end_ref[1]

        # 합치기
        left_pt = np.array([[0, left_y_global]], dtype=np.float32)
        right_pt = np.array([[img_width - 1, right_y_global]], dtype=np.float32)
        
        full_line_pts = np.concatenate((left_pt, pts_global, right_pt), axis=0)
        full_line_pts = full_line_pts.astype(np.int32)

        dy = right_y_global - left_y_global
        dx = img_width - 1
        a = dy / dx if dx != 0 else 0
        b = left_y_global

        return float(a), float(b), full_line_pts

    def _calculate_angle_std(self, cnt, angle_map):
        # (기존 로직 유지)
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
    print("Updated roi_using.py: Weighted Score Logic (Length vs StdDev).")