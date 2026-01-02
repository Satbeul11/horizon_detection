import cv2
import numpy as np
import os
import glob
import math


class FastHorizonDetector:
    def __init__(
            self,
            num_regions=9,  # 수평 방향으로 나눌 영역 개수
            roi_resize_factor=0.25,  # ROI 검출용 리사이즈 비율
            scales=(1, 2, 3),  # median filter 스케일
            canny_thresh1=20,
            canny_thresh2=80,
            edge_sum_thresh=170,
            linearity_thresh=0.15  # [New] 경계선이 얼마나 직선에 가까워야 하는지 (기울기 분산 허용치)
    ):
        self.num_regions = num_regions
        self.roi_resize_factor = roi_resize_factor
        self.scales = scales
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.edge_sum_thresh = edge_sum_thresh
        self.linearity_thresh = linearity_thresh  # Gradient 급격한 변화 감지용 임계값

    # -----------------------------
    # Public API
    # -----------------------------
    def detect(self, img_bgr):
        """
        img_bgr: BGR 이미지
        return:
            - (a, b, (x1, y1, x2, y2)) : y = ax + b 파라미터 및 시각화용 좌표
            - None : 검출 실패 시
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]

        # [cite_start]1. ROI 검출 (기존 로직 유지) [cite: 17, 18, 19]
        roi_y0, roi_y1 = self._detect_roi_vertical_range(img_bgr)
        if roi_y0 >= roi_y1:
            roi_y0 = int(h * 0.3)
            roi_y1 = int(h * 0.7)

        roi_img = img_bgr[roi_y0:roi_y1, :]

        # [cite_start]2. 다중 스케일 에지 검출 (기존 로직 유지) [cite: 20, 21, 22]
        edge_combined = self._multi_scale_edge_map(roi_img)
        if edge_combined is None:
            return None

        # 3. [New] Contour 기반 기하학적 수평선 추정 (조건 1~5 반영)
        result = self._estimate_horizon_geometric(edge_combined, roi_y0, w, h)
        return result

    # -----------------------------
    # 1. ROI detection (변경 없음)
    # -----------------------------
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
            mean = np.mean(pixels, axis=0)
            cov = np.cov(pixels, rowvar=False)
            if cov.shape == ():
                cov = np.eye(3, dtype=np.float32)
            cov = cov + eps * np.eye(3, dtype=np.float32)
            means.append(mean)
            covs.append(cov)

        best_dist = -1.0
        best_pair = (0, 1)

        for i in range(len(regions) - 1):
            m1, m2 = means[i], means[i + 1]
            S1, S2 = covs[i], covs[i + 1]
            diff = (m1 - m2).reshape(3, 1)
            S = 0.5 * (S1 + S2)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            D = float(diff.T @ S_inv @ diff)
            if D > best_dist:
                best_dist = D
                best_pair = (i, i + 1)

        y0_small = regions[best_pair[0]][0]
        y1_small = regions[best_pair[1]][1]
        y0_orig = int(y0_small / scale)
        y1_orig = int(y1_small / scale)
        y0_orig = max(0, min(y0_orig, h - 1))
        y1_orig = max(0, min(y1_orig, h))

        return y0_orig, y1_orig

    # -----------------------------
    # 2. Multi-scale edge detection (변경 없음)
    # -----------------------------
    def _multi_scale_edge_map(self, roi_img_bgr):
        if roi_img_bgr is None or roi_img_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        edge_sum = np.zeros((h, w), dtype=np.uint16)

        for s in self.scales:
            ksize = 10 * s + 1
            if ksize < 3: ksize = 3
            if ksize % 2 == 0: ksize += 1

            smoothed = cv2.medianBlur(gray, ksize)
            edges = cv2.Canny(smoothed, self.canny_thresh1, self.canny_thresh2)
            edge_sum += (edges > 0).astype(np.uint16) * 255

        edge_combined = np.zeros_like(gray, dtype=np.uint8)
        edge_combined[edge_sum >= self.edge_sum_thresh] = 255

        return edge_combined

    # -----------------------------
    # 3. Geometric Horizon Estimation (New)
    # -----------------------------
    def _estimate_horizon_geometric(self, edge_img, roi_y0, img_width, img_height):
        """
        허프 변환 대신 Contour 분석을 사용하여 수평선을 검출합니다.

        <조건 반영>
        1. Canny 이후 경계선(Contour) 검출
        2. Gradient 변화가 급격한(둥근 파도 등) 경계선 제외
        3. 가장 긴 경계선 선택
        4. 양 끝점을 기준으로 이미지 끝까지 연장 (반직선)
        5. 결과적으로 이미지를 이분할하는 선 도출
        """

        # [조건 1] 경계선(Contour) 검출
        # RETR_EXTERNAL: 가장 바깥쪽 라인만, CHAIN_APPROX_NONE: 모든 점 좌표 저장
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None

        valid_contours = []

        for cnt in contours:
            # 점의 개수가 너무 적으면 직선 판단 불가하므로 패스 (노이즈 제거)
            if len(cnt) < 10:
                continue

            # [조건 2] Gradient(기울기) 급격한 변화 체크
            # 수평선은 기울기가 거의 일정해야 함. 백파(whitecap)는 둥글어서 기울기가 급변함.
            if self._check_gradient_stability(cnt):
                valid_contours.append(cnt)

        if not valid_contours:
            return None

        # [조건 3] 가장 긴 점의 집합(경계선) 선택
        # cv2.arcLength로 길이를 측정하여 가장 긴 것 선택
        best_cnt = max(valid_contours, key=lambda c: cv2.arcLength(c, False))

        # 좌표 처리를 위해 차원 축소 및 X축 기준 정렬
        # Contour는 (N, 1, 2) 형태이므로 (N, 2)로 변환
        pts = best_cnt.squeeze()

        # x좌표 기준으로 정렬 (0~n 순서 보장)
        pts = pts[pts[:, 0].argsort()]

        n = len(pts)
        if n < 2: return None

        # [조건 4] 끝 점을 이용한 연장 (반직선)
        # 0, 1번째 점을 잇는 선 -> 왼쪽 끝(x=0)으로 연장
        # n-2, n-1번째 점을 잇는 선 -> 오른쪽 끝(x=Width)으로 연장

        p0 = pts[0]  # 0번째 점 (x가 가장 작은 점)
        p1 = pts[1]  # 1번째 점

        pn = pts[-1]  # n번째 점 (x가 가장 큰 점)
        pn_1 = pts[-2]  # n-1번째 점

        # 4-1. 왼쪽 확장: (p1 -> p0) 방향 벡터를 x=0까지 연장
        left_y = self._extrapolate_y(p1, p0, 0)

        # 4-2. 오른쪽 확장: (pn_1 -> pn) 방향 벡터를 x=img_width까지 연장
        right_y = self._extrapolate_y(pn_1, pn, img_width - 1)

        # ROI 좌표계(roi_y0)를 전체 이미지 좌표계로 변환
        left_y_global = left_y + roi_y0
        right_y_global = right_y + roi_y0

        # 화면 밖으로 나가는 것 클리핑 (옵션)
        left_y_global = np.clip(left_y_global, 0, img_height)
        right_y_global = np.clip(right_y_global, 0, img_height)

        # [조건 5] 이미지를 이분할 (결과 반환)
        # detect 함수의 반환 규격인 y = ax + b 형태로 변환
        # (0, left_y_global) 과 (W-1, right_y_global) 두 점을 잇는 직선 방정식 구하기

        dx = (img_width - 1) - 0
        dy = right_y_global - left_y_global

        if dx == 0: return None  # 수직선 예외처리

        a = dy / dx
        b = left_y_global  # x=0 일 때의 y절편

        # 시각화용 좌표 (정수형)
        x1, y1 = 0, int(round(left_y_global))
        x2, y2 = img_width - 1, int(round(right_y_global))

        return float(a), float(b), (x1, y1, x2, y2)

    def _check_gradient_stability(self, cnt):
        """
        [조건 2 구현 함수]
        경계선을 구성하는 점들의 Gradient(순간 기울기)를 구하고,
        이 값이 급격하게 변하는지(분산/표준편차) 확인하여 필터링합니다.
        """
        pts = cnt.squeeze()

        # 노이즈를 줄이기 위해 점을 일정 간격(step)으로 건너뛰며 기울기 계산
        step = 5
        if len(pts) < step * 2:
            step = 1

        slopes = []
        for i in range(0, len(pts) - step, step):
            p_cur = pts[i]
            p_next = pts[i + step]

            dx = p_next[0] - p_cur[0]
            dy = p_next[1] - p_cur[1]

            # 수직선(dx=0)에 가까우면 기울기가 무한대가 되므로 큰 값 처리
            if dx == 0:
                angle = 90.0  # 혹은 매우 큰 기울기
            else:
                slope = dy / dx
                # 기울기(dy/dx) 자체보다는 각도(arctan)를 쓰는 것이 변화량 체크에 유리함
                angle = math.degrees(math.atan(slope))

            slopes.append(angle)

        if not slopes:
            return False

        # 각도의 표준편차 계산
        # 수평선은 거의 일직선이므로 각도 변화가 적어야 함 (std 낮음)
        # 파도나 둥근 물체는 각도가 계속 변하므로 std 높음
        std_dev = np.std(slopes)

        # 수평선은 대체로 0도 근처여야 함 (평균 각도 체크 옵션)
        mean_angle = np.mean(slopes)

        # 임계값: 표준편차가 작아야 하고(직선), 평균 각도도 너무 수직이면 안됨(수평선 가정)
        # linearity_thresh는 생성자에서 조절 가능 (기본값 설정 필요)
        # 여기서는 각도(degree) 기준 대략 10~20도 이상 꺾이면 제외
        if std_dev > 10.0:
            return False

        if abs(mean_angle) > 45.0:  # 수평선이 45도 이상 기울어질 리 없다고 가정
            return False

        return True

    def _extrapolate_y(self, p_start, p_end, target_x):
        """
        [조건 4 구현 함수]
        두 점(p_start -> p_end)을 잇는 직선 상에서 target_x일 때의 y좌표를 구함
        """
        x1, y1 = p_start
        x2, y2 = p_end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            return y2  # 수직선이면 y값 그대로 반환

        slope = dy / dx

        # y - y2 = m(x - x2)  =>  y = m(x - x2) + y2
        target_y = slope * (target_x - x2) + y2
        return target_y


if __name__ == "__main__":
    # 간단한 테스트 실행
    print("수정된 roi_using.py 모듈입니다. analysis.py에서 import하여 사용하세요.")