import cv2
import numpy as np
import os
import glob


class FastHorizonDetector:
    def __init__(
        self,
        num_regions=9,          # 수평 방향으로 나눌 영역 개수 (논문에서 9 사용)
        roi_resize_factor=0.25, # ROI 검출용 리사이즈 비율 (원본의 1/4)
        scales=(1, 2, 3),       # median filter 스케일 s (커널크기 = 10*s + 1)
        canny_thresh1=20, # origin 50
        canny_thresh2=80, # origin 150
        edge_sum_thresh=170,    # 논문에서 사용한 threshold ≈ 170
        hough_threshold=10      # HoughLines 최소 투표 수, origin = 50
    ):
        self.num_regions = num_regions
        self.roi_resize_factor = roi_resize_factor
        self.scales = scales
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.edge_sum_thresh = edge_sum_thresh
        self.hough_threshold = hough_threshold

    # -----------------------------
    # Public API
    # -----------------------------
    def detect(self, img_bgr):
        """
        img_bgr: BGR(OpenCV) 이미지 (np.ndarray, uint8)
        return:
            - (a, b, (x1, y1, x2, y2))  : y = a x + b 직선 파라미터 및 그려질 두 점 (원본 좌표계)
            - None                      : 실패 시
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]

        # 1. ROI 검출 (리사이즈된 이미지에서)
        roi_y0, roi_y1 = self._detect_roi_vertical_range(img_bgr)
        # 안전장치: 잘못 검출된 경우 기본 중앙부 사용
        if roi_y0 >= roi_y1:
            roi_y0 = int(h * 0.3)
            roi_y1 = int(h * 0.7)

        roi_img = img_bgr[roi_y0:roi_y1, :]

        # 2. 다중 스케일 에지 검출 및 합성
        edge_combined = self._multi_scale_edge_map(roi_img)
        if edge_combined is None:
            return None

        # 3. Hough + Median 필터 기반 outlier 제거 + 최소제곱 피팅
        result = self._estimate_horizon_from_edges(edge_combined, roi_y0, w, h)
        return result

    # -----------------------------
    # 1. ROI detection
    # -----------------------------
    def _detect_roi_vertical_range(self, img_bgr):
        """
        논문 방식에 따라 리사이즈된 이미지에서
        수평 방향으로 N개의 겹치는 구역을 만들고,
        인접 구역 간 Bhattacharyya distance가 가장 큰 지점을 ROI로 선택.
        """
        h, w = img_bgr.shape[:2]
        scale = self.roi_resize_factor

        # 너무 작은 이미지는 그냥 전체 사용
        if h < 50 or w < 50:
            return 0, h

        small_h = max(1, int(h * scale))
        small_w = max(1, int(w * scale))
        img_small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

        N = self.num_regions
        if N < 2:
            return 0, h

        # 50% overlap을 고려한 구역 높이, 스텝 설정
        step = small_h // (N + 1)
        region_height = step * 2  # 대략 50% overlap

        regions = []
        for i in range(N):
            y0 = i * step
            y1 = y0 + region_height
            if y0 >= small_h:
                break
            y1 = min(y1, small_h)
            if y1 - y0 < 5:
                continue
            regions.append((y0, y1))

        if len(regions) < 2:
            return 0, h

        # 각 구역의 mean, covariance 계산
        means = []
        covs = []
        eps = 1e-6

        for (y0, y1) in regions:
            region = img_small[y0:y1, :]
            # (N, 3)
            pixels = region.reshape(-1, 3).astype(np.float32)
            mean = np.mean(pixels, axis=0)
            # 소규모 영역에서 cov가 singular 될 수 있어 regularization
            cov = np.cov(pixels, rowvar=False)
            if cov.shape == ():
                cov = np.eye(3, dtype=np.float32)
            cov = cov + eps * np.eye(3, dtype=np.float32)
            means.append(mean)
            covs.append(cov)

        # 인접 구역 간 Bhattacharyya distance 유사 형태 계산
        best_dist = -1.0
        best_pair = (0, 1)

        for i in range(len(regions) - 1):
            m1, m2 = means[i], means[i + 1]
            S1, S2 = covs[i], covs[i + 1]
            diff = (m1 - m2).reshape(3, 1)
            # (S1 + S2)/2 사용 (일반적인 Bhattacharyya distance 근사)
            S = 0.5 * (S1 + S2)
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            D = float(diff.T @ S_inv @ diff)  # scalar
            if D > best_dist:
                best_dist = D
                best_pair = (i, i + 1)

        # 두 구역의 합집합을 ROI로 사용
        y0_small = regions[best_pair[0]][0]
        y1_small = regions[best_pair[1]][1]

        # 원본 이미지 좌표로 매핑
        y0_orig = int(y0_small / scale)
        y1_orig = int(y1_small / scale)

        # 범위 보정
        y0_orig = max(0, min(y0_orig, h - 1))
        y1_orig = max(0, min(y1_orig, h))

        return y0_orig, y1_orig

    # -----------------------------
    # 2. Multi-scale edge detection
    # -----------------------------
    def _multi_scale_edge_map(self, roi_img_bgr):
        if roi_img_bgr is None or roi_img_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        edge_sum = np.zeros((h, w), dtype=np.uint16)
        edge_maps = []

        for s in self.scales:
            ksize = 10 * s + 1  # 논문: i,j in [-5s,5s] → window size = 10s+1
            # 커널 크기는 홀수이고 양수여야 한다.
            if ksize < 3:
                ksize = 3
            if ksize % 2 == 0:
                ksize += 1

            smoothed = cv2.medianBlur(gray, ksize)
            edges = cv2.Canny(smoothed, self.canny_thresh1, self.canny_thresh2)
            edge_maps.append(edges)
            edge_sum += (edges > 0).astype(np.uint16) * 255  # 0 또는 255를 누적

        # 누적된 edge map을 threshold
        edge_combined = np.zeros_like(gray, dtype=np.uint8)
        edge_combined[edge_sum >= self.edge_sum_thresh] = 255

        return edge_combined

    # -----------------------------
    # 3. Horizon line estimation
    # -----------------------------
    def _estimate_horizon_from_edges(self, edge_img, roi_y0, img_width, img_height):
        """
        edge_img: ROI 영역(부분 이미지)에 대한 binary edge map (0/255)
        roi_y0: ROI가 원본 상에서 시작되는 y좌표
        img_width, img_height: 원본 이미지 크기
        """
        # Hough Transform으로 초기 직선 후보 추출
        lines = cv2.HoughLines(edge_img, 1, np.pi / 180.0, self.hough_threshold)
        if lines is None or len(lines) == 0:
            return None

        # 가장 첫 번째(득표 수가 가장 높은) 직선을 사용
        rho, theta = lines[0][0]

        # theta가 수평에 가까운 경우만 사용 (지평선 가정)
        if abs(np.sin(theta)) < 1e-3:
            return None

        # 후보 직선: x cosθ + y sinθ = ρ → y = (ρ - x cosθ) / sinθ
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # edge 점들 좌표 수집 (ROI 좌표계)
        ys, xs = np.where(edge_img > 0)
        if len(xs) < 10:
            return None

        xs_float = xs.astype(np.float32)
        ys_float = ys.astype(np.float32)

        # 각 점에 대한 후보 직선에서의 예측 y
        y_pred = (rho - xs_float * cos_t) / (sin_t + 1e-6)
        residuals = ys_float - y_pred

        # Median absolute deviation을 이용한 inlier 선택
        abs_res = np.abs(residuals)
        med_abs = np.median(abs_res)
        if med_abs < 1.0:
            med_abs = 1.0  # 너무 작으면 최소 폭 보장

        inlier_mask = abs_res <= (1.5 * med_abs)
        if np.sum(inlier_mask) < 10:
            # inlier가 너무 적으면 실패로 처리
            return None

        xs_in = xs_float[inlier_mask]
        ys_in = ys_float[inlier_mask]

        # 최소제곱 직선 피팅: y = a x + b (ROI 좌표계)
        A = np.vstack([xs_in, np.ones_like(xs_in)]).T
        try:
            (a, b), *_ = np.linalg.lstsq(A, ys_in, rcond=None)
        except np.linalg.LinAlgError:
            return None

        # 원본 좌표계로 변환: y_global = a * x + (b + roi_y0)
        b_global = b + roi_y0

        # 시각화를 위한 두 점 (x=0, x=img_width-1 기준)
        x1, x2 = 0, img_width - 1
        y1 = a * x1 + b_global
        y2 = a * x2 + b_global

        # 화면 밖을 벗어나더라도 clipping
        y1 = float(np.clip(y1, 0, img_height - 1))
        y2 = float(np.clip(y2, 0, img_height - 1))

        return float(a), float(b_global), (int(x1), int(round(y1)), int(x2), int(round(y2)))

def run_on_val_folder():
    # 입력 폴더 경로
    input_dir = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_data"

    # ✅ val_data 안에 results 폴더 만들기
    output_dir = os.path.join(input_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] 결과 저장 폴더:", output_dir)

    detector = FastHorizonDetector()

    # 확장자 대소문자 둘 다 처리
    img_paths = []
    img_paths.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
    img_paths.extend(glob.glob(os.path.join(input_dir, "*.JPG")))

    print(f"총 {len(img_paths)}장의 이미지를 처리합니다.")

    # 한글 파일명 지원용 로더
    def imread_unicode(path):
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img

    def imwrite_unicode(path, img):
        import os
        import numpy as np
        ext = os.path.splitext(path)[1]  # ".jpg" 같은 확장자
        # 이미지 → 메모리 버퍼(바이너리)로 인코딩
        success, buf = cv2.imencode(ext, img)
        if not success:
            return False
        # 유니코드 경로에 안전하게 쓰기
        buf.tofile(path)
        return True

    for idx, img_path in enumerate(sorted(img_paths), start=1):
        print(f"[{idx}/{len(img_paths)}] {img_path}")

        img = imread_unicode(img_path)
        if img is None:
            print("  -> 이미지 로드 실패, 건너뜀")
            continue

        result = detector.detect(img)

        if result is None:
            print("  -> Horizon detection 실패, 원본만 저장")
            vis = img.copy()
        else:
            a, b, (x1, y1, x2, y2) = result
            print(f"  -> y = {a:.4f} x + {b:.2f}")
            vis = img.copy()
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        save_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, save_name)

        ok = imwrite_unicode(save_path, vis)
        print(f"  -> 저장 시도: {save_path}, 성공 여부: {ok}")


# -----------------------------
# 사용 예시 (참고용)
# -----------------------------
if __name__ == "__main__":
    # 이미지 읽기
    run_on_val_folder()

    if result is not None:
        a, b, (x1, y1, x2, y2) = result
        print(f"Horizon line: y = {a:.4f} x + {b:.2f}")
        # 시각화
        vis = img.copy()
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Horizon Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Horizon detection failed.")