# export_roi_overlay_images.py
# roi_using.py 와 같은 폴더에 두고 실행하세요.

import os
import glob
import cv2
import numpy as np

from roi_using import FastHorizonDetector


def imread_unicode(path: str):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    buf.tofile(path)
    return True


def compute_all_roi_distances(img_bgr: np.ndarray, detector: FastHorizonDetector):
    """ROI 9개(또는 실제 생성된 개수) + 인접쌍 거리(D) 전부 계산"""
    h, w = img_bgr.shape[:2]
    scale = detector.roi_resize_factor

    if h < 50 or w < 50:
        return None

    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    img_small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    N = detector.num_regions
    if N < 2:
        return None

    step = small_h // (N + 1)
    region_height = step * 2  # 50% overlap

    regions = []
    for i in range(N):
        y0 = i * step
        y1 = min(y0 + region_height, small_h)
        if y0 >= small_h:
            break
        if y1 - y0 < 5:
            continue
        regions.append((y0, y1))

    if len(regions) < 2:
        return None

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

    pair_distances = []  # (i, i+1, D)
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
        pair_distances.append((i, i + 1, D))

    # regions를 원본 좌표로 변환
    h0, w0 = h, w
    regions_orig = []
    for (ys0, ys1) in regions:
        yo0 = int(ys0 / scale)
        yo1 = int(ys1 / scale)
        yo0 = max(0, min(yo0, h0 - 1))
        yo1 = max(0, min(yo1, h0))
        regions_orig.append((yo0, yo1))

    return {
        "regions_orig": regions_orig,      # [(y0,y1), ...]  (보통 9개)
        "pair_distances": pair_distances,  # [(0,1,D0), (1,2,D1), ...] (보통 8개)
    }


def draw_roi_boxes_and_distances(img_bgr, regions_orig, pair_distances):
    """
    - 9개 ROI 구역: 사각형(가로 전체, 세로 y0~y1)
    - 인접 경계 위치에 D 값을 텍스트로 표시
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # 폰트/두께
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.0, w / 2000.0))   # 화면 크기 따라 적당히
    thickness = 2

    # 1) ROI 박스(구역) 그리기
    for k, (y0, y1) in enumerate(regions_orig):
        # ROI 영역 박스
        cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 255, 0), 2)
        # ROI 인덱스 라벨 (왼쪽 위)
        label = f"ROI {k}"
        cv2.putText(vis, label, (10, max(20, y0 + 25)), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # 2) 거리값을 “경계선”에 표시
    # 경계선 y = regions_orig[i][1] (i의 끝 = i+1의 시작 근처)
    # 텍스트는 오른쪽 여백에 표시
    x_text = int(w * 0.70)  # 오른쪽 쪽에
    for (i, j, dist) in pair_distances:
        # i 구역 끝 y를 경계로 사용 (가독성 위해 약간 위/아래 조정)
        y_boundary = regions_orig[i][1]
        y_text = int(np.clip(y_boundary - 5, 20, h - 10))

        text = f"D({i}-{j})={dist:.4f}"
        # 경계선(수평선)도 얇게 하나 표시
        cv2.line(vis, (0, y_boundary), (w - 1, y_boundary), (255, 255, 0), 1)
        # 텍스트
        cv2.putText(vis, text, (x_text, y_text), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    return vis


def main():
    input_dir = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_data"

    # ✅ 기존 results 폴더 건드리지 않게, 시각화 저장 폴더를 분리
    out_dir = os.path.join(input_dir, "roi_overlay_results")
    os.makedirs(out_dir, exist_ok=True)

    detector = FastHorizonDetector(num_regions=9)  # 9개 분할 원하면 명시적으로

    # ✅ val_data 바로 아래 jpg/jpeg만 (results 하위폴더 제외)
    img_paths = []
    img_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
    img_paths += glob.glob(os.path.join(input_dir, "*.jpeg"))

    # 중복 제거 + 안전 필터
    img_paths = sorted(set(os.path.normcase(os.path.abspath(p)) for p in img_paths))
    input_dir_norm = os.path.normcase(os.path.abspath(input_dir))
    img_paths = [p for p in img_paths if os.path.normcase(os.path.dirname(p)) == input_dir_norm]

    total = len(img_paths)
    for idx, img_path in enumerate(img_paths, start=1):
        img_id = os.path.basename(img_path)
        print(f"[{idx}/{total}] {img_id}")

        img = imread_unicode(img_path)
        if img is None:
            print("  -> load fail, skip")
            continue

        debug = compute_all_roi_distances(img, detector)
        if debug is None:
            print("  -> ROI calc fail, skip")
            continue

        regions_orig = debug["regions_orig"]
        pair_distances = debug["pair_distances"]

        vis = draw_roi_boxes_and_distances(img, regions_orig, pair_distances)

        save_path = os.path.join(out_dir, os.path.splitext(img_id)[0] + "_roi_overlay.jpg")
        imwrite_unicode(save_path, vis)

    print("\n[DONE] Saved overlays to:", out_dir)


if __name__ == "__main__":
    main()
