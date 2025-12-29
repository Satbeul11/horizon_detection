# export_roi_overlay_with_horizon.py
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
    """ROI 분할(보통 9개) + 인접쌍 거리(보통 8개) 전부 계산"""
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
    regions_orig = []
    for (ys0, ys1) in regions:
        yo0 = int(ys0 / scale)
        yo1 = int(ys1 / scale)
        yo0 = max(0, min(yo0, h - 1))
        yo1 = max(0, min(yo1, h))
        regions_orig.append((yo0, yo1))

    return {
        "regions_orig": regions_orig,      # [(y0,y1), ...]
        "pair_distances": pair_distances,  # [(0,1,D), (1,2,D), ...]
    }


def draw_roi_boxes_and_distances(vis, regions_orig, pair_distances):
    """ROI 박스(구역) + 인접 경계에 D 텍스트"""
    h, w = vis.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.0, w / 2000.0))
    thickness = 2

    # ROI 박스
    for k, (y0, y1) in enumerate(regions_orig):
        cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"ROI {k}", (10, max(25, y0 + 25)),
                    font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # 거리값(경계)
    x_text = int(w * 0.68)
    for (i, j, dist) in pair_distances:
        y_boundary = regions_orig[i][1]
        y_text = int(np.clip(y_boundary - 5, 20, h - 10))

        cv2.line(vis, (0, y_boundary), (w - 1, y_boundary), (255, 255, 0), 1)
        cv2.putText(vis, f"D({i}-{j})={dist:.4f}", (x_text, y_text),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    return vis


def draw_horizon_line(vis, horizon_result):
    """
    roi_using.py의 detector.detect() 결과를 받아 최종 수평선 그리기
    - detect가 반환하는 (a, b, (x1,y1,x2,y2)) 형태에 맞춰 그림
    """
    if horizon_result is None:
        return vis

    # roi_using.py 내부 구현에 맞춰 unpack 시도
    try:
        a, b, (x1, y1, x2, y2) = horizon_result
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        # 라벨
        cv2.putText(vis, "Horizon", (max(10, int(min(x1, x2)) + 10), max(30, int(min(y1, y2)) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception:
        # 반환 형식이 다르면 조용히 스킵
        pass

    return vis


def main():
    input_dir = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_data"

    # ✅ 기존 results 폴더 건드리지 않도록 새 폴더
    out_dir = os.path.join(input_dir, "roi_overlay_with_horizon")
    os.makedirs(out_dir, exist_ok=True)

    detector = FastHorizonDetector(num_regions=9)

    # ✅ val_data 바로 아래 jpg/jpeg만 (하위 results 제외)
    img_paths = []
    img_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
    img_paths += glob.glob(os.path.join(input_dir, "*.jpeg"))

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

        # (1) ROI/거리 계산(분석용 오버레이)
        debug = compute_all_roi_distances(img, detector)
        if debug is None:
            print("  -> ROI calc fail, skip")
            continue

        # (2) 최종 수평선 검출(원본 알고리즘 사용)
        horizon_result = detector.detect(img)

        # (3) 그리기
        vis = img.copy()
        vis = draw_roi_boxes_and_distances(vis, debug["regions_orig"], debug["pair_distances"])
        vis = draw_horizon_line(vis, horizon_result)

        # (4) 저장
        save_path = os.path.join(out_dir, os.path.splitext(img_id)[0] + "_roi_dist_horizon.jpg")
        imwrite_unicode(save_path, vis)

    print("\n[DONE] Saved overlays to:", out_dir)


if __name__ == "__main__":
    main()
