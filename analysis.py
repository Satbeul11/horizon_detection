# export_roi_bhatt_to_excel.py
# 같은 폴더에 roi_using.py 와 함께 두고 실행하세요.

import os
import glob
import cv2
import numpy as np
import pandas as pd

from roi_using import FastHorizonDetector  # <- 같은 폴더의 roi_using.py import


# -------------------------
# 한글/유니코드 경로 안전 I/O
# -------------------------
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


# -------------------------
# roi_using.py의 ROI 선택 로직을 "외부"에서 그대로 재현
# (roi_using.py는 수정하지 않음)
# -------------------------
def compute_all_roi_distances(img_bgr: np.ndarray, detector: FastHorizonDetector):
    """
    Returns dict:
      - regions_small: [(y0,y1), ...] on resized image
      - regions_orig:  [(y0,y1), ...] mapped to original
      - pair_distances: [(i, i+1, D), ...]
      - best_pair: (i, i+1) or None
      - best_dist: float or None
      - roi_y0_orig, roi_y1_orig: final selected ROI range on original
      - meta: scale, small_h, small_w
    """
    h, w = img_bgr.shape[:2]
    scale = detector.roi_resize_factor

    # roi_using.py와 동일: 너무 작은 이미지는 전체 사용
    if h < 50 or w < 50:
        return {
            "regions_small": [],
            "regions_orig": [],
            "pair_distances": [],
            "best_pair": None,
            "best_dist": None,
            "roi_y0_orig": 0,
            "roi_y1_orig": h,
            "meta": {"scale": scale, "small_h": None, "small_w": None},
        }

    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    img_small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)

    N = detector.num_regions
    if N < 2:
        return {
            "regions_small": [],
            "regions_orig": [],
            "pair_distances": [],
            "best_pair": None,
            "best_dist": None,
            "roi_y0_orig": 0,
            "roi_y1_orig": h,
            "meta": {"scale": scale, "small_h": small_h, "small_w": small_w},
        }

    # roi_using.py와 동일: 50% overlap
    step = small_h // (N + 1)
    region_height = step * 2

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
        return {
            "regions_small": regions,
            "regions_orig": [],
            "pair_distances": [],
            "best_pair": None,
            "best_dist": None,
            "roi_y0_orig": 0,
            "roi_y1_orig": h,
            "meta": {"scale": scale, "small_h": small_h, "small_w": small_w},
        }

    # mean/cov 계산 (roi_using.py와 동일)
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

    # 인접쌍 거리 D + best_pair 선택 (roi_using.py와 동일)
    best_dist = -1.0
    best_pair = (0, 1)
    pair_distances = []

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

        if D > best_dist:
            best_dist = D
            best_pair = (i, i + 1)

    # best_pair 두 구역 합집합을 ROI로 (roi_using.py와 동일)
    y0_small = regions[best_pair[0]][0]
    y1_small = regions[best_pair[1]][1]

    y0_orig = int(y0_small / scale)
    y1_orig = int(y1_small / scale)
    y0_orig = max(0, min(y0_orig, h - 1))
    y1_orig = max(0, min(y1_orig, h))

    # 모든 후보 regions도 원본좌표로 변환
    regions_orig = []
    for (ys0, ys1) in regions:
        yo0 = int(ys0 / scale)
        yo1 = int(ys1 / scale)
        yo0 = max(0, min(yo0, h - 1))
        yo1 = max(0, min(yo1, h))
        regions_orig.append((yo0, yo1))

    return {
        "regions_small": regions,
        "regions_orig": regions_orig,
        "pair_distances": pair_distances,
        "best_pair": best_pair,
        "best_dist": best_dist,
        "roi_y0_orig": y0_orig,
        "roi_y1_orig": y1_orig,
        "meta": {"scale": scale, "small_h": small_h, "small_w": small_w},
    }


def main():
    # roi_using.py의 run_on_val_folder()와 "같은 데이터 경로" 기본값 사용
    input_dir = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_data"

    output_dir = os.path.join(input_dir, "excel_results")
    os.makedirs(output_dir, exist_ok=True)

    detector = FastHorizonDetector()

    img_paths = []
    img_paths.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
    img_paths.extend(glob.glob(os.path.join(input_dir, "*.JPG")))
    img_paths = sorted(img_paths)

    per_image_rows = []
    pair_rows = []
    region_rows = []

    for idx, img_path in enumerate(img_paths, start=1):
        img_id = os.path.basename(img_path)
        print(f"[{idx}/{len(img_paths)}] {img_id}")

        img = imread_unicode(img_path)
        if img is None:
            print("  -> 이미지 로드 실패, 건너뜀")
            continue

        h, w = img.shape[:2]
        debug = compute_all_roi_distances(img, detector)

        # 1) 이미지 단위 요약
        per_image_rows.append({
            "img_id": img_id,
            "img_h": h,
            "img_w": w,
            "scale": debug["meta"]["scale"],
            "small_h": debug["meta"]["small_h"],
            "small_w": debug["meta"]["small_w"],
            "roi_y0_orig": debug["roi_y0_orig"],
            "roi_y1_orig": debug["roi_y1_orig"],
            "best_pair_i": debug["best_pair"][0] if debug["best_pair"] else None,
            "best_pair_j": debug["best_pair"][1] if debug["best_pair"] else None,
            "best_dist": debug["best_dist"],
            "num_regions": len(debug["regions_orig"]),
            "num_pairs": len(debug["pair_distances"]),
        })

        # 2) 인접쌍 거리(전부)
        for (i, j, dist) in debug["pair_distances"]:
            pair_rows.append({
                "img_id": img_id,
                "pair_i": i,
                "pair_j": j,
                "bhatt_like_dist": dist,
                "is_best": 1 if debug["best_pair"] == (i, j) else 0,
            })

        # 3) 후보 ROI 구역(전부) + crop 저장
        for k, (y0, y1) in enumerate(debug["regions_orig"]):
            region_rows.append({
                "img_id": img_id,
                "roi_idx": k,
                "y0_orig": y0,
                "y1_orig": y1,
                "height": max(0, y1 - y0),
            })


    # 엑셀 저장 (시트 3개)
    xlsx_path = os.path.join(output_dir, "roi_bhattacharyya_distances.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(per_image_rows).to_excel(writer, index=False, sheet_name="per_image_summary")
        pd.DataFrame(pair_rows).to_excel(writer, index=False, sheet_name="adjacent_pair_dist")
        pd.DataFrame(region_rows).to_excel(writer, index=False, sheet_name="all_regions")

    print("\n[DONE] Excel saved:", xlsx_path)
    print("[DONE] All ROI crops saved in:", crops_dir)


if __name__ == "__main__":
    main()
