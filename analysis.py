# export_roi_all_distances_to_excel.py
# 같은 폴더에 roi_using.py 와 함께 두고 실행하세요.

import os
import glob
import cv2
import numpy as np
import pandas as pd

from roi_using import FastHorizonDetector


def imread_unicode(path: str):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def compute_all_roi_distances(img_bgr: np.ndarray, detector: FastHorizonDetector):
    """
    roi_using.py의 ROI 분할/거리 계산 로직을 외부에서 동일하게 재현.
    - N개 regions -> (N-1)개 인접쌍 거리값 D 모두 반환
    """
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
        pair_distances.append((i, i + 1, D))

        if D > best_dist:
            best_dist = D
            best_pair = (i, i + 1)

    # best_pair에 해당하는 최종 ROI (원본좌표)
    y0_small = regions[best_pair[0]][0]
    y1_small = regions[best_pair[1]][1]
    roi_y0_orig = int(y0_small / scale)
    roi_y1_orig = int(y1_small / scale)
    roi_y0_orig = max(0, min(roi_y0_orig, h - 1))
    roi_y1_orig = max(0, min(roi_y1_orig, h))

    # 모든 후보 region도 원본좌표로
    regions_orig = []
    for (ys0, ys1) in regions:
        yo0 = int(ys0 / scale)
        yo1 = int(ys1 / scale)
        yo0 = max(0, min(yo0, h - 1))
        yo1 = max(0, min(yo1, h))
        regions_orig.append((yo0, yo1))

    return {
        "regions_orig": regions_orig,
        "pair_distances": pair_distances,  # ✅ 여기 안에 8개가 들어감(9분할이면)
        "best_pair": best_pair,
        "best_dist": best_dist,
        "roi_y0_orig": roi_y0_orig,
        "roi_y1_orig": roi_y1_orig,
        "meta": {"scale": scale, "small_h": small_h, "small_w": small_w},
    }


def main():
    input_dir = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_data"

    # ✅ 이미지 저장 안 할 거라, 기존 results 폴더와 충돌 없게 엑셀 폴더만 따로
    output_dir = os.path.join(input_dir, "excel_results")
    os.makedirs(output_dir, exist_ok=True)

    detector = FastHorizonDetector()

    # ✅ val_data "바로 아래" jpg만: results 하위폴더 jpg는 제외
    # val_data 바로 아래 jpg/jpeg만 수집
    img_paths = []
    img_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
    img_paths += glob.glob(os.path.join(input_dir, "*.jpeg"))

    # ✅ Windows 대소문자/경로 차이로 인한 중복 제거
    img_paths = sorted(
        set(os.path.normcase(os.path.abspath(p)) for p in img_paths)
    )

    # (안전장치) 하위 폴더(results 등) 완전 제외
    input_dir_norm = os.path.normcase(os.path.abspath(input_dir))
    img_paths = [
        p for p in img_paths
        if os.path.normcase(os.path.dirname(p)) == input_dir_norm
    ]

    per_image_wide = []
    pair_long = []
    all_regions = []

    for idx, img_path in enumerate(img_paths, start=1):
        img_id = os.path.basename(img_path)
        print(f"[{idx}/{len(img_paths)}] {img_id}")

        img = imread_unicode(img_path)
        if img is None:
            print("  -> load fail, skip")
            continue

        h, w = img.shape[:2]
        debug = compute_all_roi_distances(img, detector)
        if debug is None:
            print("  -> ROI debug fail, skip")
            continue

        # ---------- (A) 이미지당 1행(WIDE): D_0_1 ~ D_7_8 칼럼으로 저장 ----------
        row = {
            "img_id": img_id,
            "img_h": h,
            "img_w": w,
            "num_regions": len(debug["regions_orig"]),
            "num_pairs": len(debug["pair_distances"]),
            "best_pair_i": debug["best_pair"][0],
            "best_pair_j": debug["best_pair"][1],
            "best_dist": debug["best_dist"],
            "roi_y0_orig": debug["roi_y0_orig"],
            "roi_y1_orig": debug["roi_y1_orig"],
        }

        # ✅ 8개 거리값 전부 넣기 (인접쌍 개수만큼)
        for (i, j, dist) in debug["pair_distances"]:
            row[f"D_{i}_{j}"] = dist

        per_image_wide.append(row)

        # ---------- (B) 롱포맷: 이미지-쌍거리 1개당 1행 ----------
        for (i, j, dist) in debug["pair_distances"]:
            pair_long.append({
                "img_id": img_id,
                "pair_i": i,
                "pair_j": j,
                "bhatt_like_dist": dist,
                "is_best": 1 if debug["best_pair"] == (i, j) else 0,
            })

        # ---------- (C) 모든 ROI 구역 좌표 ----------
        for k, (y0, y1) in enumerate(debug["regions_orig"]):
            all_regions.append({
                "img_id": img_id,
                "roi_idx": k,
                "y0_orig": y0,
                "y1_orig": y1,
                "height": max(0, y1 - y0),
            })

    # ---------- 엑셀 저장 ----------
    xlsx_path = os.path.join(output_dir, "roi_all_pair_distances.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(per_image_wide).to_excel(writer, index=False, sheet_name="per_image_wide")
        pd.DataFrame(pair_long).to_excel(writer, index=False, sheet_name="pair_dist_long")
        pd.DataFrame(all_regions).to_excel(writer, index=False, sheet_name="all_regions")

    print("\n[DONE] Excel saved:", xlsx_path)


if __name__ == "__main__":
    main()
