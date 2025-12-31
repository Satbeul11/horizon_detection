# export_roi_overlay_with_horizon_stats_edges.py
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


def compute_roi_stats_and_distances(img_bgr: np.ndarray, detector: FastHorizonDetector):
    """(이전과 동일) ROI들 mean/cov + 인접쌍 D 계산"""
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

    regions_small = []
    for i in range(N):
        y0 = i * step
        y1 = min(y0 + region_height, small_h)
        if y0 >= small_h:
            break
        if y1 - y0 < 5:
            continue
        regions_small.append((y0, y1))

    if len(regions_small) < 2:
        return None

    means, covs = [], []
    eps = 1e-6
    for (y0, y1) in regions_small:
        region = img_small[y0:y1, :]
        pixels = region.reshape(-1, 3).astype(np.float32)
        mean = np.mean(pixels, axis=0)
        cov = np.cov(pixels, rowvar=False)
        if cov.shape == ():
            cov = np.eye(3, dtype=np.float32)
        cov = cov + eps * np.eye(3, dtype=np.float32)
        means.append(mean)
        covs.append(cov)

    pair_distances = []
    for i in range(len(regions_small) - 1):
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

    regions_orig = []
    for (ys0, ys1) in regions_small:
        yo0 = int(ys0 / scale)
        yo1 = int(ys1 / scale)
        yo0 = max(0, min(yo0, h - 1))
        yo1 = max(0, min(yo1, h))
        regions_orig.append((yo0, yo1))

    return {
        "regions_orig": regions_orig,
        "means": means,
        "covs": covs,
        "pair_distances": pair_distances,
    }


def overlay_edges_on_full_image(vis, roi_y0, roi_y1, edge_combined, alpha=0.55):
    """
    edge_combined: ROI 좌표계(roi_y0~roi_y1)에서의 0/255 edge map
    원본 이미지 vis에 edge 위치를 색으로 덧칠(알파 블렌딩)
    """
    if edge_combined is None:
        return vis

    h, w = vis.shape[:2]
    roi_y0 = int(np.clip(roi_y0, 0, h))
    roi_y1 = int(np.clip(roi_y1, 0, h))
    if roi_y0 >= roi_y1:
        return vis

    # ROI 영역만 복사
    roi_vis = vis[roi_y0:roi_y1, :].copy()

    # edge 마스크
    mask = edge_combined > 0
    if mask.shape != roi_vis.shape[:2]:
        # 혹시 크기 불일치(드문 케이스) 방어
        edge_combined_rs = cv2.resize(edge_combined, (roi_vis.shape[1], roi_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = edge_combined_rs > 0

    # edge 색 (BGR) - 핑크/마젠타 계열로 표시
    color = np.zeros_like(roi_vis, dtype=np.uint8)
    color[mask] = (255, 0, 255)

    blended = cv2.addWeighted(roi_vis, 1 - alpha, color, alpha, 0)
    vis[roi_y0:roi_y1, :] = blended
    return vis


def draw_roi_boxes_distances_stats(vis, regions_orig, pair_distances, means, covs, show_full_cov=False):
    """ROI 박스 + mean/cov(대각) + 경계별 D 텍스트"""
    h, w = vis.shape[:2]
    regions_orig = regions_orig[:9]
    means = means[:9]
    covs = covs[:9]
    pair_distances = pair_distances[:8]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.9, w / 2200.0))
    thickness = 2

    for k, (y0, y1) in enumerate(regions_orig):
        cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"ROI {k}", (10, max(25, y0 + 25)),
                    font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        m = means[k]
        c = covs[k]
        mean_txt = f"mean(BGR)=({m[0]:.1f},{m[1]:.1f},{m[2]:.1f})"

        if not show_full_cov:
            d = np.diag(c)
            cov_txt = f"cov_diag=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f})"
            y_text1 = min(y1 - 10, y0 + 55)
            y_text2 = min(y1 - 10, y0 + 80)
            cv2.putText(vis, mean_txt, (10, y_text1),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)
            cv2.putText(vis, cov_txt, (10, y_text2),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)
        else:
            row1 = f"cov[0]=({c[0,0]:.1f},{c[0,1]:.1f},{c[0,2]:.1f})"
            row2 = f"cov[1]=({c[1,0]:.1f},{c[1,1]:.1f},{c[1,2]:.1f})"
            row3 = f"cov[2]=({c[2,0]:.1f},{c[2,1]:.1f},{c[2,2]:.1f})"
            y_text1 = min(y1 - 10, y0 + 55)
            y_text2 = min(y1 - 10, y0 + 80)
            y_text3 = min(y1 - 10, y0 + 105)
            y_text4 = min(y1 - 10, y0 + 130)
            cv2.putText(vis, mean_txt, (10, y_text1),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)
            cv2.putText(vis, row1, (10, y_text2),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)
            cv2.putText(vis, row2, (10, y_text3),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)
            cv2.putText(vis, row3, (10, y_text4),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)

    x_text = int(w * 0.68)
    for (i, j, dist) in pair_distances:
        y_boundary = regions_orig[i][1]
        y_text = int(np.clip(y_boundary - 5, 20, h - 10))
        cv2.line(vis, (0, y_boundary), (w - 1, y_boundary), (255, 255, 0), 1)
        cv2.putText(vis, f"D({i}-{j})={dist:.4f}", (x_text, y_text),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    return vis


def draw_horizon_line(vis, horizon_result):
    if horizon_result is None:
        return vis
    try:
        a, b, (x1, y1, x2, y2) = horizon_result
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(vis, "Horizon", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    return vis


def main():
    input_dir = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_data"
    out_dir = os.path.join(input_dir, "roi_overlay_with_horizon_stats_edges_low_thresh")
    os.makedirs(out_dir, exist_ok=True)

    detector = FastHorizonDetector(num_regions=9)

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
            continue

        # (A) ROI stats / distances (mean/cov 포함)
        debug = compute_roi_stats_and_distances(img, detector)
        if debug is None:
            continue

        # (B) detect()와 동일한 ROI 구간에서 edge_combined 생성
        roi_y0, roi_y1 = detector._detect_roi_vertical_range(img)          # :contentReference[oaicite:4]{index=4}
        roi_img = img[roi_y0:roi_y1, :]
        edge_combined = detector._multi_scale_edge_map(roi_img)            # :contentReference[oaicite:5]{index=5}

        # (C) 최종 수평선
        horizon_result = detector.detect(img)                              # :contentReference[oaicite:6]{index=6}

        # (D) 그리기 순서: Edge → ROI/텍스트 → Horizon
        vis = img.copy()
        vis = overlay_edges_on_full_image(vis, roi_y0, roi_y1, edge_combined, alpha=0.55)
        vis = draw_roi_boxes_distances_stats(
            vis,
            debug["regions_orig"],
            debug["pair_distances"],
            debug["means"],
            debug["covs"],
            show_full_cov=False
        )
        vis = draw_horizon_line(vis, horizon_result)

        save_path = os.path.join(out_dir, os.path.splitext(img_id)[0] + "_ALL_overlay.jpg")
        imwrite_unicode(save_path, vis)

    print("\n[DONE] Saved to:", out_dir)


if __name__ == "__main__":
    main()
