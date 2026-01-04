# analysis_old.py
# roi_using_old.py 와 같은 폴더에 두고 실행하세요.

import os
import glob
import cv2
import numpy as np

# 같은 폴더의 roi_using_old.py에서 클래스 임포트
from roi_using_old import FastHorizonDetector


def imread_unicode(path: str):
    """한글 경로 이미지 읽기"""
    stream = np.fromfile(path, dtype=np.uint8)
    if stream.size == 0:
        return None
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img) -> bool:
    """한글 경로 이미지 저장"""
    ext = os.path.splitext(path)[1]
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    with open(path, "wb") as f:
        buf.tofile(f)
    return True


def compute_roi_stats_and_distances(img_bgr: np.ndarray, detector: FastHorizonDetector):
    """
    (기존 코드 유지) Bhattacharyya 거리 계산용 통계
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
        mean = np.mean(pixels, axis=0)                # (3,)
        cov = np.cov(pixels, rowvar=False)            # (3,3)
        if cov.shape == ():
            cov = np.eye(3, dtype=np.float32)
        cov = cov + eps * np.eye(3, dtype=np.float32)

        means.append(mean)
        covs.append(cov)

    # 인접쌍 거리 D
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

    # small -> original 좌표 변환(ROI 박스용)
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


def draw_roi_boxes_distances_stats(
    img_bgr,
    regions_orig,
    pair_distances,
    means,
    covs,
    show_full_cov=False,
    force_nine=False
):
    """(기존 코드 유지) 통계 박스 그리기"""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    if force_nine:
        regions_orig = regions_orig[:9]
        means = means[:9]
        covs = covs[:9]
        pair_distances = pair_distances[:8]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.9, w / 2200.0))
    thickness = 2

    # ROI 박스 + mean/cov 텍스트
    for k, (y0, y1) in enumerate(regions_orig):
        cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"ROI {k}", (10, max(25, y0 + 25)),
                    font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        m = means[k]
        c = covs[k]
        mean_txt = f"mean(BGR)=({m[0]:.1f},{m[1]:.1f},{m[2]:.1f})"

        if not show_full_cov:
            diag = np.diag(c)
            cov_txt1 = f"cov_diag=({diag[0]:.1f},{diag[1]:.1f},{diag[2]:.1f})"
            y_text1 = min(y1 - 10, y0 + 55)
            y_text2 = min(y1 - 10, y0 + 80)
            cv2.putText(vis, mean_txt, (10, y_text1),
                        font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)
            cv2.putText(vis, cov_txt1, (10, y_text2),
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

    # 경계별 Bhattacharyya 거리 텍스트
    x_text = int(w * 0.68)
    for (i, j, dist) in pair_distances:
        if i >= len(regions_orig) - 1:
            continue
        y_boundary = regions_orig[i][1]
        y_text = int(np.clip(y_boundary - 5, 20, h - 10))
        cv2.line(vis, (0, y_boundary), (w - 1, y_boundary), (255, 255, 0), 1)
        cv2.putText(vis, f"D({i}-{j})={dist:.4f}", (x_text, y_text),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

    return vis


def draw_roi_visuals(vis, debug_info):
    """
    debug_info를 이용하여:
    1. 실제 Edge Detection에 사용된 ROI 영역을 반투명한 회색으로 표시
    2. 추출된 Canny Edge 후보점들을 Cyan(하늘색)으로 표시
    """
    if not debug_info:
        return vis

    h, w = vis.shape[:2]
    
    # roi_using_old.py에서 넘겨준 debug_info 키값 확인
    roi_y0, roi_y1 = debug_info.get("roi_y", (0, 0))
    edge_map = debug_info.get("edges", None)

    # 1. ROI 영역 회색 오버레이 (Transparency)
    if roi_y1 > roi_y0:
        overlay = vis.copy()
        # 회색(128, 128, 128)으로 해당 영역 채우기
        cv2.rectangle(overlay, (0, roi_y0), (w, roi_y1), (80, 80, 80), -1)
        
        alpha = 0.6  # 투명도 (0.0: 투명 ~ 1.0: 불투명)
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    # 2. Canny Edge 후보점 표시
    if edge_map is not None:
        # edge_map은 ROI 영역 기준 좌표이므로, 전체 이미지 좌표로 변환 필요
        # 값이 0보다 큰(에지인) 위치의 y, x 인덱스 추출
        ys_roi, xs_roi = np.where(edge_map > 0)
        
        # 원본 좌표계로 변환 (ROI 시작점 roi_y0 더하기)
        ys_global = ys_roi + roi_y0
        xs_global = xs_roi
        
        # 이미지 범위 내에 있는 좌표만 필터링 (안전장치)
        valid_mask = (ys_global >= 0) & (ys_global < h) & (xs_global >= 0) & (xs_global < w)
        
        valid_ys = ys_global[valid_mask]
        valid_xs = xs_global[valid_mask]
        
        # 픽셀 색칠하기: Cyan (Blue=255, Green=255, Red=0)
        # 점이 많을 경우 for문보다 이렇게 인덱싱으로 처리하는 것이 훨씬 빠름
        point_color = (100, 255, 255)

        # for문을 돌며 원(circle)으로 그리기 -> 점이 훨씬 굵고 진하게 보임
        # radius=1 이면 지름 3픽셀 정도, radius=2 이면 지름 5픽셀 정도
        for x, y in zip(valid_xs, valid_ys):
            cv2.circle(vis, (x, y), 1, point_color, 1)

    return vis


def draw_horizon_line(vis, horizon_result):
    if horizon_result is None:
        return vis
    try:
        a, b, (x1, y1, x2, y2) = horizon_result
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(vis, "Horizon",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    return vis


def main():
    # 경로 설정
    input_dir = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o\island_x\select(200)"
    out_dir = os.path.join(input_dir, "논문 그대로(최적화)")
    os.makedirs(out_dir, exist_ok=True)

    detector = FastHorizonDetector()

    # 이미지 파일 리스트업
    img_paths = []
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    for pattern in patterns:
        img_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    # 중복 제거 및 정렬
    img_paths = sorted(set(os.path.normpath(p) for p in img_paths))
    input_dir_norm = os.path.normcase(os.path.abspath(input_dir))
    img_paths = [p for p in img_paths if os.path.normcase(os.path.dirname(p)) == input_dir_norm]

    total = len(img_paths)
    for idx, img_path in enumerate(img_paths, start=1):
        img_id = os.path.basename(img_path)
        print(f"[{idx}/{total}] Processing: {img_id}")

        img = imread_unicode(img_path)
        if img is None:
            print("  -> load fail, skip")
            continue

        # 1. 통계용 디버그 정보 계산
        debug = compute_roi_stats_and_distances(img, detector)
        if debug is None:
            print("  -> ROI calc fail, skip")
            continue

        # 2. 수평선 검출 실행 (roi_using_old.py)
        # 중요: roi_using_old.py가 수정되어 (result, debug_info)를 반환해야 함
        horizon_result, debug_info = detector.detect(img)

        # 3. 시각화 그리기
        
        # 3-1. 통계 박스 (초록색 박스들)
        vis = draw_roi_boxes_distances_stats(
            img,
            debug["regions_orig"],
            debug["pair_distances"],
            debug["means"],
            debug["covs"],
            show_full_cov=False,
            force_nine=True
        )

        # 3-2. ✅ 회색 ROI 영역 및 Edge 포인트 표시 (이 부분이 빠져 있었습니다)
        vis = draw_roi_visuals(vis, debug_info)

        # 3-3. 수평선 그리기
        vis = draw_horizon_line(vis, horizon_result)

        # 4. 저장
        save_path = os.path.join(out_dir, os.path.splitext(img_id)[0] + "_roi_dist_horizon_stats.jpg")
        imwrite_unicode(save_path, vis)

    print("\n[DONE] Saved overlays to:", out_dir)


if __name__ == "__main__":
    main()