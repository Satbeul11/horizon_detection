import os
import shutil

# 원본 폴더 (트리 구조로 JPG가 들어 있음)
src_root = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\U_net\labels\val\[라벨]남해_여수항_8구역_SEG"

# JPG 파일을 모아서 넣을 폴더
dst_folder = r"C:\Users\LEEJINSE\Desktop\Horizon_detection\Algorithm_based\val_label"

# 폴더 없으면 생성
os.makedirs(dst_folder, exist_ok=True)

# 재귀적으로 모든 파일 탐색
for root, dirs, files in os.walk(src_root):
    for file in files:
        if file.lower().endswith(".png"):
            src_path = os.path.join(root, file)

            # 파일명 충돌 방지를 위해 원래 경로 기반 이름 부여
            # 예: 여수항_맑음_...jpg → root폴더명_파일명.jpg
            new_name = f"{os.path.basename(root)}_{file}"
            dst_path = os.path.join(dst_folder, new_name)

            # 복사
            shutil.copy2(src_path, dst_path)

print("🔵 모든 JPG 파일 복사 완료!")
