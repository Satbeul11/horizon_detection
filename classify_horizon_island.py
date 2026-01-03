import os
import json
import shutil

def move_images_by_island():
    # 1. 경로 설정
    # 이미 horizon_o 폴더에 모여있는 파일들을 대상으로 합니다.
    source_root = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o"
    
    # 이동할 목적지 폴더 설정 (source_root 안에 생성)
    path_island_o = os.path.join(source_root, "island_o")
    path_island_x = os.path.join(source_root, "island_x")

    # 2. 결과 폴더가 없으면 생성
    os.makedirs(path_island_o, exist_ok=True)
    os.makedirs(path_island_x, exist_ok=True)

    print(f"'{source_root}' 폴더 내의 파일 분류 및 이동을 시작합니다...")

    processed_count = 0
    moved_o = 0
    moved_x = 0

    # 3. 폴더 내 파일 탐색 (os.listdir 사용: 하위 폴더로 재귀진입 하지 않음)
    # 새로 만든 island_o 폴더 안을 또 뒤지지 않게 하기 위함
    files = os.listdir(source_root)

    for file in files:
        # _meta.json이 아닌, 라벨링 데이터(.json) 파일만 찾습니다.
        if file.endswith(".json") and not file.endswith("_meta.json"):
            
            src_json = os.path.join(source_root, file)
            
            # 혹시 폴더가 .json 이름으로 되어있을 경우를 대비해 파일인지 확인
            if not os.path.isfile(src_json):
                continue

            # 파일명 베이스 추출 (예: 여수항_..._0010)
            base_name = os.path.splitext(file)[0]
            
            # 세트 파일 경로 정의
            src_jpg = os.path.join(source_root, base_name + ".jpg")
            src_meta = os.path.join(source_root, base_name + "_meta.json")

            # 4. JSON 내용을 읽어 'island' 라벨이 있는지 확인
            has_island = False
            try:
                with open(src_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "shapes" in data:
                        for shape in data["shapes"]:
                            # 라벨이 island 이거나, (혹시 모를) category_id 확인
                            # 보통 island는 id가 3번인 경우가 많지만, label 텍스트가 가장 확실합니다.
                            if shape.get("label") == "island":
                                has_island = True
                                break
            except Exception as e:
                print(f"JSON 읽기 에러 ({file}): {e}")
                continue

            # 5. 목적지 결정
            if has_island:
                target_folder = path_island_o
            else:
                target_folder = path_island_x

            # 6. 파일 이동 실행 (shutil.move 사용)
            try:
                # (1) JPG 이동
                if os.path.exists(src_jpg):
                    shutil.move(src_jpg, os.path.join(target_folder, base_name + ".jpg"))
                
                # (2) 라벨링 JSON 이동
                # 현재 열고 있는 파일이므로 닫힌 후 이동됨 (with 구문 밖이라 안전)
                shutil.move(src_json, os.path.join(target_folder, file))

                # (3) META JSON 이동
                if os.path.exists(src_meta):
                    shutil.move(src_meta, os.path.join(target_folder, base_name + "_meta.json"))

                # 카운트
                processed_count += 1
                if has_island:
                    moved_o += 1
                else:
                    moved_x += 1
                
                if processed_count % 50 == 0:
                    print(f"{processed_count}세트 이동 중... (Island O: {moved_o}, X: {moved_x})")

            except Exception as e:
                print(f"파일 이동 중 에러 발생 ({base_name}): {e}")

    print("-" * 30)
    print("모든 작업이 완료되었습니다.")
    print(f"총 이동된 세트: {processed_count}")
    print(f"Island 있음 (-> island_o): {moved_o}")
    print(f"Island 없음 (-> island_x): {moved_x}")

if __name__ == "__main__":
    move_images_by_island()