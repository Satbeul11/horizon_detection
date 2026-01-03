import os
import json
import shutil

def classify_dataset_by_horizon():
    # 1. 경로 설정 (사용자 환경에 맞게 수정 필요)
    source_root = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\남해_여수항_7구역_SEG"
    
    path_horizon_o = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o"
    path_horizon_x = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_x"

    # 2. 결과 폴더 생성
    os.makedirs(path_horizon_o, exist_ok=True)
    os.makedirs(path_horizon_x, exist_ok=True)

    print("작업을 시작합니다...")
    
    processed_count = 0
    copied_o = 0
    copied_x = 0

    # 3. 폴더 재귀 탐색
    for root, dirs, files in os.walk(source_root):
        for file in files:
            # 기준 파일: _meta가 아닌 순수 라벨링 .json 파일 찾기
            if file.endswith(".json") and not file.endswith("_meta.json"):
                
                # 파일명 분리 (예: 여수항_..._0010.json -> base_name: 여수항_..._0010)
                base_name = os.path.splitext(file)[0]
                
                # 3개의 파일 경로 생성
                src_json = os.path.join(root, file)                          # 라벨링 파일
                src_jpg = os.path.join(root, base_name + ".jpg")             # 이미지 파일
                src_meta = os.path.join(root, base_name + "_meta.json")      # 메타 파일

                # 4. JSON 내용을 읽어 분류 기준 판단 (Horizon 여부)
                target_folder = path_horizon_x  # 기본값: 없음(x)
                has_horizon = False
                
                try:
                    with open(src_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "shapes" in data:
                            for shape in data["shapes"]:
                                if shape.get("label") == "horizon" or shape.get("category_id") == 7:
                                    has_horizon = True
                                    break
                    
                    # Horizon이 있으면 목적지를 변경
                    if has_horizon:
                        target_folder = path_horizon_o
                        
                except Exception as e:
                    print(f"JSON 읽기 오류 ({file}): {e}")
                    continue

                # 5. 세 파일 모두 복사 실행
                # 파일이 실제로 존재할 때만 복사 (jpg나 meta가 없을 수도 있으므로 체크)
                try:
                    # (1) JSON 파일 복사
                    shutil.copy2(src_json, os.path.join(target_folder, file))
                    
                    # (2) JPG 파일 복사
                    if os.path.exists(src_jpg):
                        shutil.copy2(src_jpg, os.path.join(target_folder, base_name + ".jpg"))
                    
                    # (3) META JSON 파일 복사
                    if os.path.exists(src_meta):
                        shutil.copy2(src_meta, os.path.join(target_folder, base_name + "_meta.json"))

                    # 카운트 증가
                    processed_count += 1
                    if has_horizon:
                        copied_o += 1
                    else:
                        copied_x += 1
                        
                    if processed_count % 100 == 0:
                        print(f"{processed_count}세트 처리 중... (O: {copied_o}, X: {copied_x})")

                except Exception as e:
                    print(f"복사 중 오류 발생 ({base_name}): {e}")

    print("-" * 30)
    print("분류 작업 완료")
    print(f"총 처리된 세트: {processed_count}")
    print(f"Horizon 있음 (horizon_o): {copied_o}")
    print(f"Horizon 없음 (horizon_x): {copied_x}")

if __name__ == "__main__":
    classify_dataset_by_horizon()