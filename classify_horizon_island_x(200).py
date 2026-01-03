import os
import shutil

def select_copy_images_interval():
    # 1. 경로 설정
    source_root = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o\island_x"
    target_root = r"C:\Users\user\OneDrive - 국립한국해양대학교\바탕 화면\Projects\horizon_o\island_x\select(200)"

    # 2. 결과 폴더 생성
    os.makedirs(target_root, exist_ok=True)

    print("파일 리스트를 불러오고 정렬하는 중입니다...")

    # 3. 파일 목록 정리 (세트 기준을 잡기 위해 .json 파일만 먼저 리스트업)
    # _meta.json은 제외하고, 순수 라벨링 파일(.json)을 기준으로 삼습니다.
    all_files = os.listdir(source_root)
    base_files = []

    for file in all_files:
        if file.endswith(".json") and not file.endswith("_meta.json"):
            base_files.append(file)
    
    # **중요**: 파일 순서가 1, 2, 3... 처럼 되도록 이름순 정렬
    base_files.sort()

    total_files = len(base_files)
    print(f"총 발견된 데이터 세트: {total_files}개")
    print(f"40개 간격으로 최대 200세트를 복사합니다...")

    copied_count = 0
    
    # 4. 간격 두고 선택 (0부터 끝까지, 40씩 건너뜀)
    # range(시작, 끝, 간격) -> 0, 40, 80, 120 ...
    for i in range(0, total_files, 40):
        
        # 200세트가 채워지면 중단
        if copied_count >= 200:
            print("목표 수량(200세트)을 채웠으므로 작업을 중단합니다.")
            break

        # 현재 선택된 파일명 (예: 여수항_..._0010.json)
        current_json = base_files[i]
        base_name = os.path.splitext(current_json)[0]

        # 3개 파일 경로 설정
        src_json = os.path.join(source_root, current_json)
        src_jpg = os.path.join(source_root, base_name + ".jpg")
        src_meta = os.path.join(source_root, base_name + "_meta.json")

        # 5. 파일 복사 실행
        try:
            # (1) 라벨링 JSON 복사
            shutil.copy2(src_json, os.path.join(target_root, current_json))

            # (2) JPG 복사 (있으면)
            if os.path.exists(src_jpg):
                shutil.copy2(src_jpg, os.path.join(target_root, base_name + ".jpg"))
            
            # (3) META JSON 복사 (있으면)
            if os.path.exists(src_meta):
                shutil.copy2(src_meta, os.path.join(target_root, base_name + "_meta.json"))
            
            copied_count += 1
            # 진행상황 출력
            print(f"[{copied_count}/200] 복사됨: {base_name}")

        except Exception as e:
            print(f"복사 중 에러 발생 ({base_name}): {e}")

    print("-" * 30)
    print("작업 완료")
    print(f"원본 파일 총 개수: {total_files}")
    print(f"복사된 세트 수: {copied_count}")

    # 혹시 원본 파일이 모자라서 200개를 못 채운 경우 안내
    if copied_count < 200:
        print(f"※ (참고) 파일이 {total_files}개라서 40간격으로 {copied_count}개만 선택되었습니다.")

if __name__ == "__main__":
    select_copy_images_interval()