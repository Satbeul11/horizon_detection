# Sea Segmentation using U-Net

본 프로젝트는 해양 객체(하늘, 바다, 섬, 암초, 부두 등)를 구분하기 위한
U-Net 기반 이미지 세그멘테이션 모델입니다.
JSON 형식의 라벨 데이터를 기반으로 학습하여, 입력 이미지에 대해 픽셀 단위 예측을 수행합니다.

## 📁 프로젝트 구조

- sea_segmentation_unet.py : 전체 학습 및 예측 코드
- 결과/
  - unet_sea_segmentation.h5 : 학습된 모델 파일 (Git에 업로드 X)
  - *_pred.jpg : 예측 결과 이미지
- 원천데이터/ : 학습용 원본 이미지
- 라벨링데이터/ : JSON 포맷 라벨 파일
- README.md

## 🧠 학습 정보

- 입력 크기: 256×256 RGB 이미지
- 클래스: sky, sea, island, rock, wharf, others
- Batch size: 8
- Epochs: 20
- Loss: categorical crossentropy
- Optimizer: Adam

## 🔧 사용 방법

python sea_segmentation_unet.py

# 결과는 결과/ 폴더에 저장됩니다.

## 📌 주의사항

- .h5 모델 파일은 GitHub에 업로드되지 않음 (.gitignore 설정됨)
- JSON 라벨 파일은 LabelMe 포맷 기반
- 대용량 파일은 별도 저장소 또는 드라이브에 업로드 요망

## 🙌 기여자

- 심건호 – 해상 객체 분류 연구, 모델 구현, 학습 데이터 구축
