import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

# 경로 설정
IMAGE_DIR = r"./"  # 예측할 이미지가 있는 폴더
MODEL_PATH = r"./unet_sea_segmentation.h5"
OUTPUT_DIR = r"./prediction"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 이미지 설정
IMG_SIZE = (256, 256)
NUM_CLASSES = 6  # 클래스 수 (sky, sea, island, rock, wharf, others)

# 모델 로드
model = load_model(MODEL_PATH)

# .png 이미지 파일 가져오기
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]

# 예측 반복
for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)

    # 이미지 로드 및 전처리
    image = Image.open(img_path).convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    image_array = np.array(image_resized) / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)  # (1, 256, 256, 3)

    # 예측
    prediction = model.predict(input_tensor)[0]  # (256, 256, 6)
    pred_mask = np.argmax(prediction, axis=-1)   # (256, 256)

    # 시각화: 왼쪽 = 원본 / 오른쪽 = 예측 마스크
    plt.figure(figsize=(10, 5))

    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized)
    plt.title("Original Image")
    plt.axis("off")

    # 예측 마스크
    plt.subplot(1, 2, 2)
    im = plt.imshow(pred_mask, cmap="jet")
    plt.title("Predicted Mask")
    plt.axis("off")

    # 컬러바 (오른쪽 마스크에 해당)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["sky", "sea", "island", "rock", "wharf", "others"])
    cbar.ax.tick_params(labelsize=8)

    # 저장
    save_path = os.path.join(OUTPUT_DIR, img_name.replace(".png", "_sidebyside.png"))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


print("✅ 모든 예측 마스크 저장 완료!")
