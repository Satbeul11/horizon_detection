import os
import glob
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image  # 꼭 상단에 추가해 주세요

# 📌 경로 설정
IMAGE_DIR = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\New_sample\원천데이터\동해_제주항2구역_SEG20\20201221"
JSON_DIR = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\New_sample\라벨링데이터\동해_제주항2구역_SEG\20201221"
OUTPUT_DIR = r"C:\Users\LEEJINSE\Desktop\sim\춘계학술대회_논문준비\결과"

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📌 학습 설정
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_CLASSES = 6
EPOCHS = 20

# 📌 클래스 매핑 (라벨 → 인덱스)
CLASS_MAPPING = {
    "sky": 0,
    "sea": 1,
    "island": 2,
    "rock": 3,
    "wharf": 4,
    "others": 5
}

# 🔹 데이터 로드 함수
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = os.path.basename(data["imagePath"])
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"⚠️ 이미지 파일 없음: {image_path}")
        return None, None

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)
        image = np.array(image)
    except Exception as e:
        print(f"⚠️ PIL 이미지 로드 실패: {image_path} | {e}")
        return None, None

    # 🎯 단일 채널 정수 마스크 (클래스 인덱스용)
    mask = np.zeros(IMG_SIZE, dtype=np.uint8)

    for shape in data["shapes"]:
        label = shape["label"]
        if label in CLASS_MAPPING:
            points = np.array(shape["points"], dtype=np.float32)
            points[:, 0] *= IMG_SIZE[0] / data["imageWidth"]
            points[:, 1] *= IMG_SIZE[1] / data["imageHeight"]
            points = points.astype(np.int32)
            cv2.fillPoly(mask, [points], CLASS_MAPPING[label])

    # 🎯 One-hot encoding
    mask = to_categorical(mask, num_classes=NUM_CLASSES)

    print(f"✅ JSON에서 불러온 이미지 경로: {data['imagePath']}")
    print(f"✅ 최종 이미지 경로: {image_path}")
    print(f"✅ 파일 존재 여부: {os.path.exists(image_path)}")

    return image / 255.0, mask


# 🔹 모든 JSON 파일 가져오기
json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

# 🔹 TensorFlow Dataset 생성
def data_generator():
    for json_file in json_files:
        image, mask = load_data(json_file)
        if image is not None and mask is not None:
            yield image, mask


dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(256, 256, NUM_CLASSES), dtype=tf.float32),
    )
)

dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 🔹 U-Net 모델 정의
def unet_model(input_size=(256, 256, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)

    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4)

    bn = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    bn = layers.Conv2D(1024, 3, activation='relu', padding='same')(bn)

    u1 = layers.UpSampling2D()(bn)
    u1 = layers.concatenate([u1, c4])
    c5 = layers.Conv2D(512, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c5)
    u2 = layers.concatenate([u2, c3])
    c6 = layers.Conv2D(256, 3, activation='relu', padding='same')(u2)

    u3 = layers.UpSampling2D()(c6)
    u3 = layers.concatenate([u3, c2])
    c7 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)

    u4 = layers.UpSampling2D()(c7)
    u4 = layers.concatenate([u4, c1])
    c8 = layers.Conv2D(64, 3, activation='relu', padding='same')(u4)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c8)

    return models.Model(inputs, outputs)

# 🔹 모델 생성 및 컴파일
model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 🔹 모델 학습
history = model.fit(dataset, epochs=EPOCHS)

# 🔹 모델 저장
model.save(os.path.join(OUTPUT_DIR, "unet_sea_segmentation.h5"))

# 🔹 예측 및 결과 저장
for json_file in json_files[:5]:  # 샘플 5개만 저장
    image, _ = load_data(json_file)
    prediction = model.predict(np.expand_dims(image, axis=0))[0]

    pred_mask = np.argmax(prediction, axis=-1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    im = plt.imshow(pred_mask, cmap="jet")  # 색상에 따라 클래스 인덱스 표현
    plt.title("Predicted Mask")

    # ▶️ 컬러바 추가 (범례 역할)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["sky", "sea", "island", "rock", "wharf", "others"])
    cbar.ax.tick_params(labelsize=8)

    output_file = os.path.join(OUTPUT_DIR, os.path.basename(json_file).replace(".json", "_pred.jpg"))
    plt.savefig(output_file)
    plt.close()