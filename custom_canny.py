import cv2
import numpy as np

class GradientCannyDetector:
    def __init__(self, ksize=3, blur_ksize=5):
        """
        :param ksize: Sobel 연산 커널 크기 (기본 3)
        :param blur_ksize: 노이즈 제거를 위한 가우시안 블러 커널 크기 (기본 5)
        """
        self.ksize = ksize
        self.blur_ksize = blur_ksize

    def detect(self, img_bgr, low_thresh, high_thresh):
        """
        이미지에서 에지뿐만 아니라 Gradient Magnitude와 Angle을 함께 추출합니다.
        
        :param img_bgr: 입력 이미지 (BGR)
        :param low_thresh: Canny Low Threshold
        :param high_thresh: Canny High Threshold
        :return: (edges, magnitude, angle)
            - edges: Canny 결과 (0 or 255)
            - magnitude: Gradient 강도 (float32)
            - angle: Gradient 방향 (각도, float32)
        """
        if img_bgr is None or img_bgr.size == 0:
            return None, None, None

        # 1. Grayscale 변환
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        # 2. 노이즈 제거 (Gaussian Blur) - Canny의 필수 전처리
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 1.4)

        # 3. Gradient 계산 (Sobel)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=self.ksize)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=self.ksize)

        # 4. Magnitude(강도)와 Angle(방향) 계산
        # magnitude: 에지의 세기 (변화량이 클수록 큼)
        # angle: 에지의 수직 방향 (0~360도)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        # 5. Canny Edge Detection
        # (이미 Sobel을 구했으므로 직접 NMS를 구현할 수도 있지만, 속도를 위해 cv2.Canny 활용)
        # 주의: cv2.Canny는 내부적으로 자체 블러링을 할 수 있으므로 원본 gray나 blurred 둘 다 사용 가능하나
        # 여기서는 gradient map과의 일치성을 위해 blurred를 넣습니다.
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        return edges, magnitude, angle

if __name__ == "__main__":
    # 테스트 코드
    print("GradientCannyDetector 모듈 테스트")
    # img = cv2.imread('test.jpg')
    # detector = GradientCannyDetector()
    # e, m, a = detector.detect(img, 50, 150)