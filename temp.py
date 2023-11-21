import os
import cv2
import numpy as np
import tifffile

def anomaly(original_image, map_combined, path, image_name, y_class, color=(255, 0, 0)):
    defect_class = 'class'
    test_output_dir = os.path.join(path, 'test')
    # 폴더 구조 생성
    if not os.path.exists(os.path.join(test_output_dir, defect_class)):
        os.makedirs(os.path.join(test_output_dir, defect_class))
    normalized_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # [0, 255] 범위로 스케일링 후 numpy 변환
    original_image_np = (normalized_image.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    original_image_np = np.ascontiguousarray(original_image_np)
    # Resize the anomaly map to the original image's shape.
    anomaly_map_resized = cv2.resize(map_combined, (original_image_np.shape[1], original_image_np.shape[0]))

    # 이미지를 0-255 범위로 normalize
    anomaly_map_normalized = cv2.normalize(anomaly_map_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 이미지 이진화 (binary thresholding)
    ret, thresh = cv2.threshold(anomaly_map_normalized, 127, 255, cv2.THRESH_BINARY)

    # 외곽선 찾기
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지에 외곽선 그리기
    highlighted_image = cv2.drawContours(original_image_np, contours, -1, color, 2)

    # 이미지 저장
    img_nm = os.path.splitext(image_name)[0]
    img_nm = img_nm + y_class + '_anomaly'
    file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
    tifffile.imwrite(file, highlighted_image)