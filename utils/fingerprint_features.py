
import cv2
import numpy as np

def radial_density(gray_img, num_rings=5):
    h, w = gray_img.shape
    center = (w // 2, h // 2)
    max_radius = int(np.hypot(w, h) / 2)
    edges = cv2.Canny(gray_img, 100, 200)
    results = []
    for i in range(num_rings):
        r1 = int(i * max_radius / num_rings)
        r2 = int((i + 1) * max_radius / num_rings)
        mask = np.zeros_like(gray_img)
        cv2.circle(mask, center, r2, 255, -1)
        cv2.circle(mask, center, r1, 0, -1)
        ring = cv2.bitwise_and(edges, edges, mask=mask)
        density = np.count_nonzero(ring) / np.count_nonzero(mask)
        results.append(round(density, 4))
    return results

def curve_direction_label(edges):
    sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    avg_grad = np.mean(sobelx)
    if avg_grad > 10:
        return "선들이 우측으로 흐름"
    elif avg_grad < -10:
        return "선들이 좌측으로 흐름"
    else:
        return "선의 방향이 정중앙 또는 다양함"

def summarize_fingerprint(gray_img):
    edges = cv2.Canny(gray_img, 100, 200)
    densities = radial_density(gray_img)
    direction = curve_direction_label(edges)
    summary = []
    for i, d in enumerate(densities):
        if d > 0.05:
            summary.append(f"{i+1}번째 나이테: 뚜렷함")
        elif d > 0.02:
            summary.append(f"{i+1}번째 나이테: 흐릿함")
        else:
            summary.append(f"{i+1}번째 나이테: 거의 없음")
    summary.append(f"곡선 방향: {direction}")
    return "".join(summary), densities
