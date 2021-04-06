# -*- coding: cp949 -*-
# -*- coding: utf-8 -*- # �ѱ� �ּ������� �̰� �ؾ���
import cv2  # opencv ���
import numpy as np


def grayscale(img):  # ����̹����� ��ȯ
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny �˰��� ��輱�� �����Ѵ�.
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # ����þ� ���� �̹����� ������ �����Ѵ�
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI ���� ���ϴ� ����� ���� �и��ؼ� ó�� �ϱ� ���� �����Ѵ�.

    mask = np.zeros_like(img)  # mask = img�� ���� ũ���� �� �̹���

    if len(img.shape) > 2:  # Color �̹���(3ä��)��� : ä���� 3�� ���, �ٻ� �̹����̴�. ä���� 1�� ��� �ܻ� �̹����̴�.
        color = color3
    else:  # ��� �̹���(1ä��)��� :
        color = color1

    # vertices�� ���� ����� �̷��� �ٰ����κ�(ROI �����κ�)�� color�� ä��
    cv2.fillPoly(mask, vertices, color)

    # �̹����� color�� ä���� ROI�� ��ħ
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):  # �� �׸���
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):  # ��ǥ�� �׸���
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # ���� ��ȯ
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)

    return lines


def weighted_img(img, initial_img, ��=1, ��=1., ��=0.):  # �� �̹��� operlap �ϱ�
    return cv2.addWeighted(initial_img, ��, img, ��, ��)


def get_fitline(img, f_lines):  # ��ǥ�� ���ϱ�
    lines = np.squeeze(f_lines)
    lines = lines.reshape(lines.shape[0] * 2, 2)
    rows, cols = img.shape[:2]
    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)

    result = [x1, y1, x2, y2]
    return result


image = cv2.imread('slope_test.jpg')  # �̹��� �б�
height, width = image.shape[:2]  # �̹��� ����, �ʺ� ����
gray_img = grayscale(image)  # ����̹����� ��ȯ
cv2.imshow('gray', gray_img)
cv2.imwrite('gray.jpg', gray_img)
blur_img = gaussian_blur(gray_img, 3)  # Blur ȿ��
cv2.imshow('blur', blur_img)
cv2.imwrite('blur.jpg', blur_img)
canny_img = canny(blur_img, 70, 210)  # Canny edge �˰����� ���� ������ �̹��� ����
cv2.imshow('canny', canny_img)
cv2.imwrite('canny.jpg', canny_img)
vertices = np.array(
    [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
    dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices)  # ROI ����
cv2.imshow('ROI', ROI_img)
cv2.imwrite('ROI.jpg', ROI_img)
line_arr = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # ���� ��ȯ���� ���� ã��
line_arr = np.squeeze(line_arr)

# ���� ���ϱ�
slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

# ���� ���� ����
line_arr = line_arr[np.abs(slope_degree) < 160]
slope_degree = slope_degree[np.abs(slope_degree) < 160]
# ���� ���� ����
line_arr = line_arr[np.abs(slope_degree) > 95]
slope_degree = slope_degree[np.abs(slope_degree) > 95]
# ���͸��� ���� ������
L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
L_lines, R_lines = L_lines[:, None], R_lines[:, None]
# ����, ������ ���� ��ǥ�� ���ϱ�
left_fit_line = get_fitline(image, L_lines)
right_fit_line = get_fitline(image, R_lines)
# ��ǥ�� �׸���
draw_fit_line(temp, left_fit_line)
draw_fit_line(temp, right_fit_line)

result = weighted_img(temp, image)  # ���� �̹����� ����� �� overlap
cv2.imshow('result', result)  # ��� �̹��� ���
cv2.imwrite('result.jpg', result)
cv2.waitKey(0)