#!/usr/bin/etc python

import cv2
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class Recognition:
    def ExtractNumber(self):
        global delta_x
        Number = 'unnamed.jpg'
        # 번호판의 글자에 윤곽선을 잡아주기 위해 OpenCV 라이브러리를 이용해 원본이미지에 전처리 과정을 해줘야 한다.
        img = cv2.imread(Number, cv2.IMREAD_COLOR)
        copy_img = img.copy()
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 이미지 컬러를 Gray 로 바꿔준다

        cv2.imwrite('gray.jpg', img2)
        blur = cv2.GaussianBlur(img2, (3, 3), 0)
        # Gray 이미지를 필터를 적용 시켜 윤곽선을 더 잘 잡을 수 있도록 한다.(엣지검출, 영상처리과정이 수월하도록)
        cv2.imwrite('blur.jpg', blur)

        canny = cv2.Canny(blur, 180, 200)
        cv2.imwrite('canny.jpg', canny)
        # 전처리 사진에서 엣지를 검출한다 (가우시안 필터 이미지에서 Canny Detection 을 사용해서 이미지를 추출한다.)
        # 여기서 엣지란 흑백 영상에서 명암의 밝기 차이에 대한 변화율이다.
        # cv2.Canny 와 cv2.GaussianBlur 의 Threshold1,2의 값은 특정 사진에 맞춰 설정해준 값이다.

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 엣지를 추출한 이미지에서 cv.findContours 사용해 canny 이미지에 대해 Contours(윤곽선)을 찾는다.
        # 여기서 Contours(윤곽선)은 같은 에너지를 가지는 점들을 연결한 선이다.
        # OpenCV는 Contour(윤곽선)을 찾을 때 검은 바탕에 찾는 물체는 흰색으로 설정해야한다.

        box1 = []
        f_count = 0
        select = 0
        plate_width = 0

        for i in range(len(contours)):  # findContours 함수로 찾은 contours(윤곽선)들의 bounding 처리해준다.
            cnt = contours[i]
            area = cv2.contourArea(cnt)  # 폐곡선 형태의 윤곽선으로 둘러싸인 면적
            x, y, w, h = cv2.boundingRect(cnt)  # 윤곽선 cnt 에 외접하는 직사각형의 좌상단 꼭지점 좌표, 가로, 세로 리턴
            rect_area = w * h  # 영역 크기
            aspect_ratio = float(w) / h  # ratio = width/height

            if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 900):
                #  전체 이미지에서 Contour의 가로 세로 비율 값과 면적을 통해, 번호판 영역에 벗어난 걸로 추정되면 제외해준다.
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 외접하는 직사각형 그리기
                box1.append(cv2.boundingRect(cnt))  # 외접하는 직사각형 리스트에 추가

        for i in range(len(box1)):  # 버블 정렬
            for j in range(len(box1) - (i + 1)):
                if box1[j][0] > box1[j + 1][0]:
                    temp = box1[j]
                    box1[j] = box1[j + 1]
                    box1[j + 1] = temp

        # rectangles 사이의 길이를 재서 번호판을 찾는다. for 문을 마친 후 count 값이 가장 큰 직사각형이 번호판의 시작점이다.
        for m in range(len(box1)):
            count = 0
            for n in range(m + 1, (len(box1) - 1)):
                delta_x = abs(box1[n + 1][0] - box1[m][0])
                if delta_x > 150:  # 일정 값 이상의 차이가 나면 종료
                    break
                delta_y = abs(box1[n + 1][1] - box1[m][1])
                if delta_x == 0:  # 기울기를 구하려면 0으로 나눌 순 없기 때문에 1로 대체
                    delta_x = 1
                if delta_y == 0:
                    delta_y = 1
                gradient = float(delta_y) / float(delta_x)  # 직사각형 사이의 tan 값
                if gradient < 0.25:  # 일정 값 미만의 tan 값이면 count 값 증가
                    count = count + 1
            # 번호판 사이즈를 잰다.
            if count > f_count:
                select = m
                f_count = count
                plate_width = delta_x
        cv2.imwrite('snake.jpg', img)

        number_plate = copy_img[box1[select][1] - 10:box1[select][3] + box1[select][1] + 20,
                       box1[select][0] - 50:200 + box1[select][0]]
        # 번호판 사이즈 부분은 상수값으로 offset 줘서 추출한다.

        resize_plate = cv2.resize(number_plate, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC + cv2.INTER_LINEAR)
        plate_gray = cv2.cvtColor(resize_plate, cv2.COLOR_BGR2GRAY)  # 번호판 영역 이미지를 Gray 바꾼다.
        ret, th_plate = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)  # cv2.threshold 흑백 값만 나오도록 이진화 처리를 한다.

        cv2.imwrite('plate_th.jpg', th_plate)
        kernel = np.ones((3, 3), np.uint8)
        er_plate = cv2.erode(th_plate, kernel, iterations=1)  # erode 함수로 검은색 글자 강조
        er_invplate = er_plate
        cv2.imwrite('er_plate.jpg', er_invplate)
        result = pytesseract.image_to_string(Image.open('er_plate.jpg'), lang='kor')  # protester 번호판 인식
        return result.replace(" ", "")


recogtest = Recognition()  # 객체 생성
result = recogtest.ExtractNumber()  # 결과값 생성
print(result)  # 출력
