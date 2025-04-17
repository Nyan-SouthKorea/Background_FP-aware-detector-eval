# base
import os
import json
import random
import time
import shutil
import copy

# pip install
from natsort import natsorted
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from IPython.display import display 


def listdir(path):
    return natsorted(os.listdir(path))

def makedirs(path):
    os.makedirs(path, exist_ok=True)

def get_iou(bbox_a, bbox_b):
    '''
    입력된 bbox 2개 대하여 iou 출력
    - bbox 포맷: [x1, y1, x2, y2]
    - 어차피 비율을 구하는 식이기 때문에 정규화 여부는 상관 없음
    '''
    a_x1, a_y1, a_x2, a_y2 = bbox_a
    b_x1, b_y1, b_x2, b_y2 = bbox_b
    # 작은 박스 구하기
    small_x1 = max(a_x1, b_x1)
    small_y1 = max(a_y1, b_y1)
    small_x2 = min(a_x2, b_x2)
    small_y2 = min(a_y2, b_y2)
    width_small = small_x2 - small_x1
    height_small = small_y2 - small_y1
    if width_small <= 0 or height_small <= 0: # 박스가 겹치지 않는다
        return 0.0
    area_small = width_small * height_small

    # box a 면적 구하기
    width_a = a_x2 - a_x1
    height_a = a_y2 - a_y1
    area_a = width_a * height_a
    
    # box b 면적 구하기
    width_b = b_x2 - b_x1
    height_b = b_y2 - b_y1
    area_b = width_b * height_b

    # IOU 구하기
    iou_down = area_a + area_b - area_small
    if iou_down == 0:
        return 0.0
    iou = area_small / iou_down
    return round(iou, 5)

# YOLO 레이블 읽고 저장하기
def read_label(label_path):
    '''
    txt로 된 YOLO label을 읽어서 bbox 형태로 반환

    label_path: 레이블 경로
    return: [[class_info, b1, b2, b3, b4], ...,]
    '''
    bbox_list = []
    with open(label_path, 'r', encoding='utf-8-sig') as f:
        full_txt = f.read()
    txt_list = full_txt.split('\n')
    if len(txt_list[-1]) == 0:
        del txt_list[-1]
    for txt in txt_list:
        splited_txt = txt.split(' ')
        # 인자 5개, 6개 예외 처리
        if len(splited_txt) == 5:
            class_info, b1, b2, b3, b4 = txt.split(' ')
            b1, b2, b3, b4 = float(b1), float(b2), float(b3), float(b4)
            bbox_list.append([class_info, b1, b2, b3, b4])
        elif len(splited_txt) == 6:
            class_info, b1, b2, b3, b4, conf = txt.split(' ')
            b1, b2, b3, b4, conf = float(b1), float(b2), float(b3), float(b4), float(conf)
            bbox_list.append([class_info, b1, b2, b3, b4, conf])
        elif len(splited_txt) == 1:
            pass
        else:
            print(f'bbox 인자 개수 이상:{len(splited_txt)}')
    return bbox_list

def write_label(label_path, bbox_list):
    '''
    [class_info, b1, b2, b3, b4] 형식의 YOLO label list를 txt로 저장
    '''
    # label_path와 bbox_list 순서가 중간에 바뀌었지만 이전 코드 양이 많아서 둘 다 호환되도록 변경 - 250326
    if isinstance(label_path, str) and isinstance(bbox_list, list):
        pass
    elif isinstance(label_path, list) and isinstance(bbox_list, str):
        print('label_path와 bbox_list 입력 순서가 바뀌었습니다. 자동 보정됩니다.')
        tmp_label_path = copy.deepcopy(bbox_list)
        tmp_bbox_list = copy.deepcopy(label_path)
        label_path = tmp_label_path
        bbox_list = tmp_bbox_list
    else:
        print('label_path와 bbox_list 타입 입력 잘못 함')

    # 로직 시작
    with open(label_path, 'w') as f:
        f.write('')
        for i, bbox in enumerate(bbox_list):
            if i == len(bbox_list)-1: enter = ''
            else: enter = '\n'
            # 인자 5개, 6개 예외 처리
            if len(bbox) == 5:
                class_info, b1, b2, b3, b4 = bbox
                f.write(f'{class_info} {b1} {b2} {b3} {b4}{enter}')
            elif len(bbox) == 6:
                class_info, b1, b2, b3, b4, conf = bbox
                f.write(f'{class_info} {b1} {b2} {b3} {b4} {conf}{enter}')
            else:
                print(f'bbox 인자 개수 이상: {len(bbox)}')

class count_bbox:
    def __init__(self, path_label_folder, show_mode=True):
        '''
        bbox개수를 class별로 산정
            - 배경 데이터 개수를 알 기 위해서는 반드시 레이블이 없는 이미지에는, 빈 레이블 파일이 존재해야 함
            - 레이블 방식은 YOLO와 동일하지만, class_no가 class_name으로 변경된 개념

        path_label_folder: label.txt가 들어있는 폴더. train, val, test가 들어있는 경로로 선택해도 알아서 인식함
        '''
        # 폴더 경로 파악
        # 1) 바로 레이블이 있거나
        # 2) train, val, test 폴더가 있거나
        label_path_list = []
        file_list = listdir(path_label_folder)
        if 'train' in file_list or 'val' in file_list or 'test' in file_list:
            print('경로 안에 train, val, test 폴더 발견')
            for mode in ['train', 'val', 'test']:
                try:
                    for label_name in listdir(f'{path_label_folder}/{mode}/labels'):
                        label_path_list.append(f'{path_label_folder}/{mode}/labels/{label_name}')
                except:
                    print(f'{mode} skip')
        else:
            for label_name in listdir(path_label_folder):
                label_path_list.append(f'{path_label_folder}/{label_name}')

        # 개수 카운트
        cnt_dic, background_cnt = {}, 0
        for label_path in tqdm(label_path_list):
            bbox_list = read_label(label_path)
            if len(bbox_list) == 0:
                background_cnt += 1
            else:
                for bbox in bbox_list:
                    class_info = bbox[0]
                    if class_info in cnt_dic:
                        cnt_dic[class_info] += 1
                    else:
                        cnt_dic[class_info] = 1
        cnt_dic = dict(sorted(cnt_dic.items()))
        cnt_dic['[BACKGROUND]'] = background_cnt
        if show_mode:
            print('바운딩 박스 Print([BACKGROUND]는 이미지 개수임)')
            # 전체 바운딩박스 개수 파악
            total_bbox = 0
            for ea in cnt_dic.values():
                total_bbox += ea
            print(f'전체 이미지:{len(label_path_list)}장 | 전체 bbox:{total_bbox}')
            class_list_for_print = list(cnt_dic.keys())
            class_list_for_print.remove('[BACKGROUND]')
            print(f'객체 리스트(순서 개념 없음): {class_list_for_print}')
            self._make_bar(cnt_dic)
        else:
            self.cnt_dic = cnt_dic

    def _make_bar(self, cnt_dic):
        '''
        key와 카운터로 이루어진 딕셔너리의 개수를 시각화 하여 bar로 나타낸다.
        '''
        # 맨 아래 Total 추가
        cnt_dic['[Total]'] = sum(list(cnt_dic.values()))

        # 퍼센테이지 Tab 추가
        percentage = []
        for ea in cnt_dic.values():
            percent = round(ea / cnt_dic['[Total]'] * 100, 2)
            per = f"{percent}%"
            percentage.append(per)

        # 최대 개수 설정
        max_cnt = max(cnt_dic.values()) # 최대 개수 찾기
        max_bar_len = 100 # bar의 최대 길이 설정

        # 그래프 출력
        cnt = 0
        for class_name, ea in cnt_dic.items():
            bar_len = int((ea/max_cnt) * max_bar_len)
            if bar_len == 0:
                bar_len = 1
            bar = '█' * bar_len
            # 결과 출력
            print(f'{cnt}: {bar} {class_name}:{ea}')
            cnt += 1
        df = pd.DataFrame({'Class Name':cnt_dic.keys(), 'EA':list(cnt_dic.values()), '%':percentage})
        display(df)
        df.to_csv(f'./count_bbox.csv')
        print('데이터가 csv로 저장되었습니다: ./count_bbox.csv')

# 바운딩박스 조작
def bbox_fix(b1, b2, b3, b4):
    '''bbox가 이미지 밖으로 나가지 않도록 수정'''
    x_center, y_center, x_len, y_len = float(b1), float(b2), float(b3), float(b4)
    x1, y1 = x_center-(x_len/2), y_center-(y_len/2)
    x2, y2 = x1+x_len, y1+y_len
    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
    x1, y1, x2, y2 = min(1, x1), min(1, y1), min(1, x2), min(1, y2)
    x2, y2 = max(x1, x2), max(y1, y2) # x2, y2가 최소한 x1, y2보다 작은 현상 방지
    b3, b4 = x2-x1, y2-y1 # x, y 길이
    b1, b2 = x1+(b3/2), y1+(b4/2) # x, y 중간 포인트
    return b1, b2, b3, b4

def x1y1x2y2_to_yolo(bbox):
    '''
    [x1, y1, x2, y2] -> [center x, center y, x width, y width] 변환
    bbox: [x1, y1, x2, y2]
    '''
    x1, y1, x2, y2 = bbox
    center_x = (x1+x2)/2
    center_y = (y1+y2)/2
    width = x2-x1
    height = y2-y1
    # rnd = 5
    # center_x, center_y, width, height = round(center_x, rnd), round(center_y, rnd), round(width, rnd), round(height, rnd)
    bbox = [center_x, center_y, width, height]
    return bbox

def yolo_to_x1y1x2y2(bbox):
    '''
    [center x, center y, x width, y width] -> [x1, y1, x2, y2] 변환
    bbox: [center x, center y, x width, y width]
    '''
    center_x, center_y, width, height = bbox
    x1 = center_x - (width/2)
    y1 = center_y - (height/2)
    x2 = x1 + width
    y2 = y1 + height
    bbox = [x1, y1, x2, y2]
    return bbox

def bbox_pix_to_nor(bbox, w, h):
    '''
    [x, y, x, y] 픽셀값 bbox를 0 ~ 1 정규화된 상태로 변환
    bbox: 입력되는 bbox
    w, h: 이미지 사이즈
    '''
    b1, b2, b3, b4 = bbox
    b1, b2, b3, b4 = b1/w, b2/h, b3/w, b4/h
    # round_no = 5
    # b1, b2, b3, b4 = round(b1, round_no), round(b2, round_no), round(b3, round_no), round(b4, round_no)
    return [b1, b2, b3, b4]

def bbox_nor_to_pix(bbox, w, h):
    '''
    [x, y, x, y] 4개의 원소를 가진 리스트를 int -> f
    w, h: 이미지 사이즈
    '''
    b1, b2, b3, b4 = bbox
    b1, b2, b3, b4 = b1*w, b2*h, b3*w, b4*h
    b1, b2, b3, b4 = int(b1), int(b2), int(b3), int(b4)
    return [b1, b2, b3, b4]

def draw(img_path, label_path, write_path, color=[0,0,255]):
    '''
    YOLO 형식의 데이터셋 그려서 저장
    img_path: 이미지 경로
    label_path: 레이블 경로
    write_path: 이미지 저장 경로
    '''
    thick, txt_size =1, 0.7
    # 이미지 읽기
    if isinstance(img_path, str):
        img = imread_kr(img_path)
    else:
        img = img_path
    img = smart_resize(img)
    h, w, c = img.shape

    # 레이블 읽기
    if isinstance(label_path, str):
        bbox_list = read_label(label_path)
    else:
        bbox_list = label_path
    
    # 그리기
    for bbox in bbox_list:
        if len(bbox) == 5:
            class_name, b1, b2, b3, b4 = bbox
            conf = ''
        elif len(bbox) == 6:
            class_name, b1, b2, b3, b4, conf = bbox
            conf = f': {round(float(conf), 2)}'
        else:
            print('bbox 인자가 5개 or 6개가 아닌 다른 무언가입니다 ㅠㅠ')
        pixel_bbox_nor = yolo_to_x1y1x2y2([b1, b2, b3, b4])
        pixel_bbox = bbox_nor_to_pix(pixel_bbox_nor, w, h)
        x1, y1, x2, y2 = pixel_bbox
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
        cv2.putText(img, f'{class_name}{conf}', (x1,y2-3), cv2.FONT_HERSHEY_SIMPLEX, txt_size, color, thick)
    
    # 저장 혹은 return
    if write_path == 'return':
        return img
    else:
        imwrite_kr(write_path, img)

def draw_auto(path_data):
    '''데이터 경로를 입력하면 자동으로 그려준다. 경로 형식은 YOLO와 동일하게 해야한다.'''
    img_list = listdir(f'{path_data}/images')
    goal = int(input(f'검출된 이미지 개수: {len(img_list)}, 몇 장 그릴지 입력: '))
    # 폴더 생성
    makedirs(f'{path_data}/draw')
    # 랜덤 셔플 후 그리기
    random.shuffle(img_list)
    for img_name in tqdm(img_list[:goal]):
        img_path = f'{path_data}/images/{img_name}'
        label_path = f'{path_data}/labels/{name(img_name)}.txt'
        write_path = f'{path_data}/draw/{img_name}'
        draw(img_path, label_path, write_path)


def draw_interest(path_data, interest_class_list, conf_thresh=0.0):
    '''
    관심 있는 class만 그리기

    path_data: 데이터 경로(경로 내에 iamges, labels 폴더가 있어야 하고 label 형식은 YOLO을 따름(class_name 적혀 있어도 됨))
    class_list: 관심 있는 class만
    conf_thresh: pred 결과를 그릴 경우 설정 가능
    '''
    thick, txt_size = 1, 0.7
    red, black = [0,0,255], [0,0,0]

    # 폴더 생성
    for idx, class_name in enumerate(interest_class_list):
        # 폴더명 만들기
        if idx == 0:
            folder = class_name
        else:
            folder = f'{folder}, {class_name}'
    makedirs(f'{path_data}/draw_interest_output/{folder}')

    # 그리기
    for img_name in tqdm(listdir(f'{path_data}/images')):
        label_name = f'{name(img_name)}.txt'
        label_path = f'{path_data}/labels/{label_name}'
        # 관심 있는 객체만 그리기
        bbox_list = read_label(label_path)
        for bbox in bbox_list:
            if bbox[0] in interest_class_list:
                # 관심 객체만 존재하는 bbox_list 만들기
                new_bbox_list = []
                for bbox in bbox_list:
                    # 색상 정보 추가
                    # (gt인 경우와 pred인 경우 예외처리)
                    if len(bbox) == 5:
                        if bbox[0] in interest_class_list:
                            bbox.append(red)
                        else:
                            bbox.append(black)
                    elif len(bbox) == 6:
                        if bbox[0] in interest_class_list and float(bbox[5]) >= conf_thresh:
                            bbox.append(red)
                        else:
                            bbox.append(black)
                    else:
                        print('bbox 인자가 5개 or 6개가 아닌 다른 무언가입니다 ㅠㅠ')
                    new_bbox_list.append(bbox)
                
                # 그리기
                img_path = f'{path_data}/images/{img_name}'
                write_path = f'{path_data}/draw_interest_output/{folder}/{img_name}'

                # 이미지 읽기
                img = smart_resize(imread_kr(img_path))
                h, w, c = img.shape

                # 원하는 색깔로 그리기 위해서 그리기 코드 별도 구현
                for bbox in new_bbox_list:
                    # (gt인 경우와 pred인 경우 예외처리)
                    if len(bbox) == 6:
                        class_name, b1, b2, b3, b4, color = bbox
                        conf = ''
                    elif len(bbox) == 7:
                        class_name, b1, b2, b3, b4, conf, color = bbox
                        conf = f': {round(float(conf), 2)}'
                    else:
                        print('bbox 인자가 5개 or 6개가 아닌 다른 무언가입니다 ㅠㅠ')
                    pixel_bbox_nor = yolo_to_x1y1x2y2([b1, b2, b3, b4])
                    pixel_bbox = bbox_nor_to_pix(pixel_bbox_nor, w, h)
                    x1, y1, x2, y2 = pixel_bbox
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
                    cv2.putText(img, f'{class_name}{conf}', (x1,y2-3), cv2.FONT_HERSHEY_SIMPLEX, txt_size, color, thick)
                imwrite_kr(write_path, img)
                break


def read_json(path):
    '''
    path: json 경로
    return: json에서 읽은 dic or list
    '''
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def name(file_name):
    '''
    file_name: 확장자가 존재하는 파일
    return 확장자를 제외한 파일의 이름
    '''
    if '/' in file_name:
        file_name = file_name.split('/')[-1]
    txt_list = file_name.split('.')
    format_len = len(txt_list[-1]) + 1
    file_name = file_name[:-format_len]
    return file_name

def smart_resize(img, max_size=1280):
    '''
    최대 변의 길이를 맞추면서 비율을 유지하여 이미지 리사이즈
    img: cv2 이미지
    max_size: 최대 크기
    return: resize된 cv2 이미지 반환
    '''
    h, w, c = img.shape
    # 이미 지정 사이즈보다 이미지가 작으면 그냥 반환
    if max(h, w) <= max_size:
        return img

    # 리사이즈 진행
    if w > h:
        img = cv2.resize(img, (max_size, int(h/w*max_size)))
    else:
        img = cv2.resize(img, (int(w/h*max_size), max_size))
    return img




# 250402_여동훈 추가
def imread_kr(img_path):
    '''윈도우 아나콘다 환경에서 한국어 경로를 입력하면 이미지로 읽어서 cv2 로 반환'''
    try:
        return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        return cv2.imread(img_path)

def imwrite_kr(write_path, img):
    '''윈도우 아나콘다 환경에서 한국어 경로를 입력하면 다양한 이미지 형식으로 출력 가능'''
    try:
        ext = '.' + write_path.split('.')[-1]  # ".jpg", ".png" 형태로 만들어줌
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            encoded_img.tofile(write_path)
    except:
        cv2.imwrite(write_path, img)