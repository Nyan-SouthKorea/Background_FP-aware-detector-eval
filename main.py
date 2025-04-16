# 기본
import os
import json
import copy

# pip install
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# custom
from utils import *

class Eval_Object_Detector:
    '''
    사물 인식 모델의 Conf. Thresh. 별 Precision, Recall 측정
    '''
    def __init__(self, path_gt, path_pred, class_list, path_output):
        '''
        모듈을 처음 실행할 때 기본적으로 입력해야 하는 값

        args:
        path_gt = 원본 레이블 txt가 담겨있는 폴더 경로
            - YOLO label 형식으로 YOLOv7의 test.py에서 --save-txt, --save-conf를 입력했을 때 출력되는 레이블 형식
            - class_no(int) x_center y_center width height로 스페이스바로 구분되며, 다음 bbox는 엔터로 구분됨
        path_pred = YOLO label 형식으로 예측된 파일의 폴더 주소
            - 자세한 설명은 path_gt 형식과 동일함
            - class_no(int) x_center y_center width height confidence로 스페이스바로 구분되며, 다음 bbox는 엔터로 구분됨
        class_list = 객체 리스트 입력
            - e.g. ['cable', 'person', 'cat']
        path_output = 결과를 저장하는 경로
        '''
        print('코드 릴리즈 날짜: 250416_v1')
        print('evaluate() 이용 시 전체적인 평가가 가능합니다.')
        print('1회 평가 이후 draw_bgfp_auto(path_img) 이용 시 Background FP에 대한 사례들을 그려볼 수 있습니다.\n')

        self.path_gt = path_gt
        self.path_pred = path_pred
        self.class_list = class_list
        self.path_output = path_output
        self.iou_thresh_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # 폴더 생성
        makedirs(self.path_output)

        # conf. thresh. 관련 설정
        self.conf_thresh_list = []
        conf_thresh_step = 1000
        for i in range(conf_thresh_step): self.conf_thresh_list.append(i/conf_thresh_step)
        # (첫 번째 conf. thresh.에서의 정보만 기록하기 원할 때 조건문으로 사용)
        self.first_conf_step = (self.conf_thresh_list[0] + self.conf_thresh_list[1])/2

        # 기본적으로 필요한 pred, gt 불러오기
        self.pred, self.gt = self._pred_and_gt_mapping()

        # 추후 confusion matrix를 그리기 위해 선언(self._cal_pr()에서 모든 conf. thrsh.를 기준으로 기록한다.)
        self._make_confusion_matrix_dic()

    def evaluate(self):
        '''
        mAP@0.5, mAP@0.5:0.95, PR 평가 및 그래프 출력 등 모든 기능 전체 자동화
        '''
        # 최종 결과창에 나타나는 images 개수와 labels 개수 출력을 위해 계산
        images, labels, total_images = self.get_images_and_labels()

        # pr과 mAP 계산
        pr_results = self.get_PR()
        best_p, best_r, best_f1, best_conf_thresh = self._get_best_f1score_and_confidence_threshold(pr_results)
        self._draw_confusion_matrix(best_conf_thresh, pr_results)
        mAP_05, mAP_0595 = self.get_mAP()

        # 결과 표를 만들기 위해 all 영역 계산 및 표 만들기
        # (Class)
        df_class = ['all']
        for class_name in self.class_list:
            df_class.append(class_name)
        # (Images)
        df_images = [total_images]
        for image in images.values():
            df_images.append(image)
        # (Labels)
        df_labels = [sum(list(labels.values()))]
        for label in labels.values():
            df_labels.append(label)
        # (P)
        p_list = list(best_p.values())
        df_p = [self.rnd(sum(p_list)/len(p_list))]
        for p in p_list:
            df_p.append(self.rnd(p))
        # (R)
        r_list = list(best_r.values())
        df_r = [self.rnd(sum(r_list)/len(r_list))]
        for r in r_list:
            df_r.append(self.rnd(r))
        # (mAP@0.5)
        mAP_05_list = list(mAP_05.values())
        df_mAP_05 = [self.rnd(sum(mAP_05_list)/len(mAP_05_list))]
        for mAP in mAP_05_list:
            df_mAP_05.append(self.rnd(mAP))
        # (mAP@0.5:0.95)
        mAP_0595_list = list(mAP_0595.values())
        df_mAP_0595 = [self.rnd(sum(mAP_0595_list)/len(mAP_0595_list))]
        for mAP in mAP_0595_list:
            df_mAP_0595.append(self.rnd(mAP))
        # (F1_score)
        f1_list = list(best_f1.values())
        df_f1 = [self.rnd(sum(f1_list)/len(f1_list))]
        for f1 in f1_list:
            df_f1.append(self.rnd(f1))
        # (conf. thresh.)
        conf_thresh_list = list(best_conf_thresh.values())
        df_conf_thresh = [self.rnd(sum(conf_thresh_list)/len(conf_thresh_list))]
        for ct in conf_thresh_list:
            df_conf_thresh.append(self.rnd(ct))
        
        # Pandas DataFrame 만들기
        data = {'Class':df_class,
        'Images':df_images, 
        'Labels':df_labels,
        '      P':df_p,
        '      R':df_r,
        'mAP@.5':df_mAP_05,
        'mAP@.5:.95':df_mAP_0595,
        'F1_score':df_f1,
        'Conf. Thr.':df_conf_thresh,
        }

        df_result = pd.DataFrame(data)

        # 터미널 출력
        print("\n=== Evaluation Results ===\n")
        print(df_result.to_string(index=False))

        # CSV로 저장
        csv_path = os.path.join(self.path_output, 'evaluation_summary.csv')
        df_result.to_csv(csv_path, index=False)
        print(f"\nCSV saved to: {csv_path}")
        
    def _make_confusion_matrix_dic(self):
        '''
        Concusion Matrix를 그릴 수 있도록 데이터를 쌓기 위해 초반에 선언해 놓는 데이터 저장소
        - self._cal_pr()에서 모든 conf. thresh. 를 기준으러 데이터를 쌓아놓고
        - 나중에 그릴 때 self._draw_confusion_matrix()에서 best conf. thresh. 기준으로 걸려서 새로 그린다.
        - 이 방법으 써야지 처음 평가할 때 부터 데이터를 쌓아놓을 수 있음
        '''
        # (2중 dic 속성 먼저 만들기)
        pred_dic = {'background_FP':[]}
        for class_name in self.class_list:
            pred_dic[class_name] = []
        # (main dic 속성 채우기)
        self.cm_gt = {'background_FN':copy.deepcopy(pred_dic)}
        for class_name in self.class_list:
            self.cm_gt[class_name] = copy.deepcopy(pred_dic)

    def _draw_confusion_matrix(self, best_conf_thresh, pr_results):
        '''
        F1 Score가 가장 높을 때의 Conf. Thresh.를 기준으로 Confusion Matrix를 그려서 저장
        '''
        # 클래스 리스트에 background FP/FN 포함하여 확장
        all_classes = ['background_FP'] + self.class_list
        all_classes.append('background_FN')  # 순서: background_FP, cls1, cls2, ..., background_FN

        # y_true, y_pred를 구축
        y_true = []
        y_pred = []

        for gt_class in self.cm_gt:
            for pred_class in self.cm_gt[gt_class]:
                conf_list = self.cm_gt[gt_class][pred_class]
                for conf in conf_list:
                    if conf >= best_conf_thresh[gt_class]:
                        y_true.append(gt_class)
                        y_pred.append(pred_class)


        # FN 데이터 추가
        for class_name, conf_thresh in best_conf_thresh.items():
            fn = pr_results[conf_thresh][class_name]['fn']
            for _ in range(fn):
                y_true.append(class_name)
                y_pred.append('background_FN')



        # confusion matrix 계산
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)

        # 시각화
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_classes, yticklabels=all_classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # 저장
        save_path = os.path.join(self.path_output, f'Confusion_Matrix.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion Matrix saved to: {save_path}")



    def get_images_and_labels(self):
        '''
        최종 결과창에 나타나는 images 개수와 labels 개수 출력을 위해 계산
        '''
        # 비어있는 dic 생성
        tmp = {}
        for class_name in self.class_list:
            tmp[class_name] = 0
        images, labels = copy.deepcopy(tmp), copy.deepcopy(tmp)
        
        # 전체 이미지 개수 계산
        total_images = len(self.gt.keys())

        # 계산(images 개수의 경우 하나의 이미지에 해당 label이 하나라도 포함되어있으면 산정)
        for file_name, bbox_list in self.gt.items():
            # 이미지 내 객체 존재 판단을 위한 bool dic 생성
            class_dic_bool = {}
            for class_name in self.class_list:
                class_dic_bool[class_name] = False
            
            # 기록
            for bbox in bbox_list:
                class_name = bbox['class_name']
                class_dic_bool[class_name] = True
                labels[class_name] += 1
            
            # images 계산 적용
            for class_name, boolen in class_dic_bool.items():
                if boolen == True:
                    images[class_name] += 1
        
        return images, labels, total_images

    def get_mAP(self):
        '''
        mAP 평가 종합 수행
        '''
        # 정답지와 예측지 불러오기      
        pred = copy.deepcopy(self.pred)
        gt = copy.deepcopy(self.gt)

        # 0.5 ~ 0.95까지 모두 담을 dic 선언
        mAP_0595 = {}
        for class_name in self.class_list:
            mAP_0595[class_name] = []

        # mAP@0.5:0.95 구하기  
        for iou_thresh in tqdm(self.iou_thresh_list, desc='mAP@0.5:0.95 계산 중...'):
            precision_dic, recall_dic = {}, {}
            conf_thresh_list = list(reversed(self.conf_thresh_list))
            for conf_thresh in conf_thresh_list:
                results = self._cal_pr(conf_thresh, iou_thresh, pred, gt)
                for class_name, result in results.items():
                    # precision 원소 추가
                    if class_name in precision_dic:
                        precision_dic[class_name].append(result['p'])
                    else:
                        precision_dic[class_name] = [result['p']]
                    
                    # recall 원소 추가
                    if class_name in recall_dic:
                        recall_dic[class_name].append(result['r'])
                    else:
                        recall_dic[class_name] = [result['r']]
            # 사다리꼴 보간법 적용
            for class_name in self.class_list:
                precision_dic[class_name] = self._interpolate_precision(precision_dic[class_name])

            # iou_thresh가 0.5일때 PR 커브 그리기
            if iou_thresh == 0.5:
                self._plot_pr_curve(precision_dic, recall_dic)
            
            # mAP0.5:0.95 구하기
            mAP = self._get_pr_curve(precision_dic, recall_dic)
            for class_name, AP in mAP.items():
                mAP_0595[class_name].append(AP)
        
        # mAP@0.5 구하기
        mAP_05 = {}
        for class_name, mAP_list in mAP_0595.items():
            mAP_05[class_name] = mAP_list[0]
        
        # mAP@0.5:0.95 구하기
        for class_name, mAP_list in mAP_0595.items():
            av_mAP = sum(mAP_list) / len(mAP_list)
            mAP_0595[class_name] = av_mAP
        
        return mAP_05, mAP_0595
            
    def get_PR(self):
        '''
        mAP를 제외한 precision, recall, f1-score 수치 및 그래프 그리기
        '''
        # 정답지와 예측지 불러오기        
        pred = copy.deepcopy(self.pred)
        gt = copy.deepcopy(self.gt)

        # conf_thresh 별로 수치 계산하기
        results = {}
        for conf_thresh in tqdm(self.conf_thresh_list, desc='Precision, Recall 계산 중...'):
            results[conf_thresh] = self._cal_pr(conf_thresh, 0.5, pred, gt)
        
        # 결과 저장
        self._plot_metrics(results)
        with open(f'{self.path_output}/results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        return results

    def _get_best_f1score_and_confidence_threshold(self, results):
        '''
        self.get_PR() 에서 도출된 results를 기반으로 best f1-score에 해당하는 p와 r을 도출
        '''
        # 비어있는 dic 생성
        tmp = {}
        for class_name in self.class_list:
            tmp[class_name] = 0
        best_p, best_r, best_f1, best_conf_thresh = copy.deepcopy(tmp), copy.deepcopy(tmp), copy.deepcopy(tmp), copy.deepcopy(tmp)

        # f1-score가 가장 높은 시점 기록 시작
        for class_name in self.class_list:
            for conf_thres, result in results.items():
                if best_f1[class_name] < result[class_name]['f1']:
                    best_f1[class_name] = result[class_name]['f1']
                    best_p[class_name] = result[class_name]['p']
                    best_r[class_name] = result[class_name]['r']
                    best_conf_thresh[class_name] = conf_thres
        return best_p, best_r, best_f1, best_conf_thresh

    def _get_pr_curve(self, precision_dic, recall_dic):
        '''PR 그래프 만들어서 AP 면적 구하기'''
        aps = {}
        for class_name, precision_list in precision_dic.items():
            recall_list = recall_dic[class_name]
            ap = 0
            for i in range(len(precision_list)):
                # for문 나가는 조건
                if i == len(precision_list)-2: break
                # 사다리꼴 면적 구하면서 더하기(아래 사각형 + 위 삼각형 따로 면적 구해서 더하기)
                # (아래 사각형 면적)
                width = recall_list[i+1] - recall_list[i]
                height = min(precision_list[i], precision_list[i+1])
                ap += (width * height)
                # (위 삼각형)
                # ap += ((width * (max(precision_list[i], precision_list[i+1])-height)) / 2)
            aps[class_name] = ap
        return aps

    def _interpolate_precision(self, precision_list):
        '''precision이 튀는것을 방지하기 위해 점진적으로 감소되는 형태로 변환'''
        new_list = []
        max_no = precision_list[-1]
        for precision in reversed(precision_list):
            if precision < max_no:
                precision = max_no
            else:
                max_no = precision
            new_list.append(precision)
        new_list.reverse()
        return new_list

    def _make_cm_result_dic(self, result, conf, pred_class, gt_class, label_name):
        '''mAP 구하는 양식에 맞게 results 안에 투입되는 하나의 원소를 만들어주는 기능'''
        return {'result':result, 'conf':conf, 'pred_class':pred_class, 'gt_class':gt_class}

    def _cal_pr(self, conf_thresh, iou_thresh, pred, gt):
        '''
        특정 class_name에 대해서 지정된 conf_thresh로 아래의 수치들을 계산해줌
            - precision
            - recall
            - background_fp = 아무것도 없는 배경에 오 인식을 할 확률. (background_fp개수 / 전체 pred 개수)
        '''
        # draw를 위해 background_fp 리스트 모아놓기
        if conf_thresh < self.first_conf_step:
            self.bgfp_list = []
                
        # confidence threshold에 따라 필터링
        pred = self.filter_by_conf_thresh(pred, conf_thresh)

        # class_list 기준 초기 카운터 설정
        results = {}
        for class_name in self.class_list:
            results[class_name] = {'tp': 0, 'fp': 0, 'fn': 0, 'bg_fp': 0, 'total_gt': 0, 'total_pred': 0}
        
        # 계산 시작
        for key in pred.keys():
            pred_bbox_list = pred[key]
            gt_bbox_list = gt[key]

            # gt 데이터에 matched 플래그 추가
            for i in range(len(gt_bbox_list)):
                gt_bbox_list[i]['matched'] = False
            
            # pred로 순회하며 tp, fp 검출
            for pred_bbox_data in pred_bbox_list:
                # pred 원소 하나를 빼서
                pred_class_name = pred_bbox_data['class_name']
                conf = pred_bbox_data['conf']
                pred_bbox = pred_bbox_data['bbox']
                # gt 원소와 하나씩 비교한다.
                found = False
                for i, gt_bbox_data in enumerate(gt_bbox_list):
                    gt_class_name = gt_bbox_data['class_name']
                    gt_bbox = gt_bbox_data['bbox']
                    # iou 매칭 여부 확인
                    if get_iou(pred_bbox, gt_bbox) >= iou_thresh:
                        # confusion matrix 기록
                        if conf_thresh < self.first_conf_step:
                            self.cm_gt[gt_class_name][pred_class_name].append(conf)
                        # 객체 매칭 여부 확인
                        if pred_class_name == gt_class_name:
                            gt_bbox_list[i]['matched'] = True
                            found = True
                            results[pred_class_name]['tp'] += 1
                            break
                if found == False:
                    results[pred_class_name]['fp'] += 1
                    # background_fp 검사
                    if self._check_background_fp(pred_bbox_data, gt_bbox_list) == True:
                        results[pred_class_name]['bg_fp'] += 1
                        if conf_thresh < self.first_conf_step:
                            self.bgfp_list.append(pred_bbox_data)
                            self.bgfp_list[-1]['label_name'] = key
                            # confusion matrix 기록
                            self.cm_gt[gt_class_name]['background_FP'].append(conf)

            # pred 순회 끝난 후 fn 계산
            for gt_bbox_data in gt_bbox_list:
                if gt_bbox_data['matched'] == False:
                    results[gt_bbox_data['class_name']]['fn'] += 1
        
        # total_pred와 total_gt 계산
        # (pred)
        for bbox_list in pred.values():
            for bbox in bbox_list:
                class_name = bbox['class_name']
                results[class_name]['total_pred'] += 1
        # (gt)
        for bbox_list in gt.values():
            for bbox in bbox_list:
                class_name = bbox['class_name']
                results[class_name]['total_gt'] += 1


        # 최종 precision, recall, bg_fp 비율 계산
        for class_name, result in results.items():
            tp = result['tp']
            fp = result['fp']
            fn = result['fn']
            bg_fp = result['bg_fp']
            result['bg_fp_cnt'] = result['bg_fp']
            total_pred = result['total_pred']
            total_gt = result['total_gt']
            results[class_name]['p'] = round(self._get_precision(tp, fp), 3)
            results[class_name]['r'] = round(self._get_recall(tp, fn), 3)
            results[class_name]['f1'] = (2 * results[class_name]['p'] * results[class_name]['r']) / max(0.00000001, (results[class_name]['p'] + results[class_name]['r']))
            results[class_name]['bg_fp'] = round(self._get_background_fp(bg_fp, total_pred), 3)

        return results


    def _check_background_fp(self, pred_bbox_data, gt_bbox_list):
        '''
        pred 1개와 gt_bbox_list를 넣었을 때, background_fp 여부를 알려주는 함수
        - 입력되는 pred_bbox_data는 fp여야 함
        - pred bbox의 50%만 gt_bbox에 겹쳐있어도 background_fp가 아닌걸로 인정
        - classification이 틀려도 localization만 맞으면 된다는 취지
        '''
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox_data['bbox']
        pred_bbox_area = (pred_x2-pred_x1) * (pred_y2-pred_y1)
        found = False
        for gt_bbox_data in gt_bbox_list:
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox_data['bbox']
            # pred_bbox 입장에서 겹치는 면적 도출
            x1 = max(pred_x1, gt_x1)
            y1 = max(pred_y1, gt_y1)
            x2 = min(pred_x2, gt_x2)
            y2 = min(pred_y2, gt_y2)
            if (x2-x1) <= 0 or (y2-y1) <= 0:
                continue
            inter_area = (x2-x1) * (y2-y1)

            # 겹치는 면적이 50% 이상인지 확인
            if inter_area / pred_bbox_area >= 0.5:
                found = True
        
        # 결과 반환
        if found == True:
            return False
        else:
            return True

    def _plot_metrics(self, results):        
        # confidence threshold들을 float형으로 오름차순 정렬 (결과 dict의 키는 float형임)
        conf_thresholds = sorted([float(th) for th in results.keys()])
        
        # 측정할 메트릭과 레이블, 저장 파일명을 딕셔너리로 정의
        metrics = {
            'p': ('Precision', 'Precision_curve.png'),
            'r': ('Recall', 'Recall_curve.png'),
            'bg_fp': ('Background FP', 'Background FP_curve.png'),
            'f1':('F1-Score', 'F1-score_curve.png')
        }
        
        # 각 메트릭에 대해 그래프 생성
        for metric_key, (metric_label, filename) in metrics.items():
            plt.figure(figsize=(12, 10))
            # 각 클래스별로 선 그리기 (라인 색상은 자동 할당되고, 범례에 표시됨)
            for cls in self.class_list:
                values = []
                for conf in conf_thresholds:
                    values.append(results[conf][cls][metric_key])
                plt.plot(conf_thresholds, values, label=cls, linewidth=2.0)

            # 평균선
            mean_values = []
            for conf in conf_thresholds:
                v_list = [results[conf][cls][metric_key] for cls in self.class_list]
                mean = sum(v_list) / len(v_list)
                mean_values.append(mean)
            plt.plot(conf_thresholds, mean_values, label='mean', linestyle='--', color='black', lw=3)
            
            plt.xlabel('Confidence Threshold')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} Curve')
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 1.0))
            plt.grid(True)
            plt.savefig(os.path.join(self.path_output, filename))
            plt.close()

    def _plot_pr_curve(self, precision_dic, recall_dic):
        plt.figure(figsize=(12, 10))
        
        # 클래스별 PR 곡선
        for cls in self.class_list:
            precision = precision_dic[cls]
            recall = recall_dic[cls]
            plt.plot(recall, precision, label=cls)

        # 평균 PR 곡선 계산
        num_points = len(next(iter(precision_dic.values())))  # 모든 클래스가 동일 길이 가정
        mean_precision = []
        mean_recall = []

        for i in range(num_points):
            p_list = [precision_dic[cls][i] for cls in self.class_list]
            r_list = [recall_dic[cls][i] for cls in self.class_list]
            mean_precision.append(sum(p_list) / len(p_list))
            mean_recall.append(sum(r_list) / len(r_list))

        # 평균 PR 선 추가
        plt.plot(mean_recall, mean_precision, label='mean', linestyle='--', color='black', linewidth=3)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(os.path.join(self.path_output, 'PR_curve.png'))
        plt.close()

    def filter_by_conf_thresh(self, annotations, conf_thresh):
        '''
        pred에서 만족하는 conf. thresh.만 남기고 모두 삭제
        '''
        new_annotations = {}
        for file_name, bbox_list in annotations.items():
            new_bbox_list = []
            for info in bbox_list:
                if info['conf'] >= conf_thresh:
                    new_bbox_list.append(info)
            new_annotations[file_name] = new_bbox_list
        return new_annotations
    
    def _get_precision(self, tp, fp):
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    def _get_recall(self, tp, fn):
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def _get_background_fp(self, bg_fp, total_pred):
        if total_pred == 0:
            return 0
        else:
            return bg_fp / total_pred
        
    def _get_annotations(self, path_annotations_folder, desc_txt):
        '''
        어노테이션이 들어있는 폴더 경로를 입력하면 파일명에 맞게 bbox_list가 담긴 dic으로 변환하여 반환
        '''
        dic_bbox_list = {}
        for annotations_file_name in tqdm(natsorted(os.listdir(path_annotations_folder)), desc=f'get annotation({desc_txt})'):
            dic_bbox_list[annotations_file_name] = self._get_annotation(f'{path_annotations_folder}/{annotations_file_name}')
        return dic_bbox_list

    def _get_annotation(self, path_annotation_file):
        '''
        어노테이션 경로를 입력하면 읽어서 리스트로 반환.
            - bbox_list로 반환하며 confidence가 있는 경우는 마지막 인자에 하나 더 추가하여 반환함
            - 기본 form: [{'class_name':'person', 'class_no':0, 'bbox':[x1, y1, x2, y2], 'conf':0.75}]
                1) conf는 경우에 따라 자동으로 감지하여 있을수도 있고 없을수도 있음
        
        args:
        path_annotation_file = 파일 경로

        return: 
        bbox_list
        '''
        with open(path_annotation_file, 'r', encoding='utf-8') as f:
            full_txt = f.read()
            split_by_enter = full_txt.split('\n')
            if split_by_enter[-1] == '':
                del split_by_enter[-1]
        bbox_list = []
        for one_enter in split_by_enter:
            split_by_space = one_enter.split(' ')
            class_no = int(split_by_space[0])
            class_name = self.class_list[class_no]
            b1, b2, b3, b4 = split_by_space[1:5]
            b1, b2, b3, b4 = float(b1), float(b2), float(b3), float(b4)
            x1, y1, x2, y2 = yolo_to_x1y1x2y2([b1, b2, b3, b4])
            bbox = {'class_name': class_name, 'class_no': class_no, 'bbox': [x1, y1, x2, y2]}
            # confidence는 인덱스 5에 위치함 (정확한 값 사용)
            if len(split_by_space) == 6:
                bbox['conf'] = float(split_by_space[5])
            bbox_list.append(bbox)
        return bbox_list

    def _pred_and_gt_mapping(self):
        '''
        pred와 gt를 불러와서 서로 이빨 빠진 곳 채워넣어서 매핑 시켜서 반환
        '''
        # 정답지와 예측지 불러오기
        pred = self._get_annotations(self.path_pred, 'pred 불러오는 중')
        gt = self._get_annotations(self.path_gt, 'gt 불러오는 중')

        # 예측지의 개수가 정답지보다 적으면 빠진 이빨 채워넣기
        if len(pred.keys()) != len(gt.keys()):
            pred = self._make_empty_pred(gt, pred)

        return pred, gt

    def _make_empty_pred(self, dic_gt_bbox_list, dic_pred_bbox_list):
        '''
        로직 오류를 막기 위해 pred 결과가 없어도 빈 감지 결과를 넣어주는 로직

        args:
        dic_gt_bbox_list = 정답지 bbox_list가 담긴 dic
        dic_pred_bbox_list = 추론 결과 bbox_list가 담긴 dic

        return:
        dic_pred_bbox_list = dic_gt_bbox_list과 길이를 맞춘 empty 추론 결과가 추가된 dic
        '''
        for file_name, bbox_list in dic_gt_bbox_list.items():
            if file_name not in dic_pred_bbox_list:
                dic_pred_bbox_list[file_name] = []
        return dic_pred_bbox_list

    def draw_bgfp_auto(self, path_img):
        '''
        자동으로 class_name과 conf_thresh 기준으로 Background FP를 그려주는 함수
        '''        
        # for conf_thresh in tqdm(self.conf_thresh_list):
        for conf_thresh in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            for class_name in self.class_list:
                self._draw_bgfp(path_img, class_name, conf_thresh)
                
    def _draw_bgfp(self, path_img, class_name, conf_thresh):
        '''
        Background FP만 그려서 반환해주는 기능
        - 반드시 auto_run()이 선행되어야 함. 그래야 self.bgfp_list 안에 bg_fp 성분들을 채울 수 있음
        '''
        # 폴더 생성
        write_path = f'{self.path_output}/draw_bgfp/{str(conf_thresh)}/{class_name}'
        makedirs(write_path)

        # 데이터 유효성 체크
        try:
            # print(f'self.bgfp_list 개수: {len(self.bgfp_list)}')
            pass
        except:
            print('self.bgfp_list가 아직 선언되지 않았습니다. 반드시 auto_run()을 먼저 실행해주세요.')
            return
        
        # conf_thresh 기준 필터링
        filtered_list = []
        for result in self.bgfp_list:
            if result['conf'] >= conf_thresh and result['class_name'] == class_name:
                filtered_list.append(result)
        
        # label_name 기준 그룹핑
        filtered = {}
        for result in filtered_list:
            if result['label_name'] in filtered:
                filtered[result['label_name']].append(result)
            else:
                filtered[result['label_name']] = [result]
        
        # 이미지 불러오기
        full_img_list, name_img_list = [], []
        for img_name in listdir(path_img):
            full_img_list.append(img_name)
            name_img_list.append(name(img_name))

        # 그리기
        for label_name, results in filtered.items():
            # 이미지 존재 유무 확인
            if name(label_name) in name_img_list:
                img_idx = name_img_list.index(name(label_name))
                img_name = full_img_list[img_idx]
            else:
                print(f'이미지 매칭 안됨: {label_name}')
                continue

            # gt 그리기
            for gt_name in listdir(self.path_gt):
                if name(gt_name) == name(img_name):
                    img = draw(img_path=f'{path_img}/{img_name}', label_path=f'{self.path_gt}/{gt_name}', write_path='return', color=[0,255,0])
                    break

            # 그리기
            bbox_list = []
            for result in results:
                class_name = result['class_name']
                x1, y1, x2, y2 = result['bbox']
                b1, b2, b3, b4 = x1y1x2y2_to_yolo([x1, y1, x2, y2])
                conf = result['conf']
                bbox_list.append([class_name, b1, b2, b3, b4, conf])
            draw(img_path=img, label_path=bbox_list, write_path=f'{write_path}/{img_name}')

    def rnd(self, no):
        '''소수점 자리수 맞춰서 반환'''
        round_no = 3
        return round(no, round_no)