import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from time import time
import sys
from datasets import Dataset

train = pd.read_csv('/home/work/skku_train/jy/boaz/hyo_train.csv')
test = pd.read_csv('/home/work/skku_train/jy/boaz/hyo_test.csv')

data = train.copy()
train_df, test_df = train_test_split(data, test_size=0.2, random_state=90, stratify=data['FLAG'])


def model_train(model_version):
    # 학습 시작 시간 기록
    start_time = time()

    # AutoGluon Predictor 설정
    predictor = TabularPredictor(label='FLAG', eval_metric='f1', path=model_version) 

    # 실시간 진행 상황을 확인할 수 있도록 fit_summary 옵션 사용
    print("모델 학습을 시작합니다. 진행 상황과 남은 시간을 모니터링할 수 있습니다...")

    # 모델 학습
    predictor.fit(
        train_data=train_df,
        time_limit=600,  # 10분 제한
        presets='best_quality',  # 높은 성능을 위한 설정
        verbosity=3,  # 실시간 로그 출력 수준 설정,
        ag_args_fit={'num_gpus': 1}
    )

    # 학습 종료 시간 기록
    end_time = time()
    elapsed_time = end_time - start_time

    # 학습 요약 정보 출력
    summary = predictor.fit_summary()

    # 예측 및 평가
    predictions = predictor.predict(test_df)
    test_df['predicted_label'] = predictions

    # 성능 평가
    from sklearn.metrics import classification_report
    report = classification_report(test_df['FLAG'], test_df['predicted_label'])
    print("\n모델 성능 평가 결과:")
    print(report)

    # 학습 시간 출력
    print(f"\n모델 학습 완료! 총 학습 시간: {elapsed_time // 60}분 {elapsed_time % 60:.2f}초")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <save_path>")
        sys.exit(1)
    
    model_version = sys.argv[1]
    model_train(model_version)