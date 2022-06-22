## Evaluation 사용법

> eval_test.py 를 실행하면 inference output 경로(folder_with_pred)에 summary.json 파일이 생성된다. (시간이 좀 소요된다)
summary.json에는 각각의 테스트 케이스마다 dice, prediction, recall 등의 결과가 클래스마다 나오며 가장 아래에는 전체 테스트셋의 평균값이 나온다.

> eval_test.py를 사용할때 input이미지와 비교하려는 이미지의 이름이 같아야한다!
