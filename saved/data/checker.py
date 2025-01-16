import csv
from collections import Counter

# 파일 경로 설정
file_path = './FERV39k/all_scenes/train.csv'

# 레이블을 저장할 리스트 생성
labels = []

# CSV 파일 읽기
with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    for row in reader:
        # 마지막 값이 레이블이므로 그 값을 리스트에 추가
        labels.append(row[-1])

# 레이블 개수 세기
label_count = Counter(labels)
# 결과 출력
for label, count in label_count.items():
    print(f"레이블 {label}: {count}개")

# 총 레이블 개수 출력
total_count = sum(label_count.values())
print(f"\n총 레이블 개수: {total_count}개")


# #train 31088개
# 레이블 0: 5854개
# 레이블 1: 5523개
# 레이블 2: 7790개
# 레이블 3: 5906개
# 레이블 4: 2502개
# 레이블 5: 1833개
# 레이블 6: 1680개

# # test 7847
# 레이블 0: 1473개
# 레이블 1: 1393개
# 레이블 2: 1958개
# 레이블 3: 1487개
# 레이블 4: 638개
# 레이블 5: 467개
# 레이블 6: 431개
