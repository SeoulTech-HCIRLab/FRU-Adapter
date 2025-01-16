import csv
from collections import Counter

# 감정 갯수를 카운팅하는 함수
def count_emotions(file_path):
    emotions = ["Happy", "Angry", "Disgust", "Sad", "Neutral", "Fear", "Surprise"]
    emotion_counter = Counter()

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            line = row[0] if len(row) > 0 else ""
            for emotion in emotions:
                if f"/{emotion}/" in line:
                    emotion_counter[emotion] += 1

    return emotion_counter

# 결과 출력하기
def main():
    file_path = "./test.csv"  # 여기에 CSV 파일 경로를 입력하세요.
    emotion_counts = count_emotions(file_path)

    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")

if __name__ == "__main__":
    main()