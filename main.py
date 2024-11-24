import tkinter as tk
import numpy as np

from algorithm_ds import *
from gui import GUI
from game import PS4Controller, Game

import tkinter as tk
from algorithm_ds import MapHandler, PosteriorCalculator, ProbabilityTrainer, DataManager, ImageProcessor
import numpy as np

def main():
    # -----------------
    # 데이터 및 모델 준비
    # -----------------
    print("데이터 및 모델을 로드 중입니다...")
    
    # MapHandler: 맵 파일 로드
    map_file = "./test_map/label01.npy"  # 맵 파일 경로
    map_handler = MapHandler()
    map_array = map_handler.load_map(map_file)

    # 데이터 및 클래스 확률 준비
    sorted_labels = ["0", "1", "3", "5", "6", "9"]
    
    # 샘플 데이터 학습
    base_dir = "data_path"  # 이미지 데이터 폴더
    img_processor = ImageProcessor(img_size=(64, 64), threshold=0.3)
    data_manager = DataManager(base_dir=base_dir, img_processor=img_processor)
    data_by_class = data_manager.load_data_from_folders()
    
    trainer = ProbabilityTrainer(data_by_class)
    trainer.train_pixel_probabilities()
    trainer.calculate_class_priors()
    
    class_pixel_probs = trainer.get_class_pixel_probs()
    class_priors = trainer.get_class_priors()
    
    # PosteriorCalculator: 후행 확률 계산 준비
    posterior_calculator = PosteriorCalculator(class_pixel_probs, class_priors)

    # -----------------
    # GUI 및 게임 준비
    # -----------------
    print("GUI를 초기화 중입니다...")

    # tkinter 기반 GUI 실행
    root = tk.Tk()
    gui = GUI(root)

    # Pygame 기반 게임 초기화
    game = Game(map_file, sorted_labels, class_pixel_probs, class_priors)

    # -----------------
    # 메인 루프 실행
    # -----------------
    print("게임을 시작합니다...")
    def run_game():
        root.withdraw()  # GUI 창 숨기기
        game.run()  # 게임 실행
        root.deiconify()  # 게임 종료 후 GUI 창 다시 표시

    # tkinter 메인 루프 실행
    root.mainloop()

if __name__ == "__main__":
    main()
