# gui_ds.py

import tkinter as tk

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Game GUI")

        # 창 크기 설정
        self.root.geometry("600x400")
        
        # 프레임 구성 (2x2 격자)
        self.create_frames()

    def create_frames(self):
        # 2x2 그리드 생성
        self.frame_ai_game = tk.Frame(self.root, bg="blue", width=300, height=200)
        self.frame_human_game = tk.Frame(self.root, bg="blue", width=300, height=200)
        self.frame_ai_prediction = tk.Frame(self.root, bg="blue", width=300, height=200)
        self.frame_answer_select = tk.Frame(self.root, bg="blue", width=300, height=200)
        
        # 각 프레임 배치
        self.frame_ai_game.grid(row=0, column=0, sticky="nsew")
        self.frame_human_game.grid(row=0, column=1, sticky="nsew")
        self.frame_ai_prediction.grid(row=1, column=0, sticky="nsew")
        self.frame_answer_select.grid(row=1, column=1, sticky="nsew")

        # 창의 행과 열 크기 조정
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # 각 프레임 내부에 텍스트 추가
        tk.Label(self.frame_ai_game, text="AI 게임 화면", bg="blue", fg="white").pack(expand=True)
        
        tk.Label(self.frame_human_game, text="사람 게임 화면", bg="blue", fg="white").pack(expand=True)
        tk.Label(self.frame_ai_prediction, text="AI의 확률 예측 그래프", bg="blue", fg="white").pack(expand=True)
        tk.Label(self.frame_answer_select, text="정답 선택 창", bg="blue", fg="white").pack(expand=True)

# GUI 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()