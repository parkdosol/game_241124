import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
from pygame.locals import *

class MapHandler:
    def __init__(self, map_size=(64, 64)):
        self.map_array = np.full(map_size, 0.5)  # 컴퓨터 맵 (회색)
        self.player_map = np.full(map_size, 0.5)  # Player 맵
        self.rover_position_ai = [map_size[0] // 2, map_size[1] // 2]  # 컴퓨터 로버 위치
        self.rover_position_player = [map_size[0] // 2, map_size[1] // 2]  # Player 로버 위치
        self.visited_white_ai = set()  # 컴퓨터 맵 방문 기록
        self.visited_white_player = set()  # Player 맵 방문 기록

    def move_rover_player(self, direction):
        """
        Update Player rover position based on joystick input.
        """
        if direction == "up" and self.rover_position_player[0] > 0:
            self.rover_position_player[0] -= 1
        elif direction == "down" and self.rover_position_player[0] < self.player_map.shape[0] - 1:
            self.rover_position_player[0] += 1
        elif direction == "left" and self.rover_position_player[1] > 0:
            self.rover_position_player[1] -= 1
        elif direction == "right" and self.rover_position_player[1] < self.player_map.shape[1] - 1:
            self.rover_position_player[1] += 1

        # Player 방문 기록 업데이트
        if tuple(self.rover_position_player) not in self.visited_white_player:
            self.visited_white_player.add(tuple(self.rover_position_player))

    def display_maps(self, axs):
        """
        Update the 2x2 grid with all plots.
        """
        # (0, 0): 컴퓨터 맵
        axs[0, 0].clear()
        display_map_ai = self.map_array.copy()
        for r, c in self.visited_white_ai:
            display_map_ai[r, c] = 1.0
        axs[0, 0].imshow(display_map_ai, cmap="gray", vmin=0, vmax=1)
        axs[0, 0].scatter(self.rover_position_ai[1], self.rover_position_ai[0], c="red", label="Rover")
        axs[0, 0].set_title("AI Map")
        axs[0, 0].axis("off")

        # (1, 0): 확률 분포
        axs[1, 0].clear()
        axs[1, 0].bar(range(10), np.random.rand(10))  # Dummy probability data
        axs[1, 0].set_title("Probability Distribution")
        axs[1, 0].set_xlabel("Class")
        axs[1, 0].set_ylabel("Probability")

        # (0, 1): Player Map
        axs[0, 1].clear()
        display_map_player = self.player_map.copy()
        for r, c in self.visited_white_player:
            display_map_player[r, c] = 1.0
        axs[0, 1].imshow(display_map_player, cmap="gray", vmin=0, vmax=1)
        axs[0, 1].scatter(self.rover_position_player[1], self.rover_position_player[0], c="blue", label="Player Rover")
        axs[0, 1].set_title("Player Map")
        axs[0, 1].axis("off")

        # (1, 1): 빈 화면
        axs[1, 1].clear()
        axs[1, 1].text(0.5, 0.5, "Empty Screen", ha="center", va="center", fontsize=12)
        axs[1, 1].set_title("Info Panel")
        axs[1, 1].axis("off")


def main():
    # Initialize MapHandler
    handler = MapHandler(map_size=(64, 64))

    # Initialize pygame for joystick input
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick connected.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick {joystick.get_name()} connected.")

    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    def update(frame):
        """
        Animation update function.
        """
        handler.display_maps(axs)
        fig.canvas.draw_idle()

    def process_joystick_input():
        """
        Process PS4 controller inputs to move the player rover.
        """
        for event in pygame.event.get():
            if event.type == JOYBUTTONDOWN:
                if joystick.get_button(11):  # PS4 "up" button
                    handler.move_rover_player("up")
                elif joystick.get_button(12):  # PS4 "down" button
                    handler.move_rover_player("down")
                elif joystick.get_button(13):  # PS4 "left" button
                    handler.move_rover_player("left")
                elif joystick.get_button(14):  # PS4 "right" button
                    handler.move_rover_player("right")

    # Start animation
    def animation_func(frame):
        process_joystick_input()
        update(frame)

    ani = FuncAnimation(fig, animation_func, interval=100)
    plt.show()

    pygame.quit()

if __name__ == "__main__":
    main()
