# game_ds.py

import pygame
import numpy as np
from algorithm_ds import MapHandler, PosteriorCalculator

class PS4Controller(): 
    """
    PS4 컨트롤러 입력을 처리하는 클래스
    """
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()

    def get_movement(self):
        """
        Reads D-pad input for rover movement.
        """
        pygame.event.pump()
        x_move, y_move = 0, 0
        if self.controller:
            if self.controller.get_button(11):  # Up (D-pad)
                y_move = -1
            elif self.controller.get_button(12):  # Down (D-pad)
                y_move = 1
            elif self.controller.get_button(13):  # Left (D-pad)
                x_move = -1
            elif self.controller.get_button(14):  # Right (D-pad)
                x_move = 1
        return x_move, y_move

    def get_label_selection(self, current_label, labels):
        """
        Adjusts label selection using L2 and R2.
        """
        if self.controller.get_button(6):  # L2
            current_label = (current_label - 1) % len(labels)
        elif self.controller.get_button(7):  # R2
            current_label = (current_label + 1) % len(labels)
        return current_label

    def submit_guess(self):
        """
        Checks if the 'X' button (1) is pressed for submitting the guess.
        """
        return self.controller.get_button(1)  # X button

class Game:
    """
    Main game class for managing the exploration and gameplay loop.
    """
    def __init__(self, map_file, sorted_labels, class_pixel_probs, class_priors):
        self.map_handler = MapHandler()
        self.map_array = self.map_handler.load_map(map_file)
        self.rover_position = [0, 0]  # Starting position of the rover
        self.visited_white = []
        self.visited_black = []
        self.sorted_labels = sorted_labels
        self.posterior_calculator = PosteriorCalculator(class_pixel_probs, class_priors)
        self.ps4_controller = PS4Controller()
        self.current_label_index = 0  # Tracks the currently selected label
        self.correct_class = np.random.choice(sorted_labels)  # Randomly select the correct class
        self.game_over = False
        self.screen = None  # Pygame screen
        self.cell_size = 20  # Size of each cell on the grid
        self.font = None

    def initialize_screen(self):
        """
        Initializes the Pygame screen.
        """
         
        map_height, map_width = self.map_array.shape
        GARO = (map_width * self.cell_size) 
        SERO = (map_height * self.cell_size + 100) 
        self.screen = pygame.display.set_mode((GARO, SERO))
        pygame.display.set_caption("AI Rover Game")
        self.font = pygame.font.Font(None, 12)

    def draw_map(self):
        """
        Draws the map, rover, and visited cells on the Pygame screen.
        """
        self.screen.fill((50, 50, 50))  # Dark background
        map_height, map_width = self.map_array.shape

        for r in range(map_height):
            for c in range(map_width):
                color = (127, 127, 127)  # Default gray
                if (r, c) in self.visited_white:
                    color = (255, 255, 255)  # White
                elif (r, c) in self.visited_black:
                    color = (0, 0, 0)  # Black
                pygame.draw.rect(
                    self.screen,
                    color,
                    (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                )

        # Draw the rover
        rover_x, rover_y = self.rover_position
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),  # Red color
            (rover_y * self.cell_size + self.cell_size // 2, rover_x * self.cell_size + self.cell_size // 2),
            self.cell_size // 2
        )

    def draw_labels(self):
        """
        Draws the label selection interface.
        """
        label_area_y = self.map_array.shape[0] * self.cell_size
        for i, label in enumerate(self.sorted_labels):
            color = (255, 255, 0) if i == self.current_label_index else (200, 200, 200)
            label_text = self.font.render(label, True, color)
            self.screen.blit(label_text, (50 + i * 100, label_area_y + 20))

    def run(self):
        """
        Runs the game loop.
        """
        self.initialize_screen()
        clock = pygame.time.Clock()

        print(f"The correct class is: {self.correct_class}")
        print("Navigate using the PS4 controller and try to guess the correct class.")

        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True

            x_move, y_move = self.ps4_controller.get_movement()
            self.move_rover(x_move, y_move)
            self.current_label_index = self.ps4_controller.get_label_selection(
                self.current_label_index, self.sorted_labels
            )

            if self.ps4_controller.submit_guess():
                if self.sorted_labels[self.current_label_index] == self.correct_class:
                    print("Correct! You win!")
                else:
                    print(f"Wrong! The correct answer was: {self.correct_class}")
                self.game_over = True

            # Draw game screen
            self.draw_map()
            self.draw_labels()
            pygame.display.flip()

            # Control frame rate
            clock.tick(30)

        pygame.quit()

    def move_rover(self, x_move, y_move):
        """
        Moves the rover on the map and updates visited cells.
        """
        new_x = max(0, min(self.rover_position[0] + y_move, self.map_array.shape[0] - 1))
        new_y = max(0, min(self.rover_position[1] + x_move, self.map_array.shape[1] - 1))
        self.rover_position = [new_x, new_y]

        # Mark the cell as visited
        if self.map_array[new_x, new_y] > 0.5:
            if (new_x, new_y) not in self.visited_white:
                self.visited_white.append((new_x, new_y))
        else:
            if (new_x, new_y) not in self.visited_black:
                self.visited_black.append((new_x, new_y))

from algorithm import *

# 게임 실행
if __name__ == "__main__":
    # 데이터 로드 및 모델 학습
    data_by_class = load_data_from_folders("./expanded", img_size=(64, 64), threshold=0.3)
    class_pixel_probs, class_counts, total_images = train_pixel_probabilities(data_by_class)
    class_priors = calculate_class_priors(class_counts, total_images)

    # 필요한 변수 설정
    sorted_labels = list(class_pixel_probs.keys())
    num_classes = len(sorted_labels)
    class_probs_list = [class_priors[label] for label in sorted_labels]

    # Load map
    map_file = "./test_map/label01.npy"
    binary_map = np.load(map_file)
    start_pos = (32, 32)  # Starting near the center of the map
    sorted_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    game = Game(map_file, sorted_labels, class_pixel_probs, class_priors)
    game.run()