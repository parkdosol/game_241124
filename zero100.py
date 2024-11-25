import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from PIL import Image
import pygame
from matplotlib.lines import Line2D
import time


def load_data_from_folders(base_dir, img_size=(64, 64), threshold=0.3):
    """
    지정된 폴더 구조에서 이미지를 로드하고 전처리합니다.
    :param base_dir: 데이터셋의 기본 디렉토리 경로
    :param img_size: 이미지 크기 (가로, 세로)
    :param threshold: 이진화 임계값
    :return: 각 클래스별 이미지 리스트 딕셔너리
    """
    label_names = ['0', '1', '3', '5', '6', '9']
    data_by_class = {label: [] for label in label_names}

    for label_name in label_names:
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue  # 폴더가 없으면 건너뜀

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(label_path, filename)
                try:
                    # 이미지 로드 및 전처리
                    img = Image.open(img_path).convert("L")
                    img = img.resize(img_size)
                    img_array = np.array(img) / 255.0  # [0, 1] 범위로 정규화
                    binary_img = (img_array > threshold).astype(np.float32)  # 이진화

                    data_by_class[label_name].append(binary_img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    return data_by_class 


def train_pixel_probabilities(data_by_class):
    """
    각 클래스별로 픽셀 확률을 계산합니다.
    :param data_by_class: 각 클래스별 이미지 리스트 딕셔너리
    :return: 각 클래스별 픽셀 확률 딕셔너리
    """
    class_pixel_probs = {}
    class_counts = {}
    total_images = 0

    for label, images in data_by_class.items():
        if len(images) == 0:
            continue  # 해당 클래스에 이미지가 없으면 건너뜀
        # 이미지 스택 생성
        image_stack = np.stack(images)  # (N, H, W)
        # 픽셀별로 평균 계산 (라플라스 스무딩 적용 가능)
        pixel_prob = (np.sum(image_stack, axis=0) + 1) / (len(images) + 2)  # 라플라스 스무딩
        class_pixel_probs[label] = pixel_prob
        class_counts[label] = len(images)
        total_images += len(images)

    return class_pixel_probs, class_counts, total_images


def calculate_class_priors(class_counts, total_images):
    """
    각 클래스의 사전 확률을 계산합니다.
    :param class_counts: 각 클래스별 이미지 수 딕셔너리
    :param total_images: 전체 이미지 수
    :return: 클래스별 사전 확률 딕셔너리
    """
    class_priors = {}
    for label, count in class_counts.items():
        class_priors[label] = count / total_images
    return class_priors


def display_map_with_rover_ai(map_array, rover_position, visited_white, visited_black, ax):
    """Display the map with the rover's position and visited locations."""
    display_map = np.full_like(map_array, 0.5, dtype=np.float32)  # Default: unexplored (gray) 

    # Update the map for visited tiles
    for r, c in visited_white:
        display_map[r, c] = 1.0  # White: visited and white tile
    for r, c in visited_black:
        display_map[r, c] = 0.0  # Black: visited and black tile

    # Display the map
    ax.clear()
    ax.imshow(display_map, cmap="gray", vmin=0, vmax=1)
    ax.scatter(rover_position[1], rover_position[0], c="red", label="Rover")  # Rover's position
    ax.set_title("AI Rover Exploration")
    ax.axis("off")


def display_map_with_rover_player(map_array, rover_position, visited_white, visited_black, ax):
    """Display the map with the rover's position and visited locations."""
    display_map = np.full_like(map_array, 0.5, dtype=np.float32)  # Default: unexplored (gray)

    # Update the map for visited tiles
    for r, c in visited_white:
        display_map[r, c] = 1.0  # White: visited and white tile
    for r, c in visited_black:
        display_map[r, c] = 0.0  # Black: visited and black tile

    # Display the map
    ax.clear()
    ax.imshow(display_map, cmap="gray", vmin=0, vmax=1)
    ax.scatter(rover_position[1], rover_position[0], c="red", label="Rover")  # Rover's position
    ax.set_title("Player Rover Exploration")
    ax.axis("off")


def calculate_posterior_probs(visited_white, map_array, class_pixel_probs, class_probs_list, sorted_labels, smoothing=1e-2):
    """
    Calculate posterior probabilities based on visited white cells and adjust with uniform distribution.
    """
    # Initialize observed map with zeros (all unvisited areas are black)
    observed = np.zeros_like(map_array, dtype=np.float32)

    # Mark visited white areas as 1
    for r, c in visited_white:
        observed[r, c] = 1

    # Number of classes and initialize posterior probabilities
    num_classes = len(sorted_labels)
    posterior_probs = np.zeros(num_classes)

    # Compute likelihood for each class
    for idx, label in enumerate(sorted_labels):
        likelihood = (
            observed * np.log(class_pixel_probs[label] + smoothing) +
            (1 - observed) * np.log(1 - class_pixel_probs[label] + smoothing)
        )
        posterior_probs[idx] = likelihood.sum()

    # Normalize probabilities
    posterior_probs = np.exp(posterior_probs - np.max(posterior_probs))  # Prevent overflow
    posterior_probs *= class_probs_list
    posterior_probs /= posterior_probs.sum()

    # Adjust with uniform distribution
    uniform_probs = np.ones(num_classes) / num_classes
    num_pixels_observed = len(visited_white)
    total_pixels = map_array.size

    # Amplify observation ratio to increase weight of observed data
    observation_ratio = (num_pixels_observed / total_pixels) ** 0.5  # Square root to boost early changes

    # Weighted combination of posterior and uniform distribution
    posterior_probs = observation_ratio * posterior_probs + (1 - observation_ratio) * uniform_probs
    posterior_probs /= posterior_probs.sum()  # Re-normalize after adjustment

    return posterior_probs


def get_next_move_to_target(rover_position, target_position):
    """Calculate the next move toward the target position (one step at a time)."""
    rover_row, rover_col = rover_position
    target_row, target_col = target_position

    if rover_row < target_row:  # Move down
        return 1, 0
    elif rover_row > target_row:  # Move up
        return -1, 0
    elif rover_col < target_col:  # Move right
        return 0, 1
    elif rover_col > target_col:  # Move left
        return 0, -1

    return 0, 0  # Already at the target


def find_nearest_white_cell(map_array, rover_position, visited):
    """Find the nearest unvisited white cell in the map."""
    white_cells = np.argwhere(map_array == 1)  # Find all white cells
    unvisited = [tuple(cell) for cell in white_cells if tuple(cell) not in visited]

    if not unvisited:
        return None

    # Find the nearest unvisited white cell
    distances = [np.abs(rover_position[0] - cell[0]) + np.abs(rover_position[1] - cell[1]) for cell in unvisited]
    nearest_cell = unvisited[np.argmin(distances)]

    return nearest_cell


class PS4Controller:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        # 조이스틱 초기화
        self.controller = None
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"Joystick '{self.controller.get_name()}' initialized.")
        else:
            print("No joystick detected!")

        # 이동 상태 저장
        self.x_move = 0
        self.y_move = 0

    def process_events(self):
        """
        Event queue에서 조이스틱 입력을 처리하고 상태를 업데이트합니다.
        """
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # 아날로그 스틱 움직임 처리
                x_axis = self.controller.get_axis(0)  # 좌/우
                y_axis = self.controller.get_axis(1)  # 위/아래
                
                # 축 민감도 조정 (예: 0.1 이상의 움직임만 인식)
                self.y_move = 0 if abs(x_axis) < 0.1 else int(round(x_axis))
                self.x_move = 0 if abs(y_axis) < 0.1 else int(round(-y_axis))  # Y축은 방향 반대
                self.x_move = -self.x_move  # Y축 방향 반전
                print(f"Axis motion: x_move={self.x_move}, y_move={self.y_move}")

            elif event.type == pygame.JOYHATMOTION:
                # 디지털 D패드 움직임 처리
                hat_input = self.controller.get_hat(0)  # (x, y)
                self.x_move, self.y_move = hat_input[1], -hat_input[0]
                print(f"D-pad motion: x_move={self.x_move}, y_move={self.y_move}")

            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"Button {event.button} pressed")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"Button {event.button} released")

    def get_movement(self):
        """
        현재 이동 상태를 반환합니다.
        """
        return self.x_move, self.y_move


class RoverSimulation:
    def __init__(self, map_array, start_position, class_pixel_probs, class_probs_list, sorted_labels):
        self.map_array = map_array 

        self.rover_position_ai = start_position
        self.visited_white_ai = set()  # Tiles visited and white
        self.visited_black_ai = set()  # Tiles visited and black
        self.target_ai = None

        self.rover_position_player = start_position
        self.visited_white_player = set()
        self.visited_black_player = set()
        self.target_player = None

        self.class_pixel_probs = class_pixel_probs
        self.class_probs_list = class_probs_list
        self.sorted_labels = sorted_labels

        self.controller = PS4Controller()  # 컨트롤러를 한 번만 초기화

        self.fig = None
        self.ax_map_ai = None
        self.ax_bar = None
        self.ax_map_player = None
        self.anim_ai = None
        self.anim_player = None

        self.start_time = time.time()  # 시뮬레이션 시작 시간
        self.elapsed_time = 0

    def update_time(self):
        """Update the elapsed time."""
        self.elapsed_time = time.time() - self.start_time

    def update_ai(self, frame):
        """Update function for AI rover."""
        self.update_time()

        if self.target_ai is None or self.target_ai == self.rover_position_ai:
            # Find the nearest unvisited white cell
            visited = self.visited_white_ai.union(self.visited_black_ai)
            self.target_ai = find_nearest_white_cell(self.map_array, self.rover_position_ai, visited)

            if self.target_ai is None:
                print("All white areas explored.")
                self.anim_ai.event_source.stop()
                return

        # Get the next move direction toward the target
        dr, dc = get_next_move_to_target(self.rover_position_ai, self.target_ai)
        self.rover_position_ai = (self.rover_position_ai[0] + dr, self.rover_position_ai[1] + dc)

        # Mark the tile as visited
        if self.map_array[self.rover_position_ai] == 1:
            self.visited_white_ai.add(self.rover_position_ai)
        else:
            self.visited_black_ai.add(self.rover_position_ai)

        # If the rover reaches the target, clear the target
        if self.rover_position_ai == self.target_ai:
            self.target_ai = None

        # Update the display
        self.update_display_ai()

    def update_player(self, frame):
        """Update function for player-controlled rover."""
        self.update_time()

        # 조이스틱 이벤트 처리
        self.controller.process_events()
        x_move, y_move = self.controller.get_movement()

        # 새 좌표 계산
        new_row = self.rover_position_player[0] + x_move
        new_col = self.rover_position_player[1] + y_move

        # 맵 경계를 벗어나지 않도록 제한
        if 0 <= new_row < self.map_array.shape[0] and 0 <= new_col < self.map_array.shape[1]:
            self.rover_position_player = (new_row, new_col)

            # 방문한 타일 기록
            if self.map_array[self.rover_position_player] == 1:
                self.visited_white_player.add(self.rover_position_player)
            else:
                self.visited_black_player.add(self.rover_position_player)

        # 플롯 업데이트
        display_map_with_rover_player(
            self.map_array, 
            self.rover_position_player, 
            self.visited_white_player, 
            self.visited_black_player, 
            self.ax_map_player
        )

        print(f"Player position: {self.rover_position_player}")

    def run(self):
        """Run the simulation."""
        self.fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 grid
        self.ax_map_ai, self.ax_bar, self.ax_map_player, self.ax_image = axs.flatten()  # Unpack axes

        # 플롯 사이 간격 추가
        self.fig.subplots_adjust(hspace=0.4, wspace=0.4)  # 높이와 너비 간격 설정

        # 오른쪽 아래 플롯에 원본 이미지를 표시
        image_path = "/home/ds/2024_boeing/src/241125/patterns.jpg"  # 업로드한 이미지 경로
        try:
            img = Image.open(image_path)  # 원본 이미지를 그대로 로드
            self.ax_image.imshow(img)  # 원본 이미지를 표시
            self.ax_image.set_title("Guess the pattern!")  # 제목 추가
            self.ax_image.axis("off")  # 축 숨기기
        except Exception as e:
            print(f"Error loading image: {e}")
            self.ax_image.text(0.5, 0.5, "Image Load Failed", ha="center", va="center", fontsize=12)

        # 굵은 가로줄 추가
        line = Line2D([0, 1], [0.5, 0.5], transform=self.fig.transFigure, color="gray", linewidth=4)
        self.fig.add_artist(line)

        # 애니메이션 시작
        self.anim_ai = FuncAnimation(self.fig, self.update_ai, interval=500, repeat=False)
        self.anim_player = FuncAnimation(self.fig, self.update_player, interval=500, repeat=False)

        plt.show()

    def update_display_ai(self):
        """Update the map and probability plots for AI."""
        self.ax_map_ai.clear()
        self.ax_bar.clear()

        # Display map
        display_map_with_rover_ai(self.map_array, self.rover_position_ai, self.visited_white_ai, self.visited_black_ai, self.ax_map_ai)

        # Calculate probabilities
        posterior_probs = calculate_posterior_probs(
            self.visited_white_ai, self.map_array, self.class_pixel_probs, self.class_probs_list, self.sorted_labels
        )

        # 전체 제목 설정
        self.fig.suptitle(
            f"Who's gonna win? : AI vs Human [{self.elapsed_time:.0f} sec]",  # 제목 텍스트에 경과 시간 추가
            fontsize=24,      # 큰 글씨 크기
            fontweight="bold",  # 굵게
            color="navy"       # 텍스트 색상 (예: 파란색)
        )
        # Display probabilities
        self.ax_bar.bar(range(len(self.sorted_labels)), posterior_probs)  # 확률 플롯 (주석 처리 on/off)
        self.ax_bar.set_title("AI Prediction")
        self.ax_bar.set_xlabel("Pattern")
        self.ax_bar.set_ylabel("Probability")
        self.ax_bar.set_xticks(range(len(self.sorted_labels)))
        self.ax_bar.set_xticklabels(self.sorted_labels)
        self.ax_bar.set_ylim(0, 1)


# 데이터 로드 및 모델 학습
data_by_class = load_data_from_folders("./expanded", img_size=(64, 64), threshold=0.3)
class_pixel_probs, class_counts, total_images = train_pixel_probabilities(data_by_class)
class_priors = calculate_class_priors(class_counts, total_images)

# 필요한 변수 설정
sorted_labels = list(class_pixel_probs.keys())
num_classes = len(sorted_labels)
class_probs_list = [class_priors[label] for label in sorted_labels]

# Load map
binary_map = np.load("./test_map/label06.npy")
start_pos = (32, 32)  # Starting near the center of the map

# Run simulation
simulation = RoverSimulation(binary_map, start_pos, class_pixel_probs, class_probs_list, sorted_labels)
simulation.run()
