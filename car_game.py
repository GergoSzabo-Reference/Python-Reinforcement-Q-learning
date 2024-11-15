import pygame
import random
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Car Game")

DEFAULT_CAR_POSITION = [WIDTH // 2, HEIGHT - int(HEIGHT*0.2)]
OBSTACLES = 10
SAVE_DATA = 0
LOAD_DATA = 0

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define car properties
CAR_WIDTH, CAR_HEIGHT = 40, 60
CAR_SPEED = 10
car_position = DEFAULT_CAR_POSITION[:] # floor divison = rounding down

# Set default images paths
car_image_path = "car.png"
goal_image_path = "goal.png"
obstacle_image_path = "obstacle.png"

# Function to load images safely
def load_image(path, default_color, size):
    if os.path.exists(path):  # Check if the image file exists
        return pygame.transform.scale(pygame.image.load(path), size)
    else:
        # Return a default rectangle if the image is not found
        surface = pygame.Surface(size)
        surface.fill(default_color)
        return surface

# Load images or default shapes
car_image = load_image(car_image_path, BLACK, (CAR_WIDTH, CAR_HEIGHT))
goal_image = load_image(goal_image_path, GREEN, (CAR_WIDTH, CAR_HEIGHT))
obstacle_image = load_image(obstacle_image_path, RED, (int(WIDTH*0.07), int(HEIGHT*0.07)))

# Define goal properties
goal_position = [WIDTH // 2, int(HEIGHT*0.1)]

# Define obstacles
obstacles = [[random.randint(0, WIDTH - 50), random.randint(100, HEIGHT - 200)] for _ in range(OBSTACLES)]

# Q-learning parameters
actions = ["LEFT", "RIGHT", "UP", "DOWN"]
q_table = np.zeros((WIDTH, HEIGHT, len(actions))) # as many args that many dimensions
# e.g. if q_table[50,50,0] is 0.9, it means that the q-value for the state (50, 50) and action "LEFT" is 0.9
learning_rate = 0.1 # takes only 10% from new infos
discount_factor = 0.9 # [0-1], how much we care about future rewards
exploration_rate = 1.0 # [0-1], how much we explore, at the beginning its 1.0=100%
exploration_decay = 0.995 # [0-1], how much we decrease exploration over time (/episode)

# Initialize episode counter
episode = 0

# Font for displaying episode count
font = pygame.font.Font(None, 36)

bad_try = 0
good_try = 0

# Function to draw objects on the screen
def draw_objects():
    global good_try, bad_try
    window.fill(WHITE)

    window.blit(goal_image, (goal_position[0], goal_position[1]))
    window.blit(car_image, (car_position[0], car_position[1]))
    for obs in obstacles:
        window.blit(obstacle_image, (obs[0], obs[1]))

    window.blit(font.render(f"Episode: {episode}", True, (0, 0, 0)), (10, 10))
    window.blit(font.render(f"Good try: {good_try}", True, (0, 255, 0)), (10, 50))
    window.blit(font.render(f"Bad try: {bad_try}", True, (255, 0, 0)), (10, 80))

    pygame.display.update()

# Function to move the car based on an action
def move_car(action):
    if action == "LEFT" and car_position[0] - CAR_SPEED >= 0:
        car_position[0] -= CAR_SPEED
    elif action == "RIGHT" and car_position[0] + CAR_SPEED + CAR_WIDTH <= WIDTH:
        car_position[0] += CAR_SPEED
    elif action == "UP" and car_position[1] - CAR_SPEED >= 0:
        car_position[1] -= CAR_SPEED
    elif action == "DOWN" and car_position[1] + CAR_SPEED + CAR_HEIGHT <= HEIGHT:
        car_position[1] += CAR_SPEED

OBSTACLE_COLLISION = 1
GOAL_REACHED = 2
NONE = 3
IS_IN_CORNER = 4

def is_in_corner(x, y):
    if x <= 0 and y <= 0:
        return True

    if x <= 0 and y >= HEIGHT:
        return True
    
    if x >= WIDTH and y <= 0:
        return True
    
    if x >= WIDTH and y >= HEIGHT:
        return True

    return False

# Function to check if the car collided with an obstacle
def check_collision():
    car_rect = pygame.Rect(car_position[0], car_position[1], CAR_WIDTH, CAR_HEIGHT)
    for obs in obstacles:
        obstacle_rect = pygame.Rect(obs[0], obs[1], 50, 50)
        if car_rect.colliderect(obstacle_rect):
            return OBSTACLE_COLLISION

    goal_rect = pygame.Rect(goal_position[0], goal_position[1], CAR_WIDTH, CAR_HEIGHT)

    if car_rect.colliderect(goal_rect):
        return GOAL_REACHED

    if is_in_corner(car_position[0], car_position[1]):
        return IS_IN_CORNER

    return NONE

# Function to get the action with the highest Q-value for a given state
def get_best_action(state):
    action_index = np.argmax(q_table[state[0], state[1]])

    return actions[action_index]


file_store_data = "q_learning_data.npz"

def save_data():
    np.savez(file_store_data,
    q_table=q_table, episode=episode, good_try = good_try, bad_try = bad_try, exploration_rate = exploration_rate, obstacles=obstacles)
    print(f"Data saved to {file_store_data}")

def load_data():
    global q_table, episode, good_try, bad_try, exploration_rate, obstacles
    if os.path.exists(file_store_data):
        data = np.load(file_store_data)
        q_table = data["q_table"]
        episode = data["episode"]
        good_try = data["good_try"]
        bad_try = data["bad_try"]
        exploration_rate = data["exploration_rate"]
        obstacles = data["obstacles"]
        print(f"Data has loaded succesfully. Episodes: {episode}, Good try: {good_try}, Bad try: {bad_try}")
    else:
        print(f"No data found at {file_store_data}")

# Main game loop
def game_loop():
    global exploration_rate, episode, good_try, bad_try
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get the current state
        state = (car_position[0], car_position[1])

        # Choose action (exploration vs exploitation)
        if random.random() < exploration_rate:
            action = random.choice(actions)
        else:
            action = get_best_action(state)

        # Perform the action
        move_car(action)
        draw_objects()

        # Calculate the reward
        collision_result = check_collision()

        if collision_result == OBSTACLE_COLLISION:
            reward = -100
            car_position[0], car_position[1] = DEFAULT_CAR_POSITION
            episode += 1
            bad_try += 1
            print(f"================Episode: {episode}=====================")
            print(q_table)
        elif collision_result == GOAL_REACHED:
            reward = 100
            car_position[0], car_position[1] = DEFAULT_CAR_POSITION
            episode += 1
            good_try += 1
            print(f"================Episode: {episode}=====================")
            print(q_table)
        elif collision_result == IS_IN_CORNER:
            reward = -10
        else:
            reward = -1

        # Update Q-value
        next_state = (car_position[0], car_position[1])
        best_future_q = np.max(q_table[next_state[0], next_state[1]])

        current_action_index = actions.index(action)

        current_q = q_table[state[0], state[1], current_action_index]
        q_table[state[0], state[1], current_action_index] = current_q + learning_rate * (reward + discount_factor * best_future_q - current_q)
        # Q(s,a)=Q(s,a)+α(r+γ⋅maxQ(s,a)−Q(s,a))

        #print(f"State: {state}, Next State: {next_state}, Best Future Q-value: {best_future_q}")

        # Decay exploration rate
        if exploration_rate > 0.01:
            exploration_rate *= exploration_decay

        # Frame rate limit
        pygame.time.delay(1)

    if SAVE_DATA:
        save_data()

    pygame.quit()

if LOAD_DATA:
    load_data()

game_loop()