import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from configs import GAME_CONFIG, STATE_FEATURES

def calculate_state_features(board, rows, cols, emptyColor):
    """
    Calculate various features from the board state that are helpful for the AI.
    
    Args:
        board: The game board 2D array
        rows: Number of rows in the board
        cols: Number of columns in the board
        emptyColor: The color value representing empty cells
    
    Returns:
        Dictionary of calculated features
    """
    features = {}
    
    # Column heights
    heights = [0] * cols
    for col in range(cols):
        for row in range(rows):
            if board[row][col] != emptyColor:
                heights[col] = rows - row
                break
    features['column_heights'] = heights
    
    # Holes (empty cells with filled cells above them)
    holes = 0
    for col in range(cols):
        found_block = False
        for row in range(rows):
            if board[row][col] != emptyColor:
                found_block = True
            elif found_block:
                holes += 1
    features['holes'] = holes
    
    # Bumpiness (sum of absolute differences between adjacent columns)
    bumpiness = 0
    for i in range(cols - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    features['bumpiness'] = bumpiness
    
    # Aggregate height
    aggregate_height = sum(heights)
    features['aggregate_height'] = aggregate_height
    
    # Max height
    max_height = max(heights) if heights else 0
    features['max_height'] = max_height
    
    # Complete lines
    complete_lines = 0
    for row in range(rows):
        if all(board[row][col] != emptyColor for col in range(cols)):
            complete_lines += 1
    features['complete_lines'] = complete_lines
    
    return features

def calculate_shaped_reward(self, app, action, score_change):
    reward = score_change
    
    # Penalties for bad board states
    features = calculate_state_features(app.board, app.rows, app.cols, app.emptyColor)
    
    # Discourage holes and uneven surfaces
    reward -= 0.5 * features['holes']
    reward -= 0.3 * features['bumpiness']
    
    # Encourage complete lines with escalating bonus
    line_bonus = {
        1: 1.0,   # Single line
        2: 3.0,   # Double line
        3: 7.0,   # Triple line
        4: 20.0   # Tetris
    }
    reward += line_bonus.get(features['complete_lines'], 0)
    
    # Height management
    max_height = features['max_height']
    height_penalty = max(0, (max_height - 10) * 0.5)
    reward -= height_penalty
    
    return reward

def plot_training_progress(episodes, scores, avg_scores, losses=None, filename=None):
    """
    Plot and save training progress metrics.
    
    Args:
        episodes: List of episode numbers
        scores: List of scores for each episode
        avg_scores: List of moving average scores
        losses: List of loss values (optional)
        filename: File path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot scores
    plt.subplot(2, 1, 1)
    plt.plot(episodes, scores, label='Score', alpha=0.6)
    plt.plot(episodes, avg_scores, label='Average Score', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot losses if available
    if losses:
        plt.subplot(2, 1, 2)
        plt.plot(episodes, losses, label='Loss', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file if filename provided
    if filename:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        plt.savefig(filename)
    
    plt.close()

def board_to_tensor(board, rows, cols, emptyColor, device='cpu'):
    """
    Convert a board 2D array to a PyTorch tensor suitable for neural network input.
    
    Args:
        board: The game board 2D array
        rows: Number of rows in the board
        cols: Number of columns in the board
        emptyColor: The color value representing empty cells
        device: PyTorch device to place the tensor on
    
    Returns:
        PyTorch tensor representing the board
    """
    tensor = torch.zeros((1, 1, rows, cols), device=device)
    
    for row in range(rows):
        for col in range(cols):
            if board[row][col] != emptyColor:
                tensor[0, 0, row, col] = 1.0
    
    return tensor

def log_metrics(episode, score, loss, time_elapsed, filename):
    """
    Log training metrics to a file.
    
    Args:
        episode: Current episode number
        score: Score achieved in the episode
        loss: Training loss value
        time_elapsed: Time elapsed since start of training
        filename: File to log to
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Create file with header if it doesn't exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("episode,score,loss,time_elapsed\n")
    
    # Append metrics
    with open(filename, 'a') as f:
        f.write(f"{episode},{score},{loss:.6f},{time_elapsed:.2f}\n")

def print_stats(episode, total_episodes, score, avg_score, loss, time_elapsed, epsilon=None):
    """
    Print training statistics in a nicely formatted way.
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        score: Score achieved in the episode
        avg_score: Moving average of scores
        loss: Training loss value
        time_elapsed: Time elapsed since start of training
        epsilon: Current exploration rate (optional)
    """
    hours, remainder = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    progress = f"[{episode}/{total_episodes}]"
    score_str = f"Score: {score} (Avg: {avg_score:.1f})"
    loss_str = f"Loss: {loss:.6f}" if loss is not None else ""
    eps_str = f"ε: {epsilon:.4f}" if epsilon is not None else ""
    
    print(f"Episode {progress} {score_str} {loss_str} {eps_str} Time: {time_str}")

class ExperienceBuffer:
    """Experience replay buffer for dictionary-based state representations"""
    
    def __init__(self, buffer_size, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.count = 0
        self.full = False
        
        # Use lists instead of tensors since we have complex dictionary objects
        self.states = [None] * buffer_size
        self.actions = [None] * buffer_size
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_states = [None] * buffer_size
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer"""
        idx = self.count % self.buffer_size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        
        self.count += 1
        if self.count >= self.buffer_size:
            self.full = True
            
    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        max_idx = self.buffer_size if self.full else self.count
        indices = np.random.choice(max_idx, batch_size, replace=False)
        
        return (
            [self.states[idx] for idx in indices],
            [self.actions[idx] for idx in indices],
            self.rewards[indices],
            [self.next_states[idx] for idx in indices],
            self.dones[indices]
        )
        
    def __len__(self):
        """Return the current size of the buffer"""
        return self.buffer_size if self.full else self.count