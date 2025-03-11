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
    """
    Calculate a shaped reward based on board state changes after applying an action.
    
    Args:
        app: The game state after the action is applied
        action: The action that was taken
        score_change: The raw score change from the action
    
    Returns:
        float: The shaped reward value
    """
    from helpers import calculate_state_features
    
    # We need to know the state before the action was taken
    # Since we already executed the action, we'll need to reconstruct the previous state
    # This is a simplification - ideally we would store the previous state features
    
    # Calculate features of the current state
    current_features = calculate_state_features(app.board, app.rows, app.cols, app.emptyColor)
    
    # Basic reward from score
    total_reward = score_change
    
    # 1. Penalty for holes - critical for good stacking
    holes = current_features['holes']
    total_reward -= 0.5 * holes
    
    # 2. Penalty for bumpiness - encourages flat surfaces
    bumpiness = current_features['bumpiness']
    total_reward -= 1 * bumpiness
    
    # 3. Reward for completed lines (already included in score_change, but we can emphasize it)
    complete_lines = current_features['complete_lines']
    if complete_lines > 0:
        # Bonus for multiple line clears
        if complete_lines == 4:  # Tetris
            total_reward += 4.0  # Extra bonus for Tetris
        elif complete_lines > 1:
            total_reward += 0.5 * complete_lines  # Small bonus for multi-line clear
    
    # 4. Penalty for height - discourages building too high
    max_height = current_features['max_height']
    # Progressive penalty that gets worse as the stack gets higher
    height_penalty = 0.2 * max_height if max_height < 10 else 0.5 * max_height
    total_reward -= height_penalty
    
    # 5. Check if hold piece was used in this action
    # This requires tracking in the controller class, but we can add a simple
    # incentive based on the action type if that information is available
    # For now we'll rely on the controller to add this bonus
    
    return total_reward

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
    """A more efficient experience replay buffer using numpy arrays"""
    
    def __init__(self, buffer_size, state_shape, device='cpu'):
        """
        Initialize the experience buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            state_shape: Shape of state tensors
            device: PyTorch device to place tensors on
        """
        self.buffer_size = buffer_size
        self.device = device
        self.count = 0
        self.full = False
        
        # Pre-allocate memory for the buffer
        self.states = torch.zeros((buffer_size, *state_shape), device=device)
        self.actions = [None] * buffer_size  # Can't pre-allocate complex objects
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_states = torch.zeros((buffer_size, *state_shape), device=device)
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
        
        # Because actions are complex objects, we handle them separately
        batch_actions = [self.actions[idx] for idx in indices]
        
        return (
            self.states[indices],
            batch_actions,
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
        
    def __len__(self):
        """Return the current size of the buffer"""
        return self.buffer_size if self.full else self.count