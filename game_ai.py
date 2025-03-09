import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
from tetris_game import *  # Import your existing Tetris game

class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        # CNN for processing the board state
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Fully connected layers for the output
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 20 * 10, 128),  # Assuming 20x10 board after convolutions
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for V-value
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

class TetrisAI:
    def __init__(self):
        # Parameters
        self.buffer_size = 50000  # Experience replay buffer size
        self.batch_size = 512
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.05  # Exploration rate
        self.experience_buffer = deque(maxlen=self.buffer_size)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the neural network model
        self.model = TetrisNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.HuberLoss()
    
    def get_state_representation(self, app):
        # Convert the current game state to a format suitable for the neural network
        # This should include board state, current piece, and next pieces
        
        # Create a tensor representation of the board
        board_tensor = torch.zeros((1, 1, 20, 10), device=self.device)
        
        # Fill in board state - 1 for filled cells, 0 for empty
        for row in range(app.rows):
            for col in range(app.cols):
                if app.board[row][col] != app.emptyColor:
                    board_tensor[0, 0, row, col] = 1.0
        
        # Note: In a more complete implementation, you would also encode:
        # - Current piece type and position
        # - Next pieces in queue
        # - Additional features like column heights and hole counts
                    
        return board_tensor
    
    def get_possible_actions(self, app):
        # Generate all possible valid placements of the current piece
        possible_actions = []
        
        # Store original piece state to restore later
        original_piece = copy.deepcopy(app.fallingPiece)
        original_row = app.fallingPieceRow
        original_col = app.fallingPieceCol
        
        # Try all rotations (maximum 4 for any piece)
        for _ in range(4):
            # Get piece width after this rotation
            piece_width = len(app.fallingPiece[0])
            
            # Try placing at each possible column
            for col in range(-1, app.cols - piece_width + 2):  # Extra margin for potential overhangs
                # Set piece to top row and current column
                app.fallingPieceRow = 0
                app.fallingPieceCol = col
                
                # Skip if this position is invalid
                if not fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol):
                    continue
                
                # Simulate dropping the piece
                while moveFallingPiece(app, 1, 0):
                    pass
                
                # Store this action
                action = {
                    'piece': copy.deepcopy(app.fallingPiece),
                    'row': app.fallingPieceRow,
                    'col': app.fallingPieceCol
                }
                possible_actions.append(action)
                
                # Reset position for next try
                app.fallingPieceRow = 0
                app.fallingPieceCol = col
            
            # Rotate piece for next iteration
            rotateFallingPiece(app)
        
        # Restore original piece state
        app.fallingPiece = original_piece
        app.fallingPieceRow = original_row
        app.fallingPieceCol = original_col
        
        return possible_actions
    
    def choose_action(self, app, state):
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            # Explore: choose random action
            possible_actions = self.get_possible_actions(app)
            return random.choice(possible_actions) if possible_actions else None
        else:
            # Exploit: choose best action according to model
            possible_actions = self.get_possible_actions(app)
            if not possible_actions:
                return None
                
            best_action = None
            best_value = float('-inf')
            
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                for action in possible_actions:
                    # Predict value of resulting state
                    resulting_state = self.get_resulting_state(app, action)
                    predicted_value = self.model(resulting_state).item()
                    
                    if predicted_value > best_value:
                        best_value = predicted_value
                        best_action = action
            
            return best_action
    
    def get_resulting_state(self, app, action):
        # Simulate applying the action to get the resulting state
        # Create a deep copy of the app
        app_copy = copy.deepcopy(app)
        
        # Apply the action
        app_copy.fallingPiece = action['piece']
        app_copy.fallingPieceRow = action['row']
        app_copy.fallingPieceCol = action['col']
        
        # Place the piece
        original_score = app_copy.score
        placeFallingPiece(app_copy)
        
        # Get the state representation
        state_representation = self.get_state_representation(app_copy)
        
        # Calculate reward (score difference)
        reward = app_copy.score - original_score
        
        return state_representation, reward, app_copy.isGameOver
    
    def add_experience(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.experience_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        # Sample batch from experience buffer and train model
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Set model to training mode
        self.model.train()
        
        # Prepare batch for training
        states, rewards, next_states, dones = [], [], [], []
        
        for state, _, reward, next_state, done in batch:
            states.append(state)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors
        states = torch.cat(states)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Compute targets
        with torch.no_grad():
            next_values = self.model(next_states).squeeze()
            # If done, next_value should be 0
            next_values[dones] = 0.0
            targets = rewards + self.gamma * next_values
        
        # Compute predictions
        predictions = self.model(states).squeeze()
        
        # Compute loss and update
        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

class GameEnvironment:
    def __init__(self):
        # Initialize with default Tetris settings
        self.app = AppWrapper()
    
    def reset(self):
        # Reset the game to initial state
        self.app = AppWrapper()
        return self.get_state()
    
    def get_state(self):
        # Get current state
        return self.app
    
    def step(self, action):
        # Apply action and return new state, reward, done
        if action is None:
            return self.get_state(), 0, True  # No valid action, game over
            
        # Copy current score for reward calculation
        pre_score = self.app.score
        
        # Apply the action
        self.app.fallingPiece = action['piece']
        self.app.fallingPieceRow = action['row']
        self.app.fallingPieceCol = action['col']
        
        # Place piece and update game
        placeFallingPiece(self.app)
        
        # Calculate reward as score difference
        reward = self.app.score - pre_score
        
        # Generate new piece
        newFallingPiece(self.app)
        
        # Check if game is over
        done = self.app.isGameOver
        
        return self.get_state(), reward, done

class AppWrapper:
    def __init__(self):
        # Initialize app similar to appStarted function
        self.rows, self.cols, self.cellSize, self.margin = gameDimensions()
        self.isGameOver = False
        self.score = 0
        self.emptyColor = "blue"
        self.board = [([self.emptyColor] * self.cols) for row in range(self.rows)]
        
        # Initialize pieces
        self.tetrisPieces = [
            [[True, True, True, True]],  # I piece
            [[True, False, False], [True, True, True]],  # J piece
            [[False, False, True], [True, True, True]],  # L piece
            [[True, True], [True, True]],  # O piece
            [[False, True, True], [True, True, False]],  # S piece
            [[False, True, False], [True, True, True]],  # T piece
            [[True, True, False], [False, True, True]]   # Z piece
        ]
        self.tetrisPieceColors = ["red", "yellow", "magenta", "pink", "cyan", "green", "orange"]
        
        # Initialize falling piece
        self.fallingPieceRow = 0
        self.fallingPieceCol = 0
        self.fallingPiece = None
        self.fallingPieceColor = None
        
        # Start with a new piece
        newFallingPiece(self)

def train_tetris_ai(episodes=1000):
    ai = TetrisAI()
    env = GameEnvironment()
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get state representation
            state_rep = ai.get_state_representation(state)
            
            # Choose action
            action = ai.choose_action(state, state_rep)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Get next state representation
            next_state_rep = ai.get_state_representation(next_state)
            
            # Add experience
            ai.add_experience(state_rep, action, reward, next_state_rep, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Train model
            loss = ai.train()
            
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
        
        # Save model periodically
        if (episode + 1) % 100 == 0:
            ai.save_model(f"tetris_model_episode_{episode+1}.pt")
    
    return ai

if __name__ == "__main__":
    trained_ai = train_tetris_ai()