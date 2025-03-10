import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tetris_game import *
from configs import GAME_CONFIG
from helpers import ExperienceBuffer

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
        
        rows = GAME_CONFIG['rows']
        cols = GAME_CONFIG['cols']
        
        # Feature processing layers
        self.feature_layers = nn.Sequential(
            nn.Linear(7, 32),  # 7 features for piece types (current + hold + next)
            nn.ReLU()
        )
        
        # Combined processing
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * rows * cols + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for V-value
        )
    
    def forward(self, board_state, piece_features):
        x1 = self.conv_layers(board_state)
        x2 = self.feature_layers(piece_features)
        x = torch.cat((x1, x2), dim=1)
        return self.fc_layers(x)

class TetrisAI:
    def __init__(self):
        # Parameters
        self.buffer_size = 50000
        self.batch_size = 512
        self.gamma = 0.99
        self.epsilon = 0.05
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the neural network model
        self.model = TetrisNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.HuberLoss()
        
        # More efficient experience buffer
        state_shape = (1, GAME_CONFIG['rows'], GAME_CONFIG['cols'])
        self.experience_buffer = ExperienceBuffer(self.buffer_size, state_shape, self.device)
    
    def get_state_representation(self, app):
        """Convert game state to tensor representation for neural network input"""
        # Create board tensor (1=filled, 0=empty)
        board_tensor = torch.zeros((1, 1, app.rows, app.cols), device=self.device)
        
        for row in range(app.rows):
            for col in range(app.cols):
                if app.board[row][col] != app.emptyColor:
                    board_tensor[0, 0, row, col] = 1.0
        
        # Create piece features tensor (current, hold, next pieces)
        piece_features = torch.zeros((1, 7), device=self.device)  # 7 possible piece types
        
        # Current piece
        for i, piece in enumerate(app.tetrisPieces):
            if piece == app.fallingPiece:
                piece_features[0, 0] = i + 1  # Add 1 to distinguish from 0 (no piece)
                break
        
        # Hold piece
        if app.holdPiece is not None:
            for i, piece in enumerate(app.tetrisPieces):
                if piece == app.holdPiece:
                    piece_features[0, 1] = i + 1
                    break
        
        # Next pieces (up to 5)
        for i in range(min(5, len(app.nextPiecesIndices))):
            if i < len(app.nextPiecesIndices):
                piece_features[0, i+2] = app.nextPiecesIndices[i] + 1
        
        return board_tensor, piece_features
    
    def get_possible_actions(self, app):
        """Generate all valid final placements for the current piece"""
        possible_actions = []
        
        # Store original piece state
        original_piece = app.fallingPiece.copy() if app.fallingPiece else None
        original_row = app.fallingPieceRow
        original_col = app.fallingPieceCol
        
        # Get unique rotations for the current piece type
        max_rotations = 4
        if app.fallingPiece == app.tetrisPieces[3]:  # O piece
            max_rotations = 1
        elif app.fallingPiece == app.tetrisPieces[0]:  # I piece
            max_rotations = 2
        
        # Try all rotations
        for rotation in range(max_rotations):
            # Reset to original piece
            app.fallingPiece = original_piece.copy()
            app.fallingPieceRow = 0
            app.fallingPieceCol = app.cols // 2
            
            # Apply rotations
            for _ in range(rotation):
                rotateFallingPiece(app)
            
            # Get piece width after rotation
            piece_width = len(app.fallingPiece[0])
            
            # Try all valid columns
            for col in range(-1, app.cols - piece_width + 2):
                app.fallingPieceRow = 0
                app.fallingPieceCol = col
                
                # Skip if position is invalid
                if not fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol):
                    continue
                
                # Hard drop to get final position
                temp_row = app.fallingPieceRow
                while fallingPieceIsLegal(app, temp_row + 1, app.fallingPieceCol):
                    temp_row += 1
                
                # Record action
                action = {
                    'piece': [row[:] for row in app.fallingPiece],  # Deep copy of piece
                    'rotation': rotation,
                    'col': col,
                    'row': temp_row
                }
                
                # Check if this is unique compared to existing actions
                is_unique = True
                for existing_action in possible_actions:
                    if (existing_action['col'] == action['col'] and 
                        existing_action['row'] == action['row'] and
                        existing_action['rotation'] == action['rotation']):
                        is_unique = False
                        break
                
                if is_unique:
                    possible_actions.append(action)
        
        # Restore original piece state
        app.fallingPiece = original_piece
        app.fallingPieceRow = original_row
        app.fallingPieceCol = original_col
        
        return possible_actions
    
    def get_resulting_state(self, app, action):
        """Simulate applying an action and return resulting state without modifying original state"""
        # Create shallow copy of the app with deep copy of critical attributes
        app_copy = type('AppCopy', (), {})()
        
        # Copy essential attributes
        app_copy.rows = app.rows
        app_copy.cols = app.cols
        app_copy.emptyColor = app.emptyColor
        app_copy.board = [row[:] for row in app.board]  # Deep copy of board
        app_copy.score = app.score
        app_copy.tetrisPieces = app.tetrisPieces
        app_copy.tetrisPieceColors = app.tetrisPieceColors
        
        # Copy piece state
        app_copy.fallingPiece = [row[:] for row in action['piece']]
        app_copy.fallingPieceRow = action['row']
        app_copy.fallingPieceCol = action['col']
        app_copy.fallingPieceColor = app.fallingPieceColor
        
        # Copy hold and next pieces for complete state representation
        app_copy.holdPiece = app.holdPiece.copy() if app.holdPiece else None
        app_copy.holdPieceColor = app.holdPieceColor
        app_copy.nextPiecesIndices = app.nextPiecesIndices[:]
        
        # Place the piece on the board
        placeFallingPiece(app_copy)
        
        # Calculate reward as score difference
        reward = app_copy.score - app.score
        
        # Create new piece to check if game is over
        newFallingPiece(app_copy)
        is_terminal = not fallingPieceIsLegal(app_copy, app_copy.fallingPieceRow, app_copy.fallingPieceCol)
        
        # Get state representation of resulting state
        state_representation = self.get_state_representation(app_copy)
        
        return state_representation, reward, is_terminal
    
    def choose_action(self, app):
        """Choose action using epsilon-greedy policy"""
        possible_actions = self.get_possible_actions(app)
        
        if not possible_actions:
            return None
        
        # Exploration: choose random action
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Exploitation: choose best action according to model
        best_action = None
        best_value = float('-inf')
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            for action in possible_actions:
                # Predict value of resulting state
                resulting_state = self.get_resulting_state(app, action)
                board_state, piece_features = resulting_state[0]
                predicted_value = self.model(board_state, piece_features).item()
                
                if predicted_value > best_value:
                    best_value = predicted_value
                    best_action = action
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return best_action
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.experience_buffer.add(state[0], action, reward, next_state[0], done)
    
    def train(self):
        """Sample batch from experience buffer and train model"""
        if len(self.experience_buffer) < self.batch_size:
            return 0
        
        # Sample random batch
        states, actions, rewards, next_states, dones = self.experience_buffer.sample(self.batch_size)
        
        # Get board states and piece features
        board_states = [state[0] for state in states]
        piece_features = [state[1] for state in states]
        next_board_states = [state[0] for state in next_states]
        next_piece_features = [state[1] for state in next_states]
        
        # Convert to tensors
        board_states = torch.cat(board_states)
        piece_features = torch.cat(piece_features)
        rewards = torch.tensor(rewards, device=self.device)
        next_board_states = torch.cat(next_board_states)
        next_piece_features = torch.cat(next_piece_features)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Compute targets
        with torch.no_grad():
            next_values = self.model(next_board_states, next_piece_features).squeeze()
            # Zero out values for terminal states
            next_values[dones] = 0.0
            targets = rewards + self.gamma * next_values
        
        # Compute predictions
        self.model.train()
        predictions = self.model(board_states, piece_features).squeeze()
        
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