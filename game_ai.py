import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tetris_game import *
from configs import GAME_CONFIG, AI_CONFIG, REWARDS
from helpers import calculate_state_features, calculate_shaped_reward, ExperienceBuffer

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
        conv_output_size = 64 * rows * cols
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + 32, 128),
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
        # Parameters from config
        self.buffer_size = AI_CONFIG['buffer_size']
        self.batch_size = AI_CONFIG['batch_size']
        self.gamma = AI_CONFIG['gamma']
        self.epsilon = AI_CONFIG['epsilon']
        self.epsilon_decay = AI_CONFIG['epsilon_decay']
        self.epsilon_min = AI_CONFIG['epsilon_min']
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the neural network model
        self.model = TetrisNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=AI_CONFIG['learning_rate'])
        self.loss_fn = nn.HuberLoss()
        
        # Experience buffer setup - consistent state shape format
        self.experience_buffer = ExperienceBuffer(
            self.buffer_size, 
            (1, 1, GAME_CONFIG['rows'], GAME_CONFIG['cols']),
            self.device
        )
    
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
            if app.fallingPiece and np.array_equal(piece, app.fallingPiece):
                piece_features[0, 0] = i + 1  # Add 1 to distinguish from 0 (no piece)
                break
        
        # Hold piece
        if app.holdPiece is not None:
            for i, piece in enumerate(app.tetrisPieces):
                if np.array_equal(piece, app.holdPiece):
                    piece_features[0, 1] = i + 1
                    break
        
        # Next pieces (up to 5)
        for i in range(min(5, len(app.nextPiecesIndices))):
            if i < len(app.nextPiecesIndices):
                piece_features[0, i+2] = app.nextPiecesIndices[i] + 1
        
        return (board_tensor, piece_features)
    
    def get_possible_actions(self, app):
        """Generate all valid final placements for the current piece"""
        possible_actions = []
        
        # Store original piece state
        original_piece = [row[:] for row in app.fallingPiece]
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
            app.fallingPiece = [row[:] for row in original_piece]
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
    
    def _rotate_piece(self, piece):
        """Helper to rotate a piece, copied from tetris_game.py _rotatePieceWithoutChecking"""
        oldNumRows = len(piece)
        oldNumCols = len(piece[0])
        newNumRows, newNumCols = oldNumCols, oldNumRows
        
        rotated = [([None] * newNumCols) for row in range(newNumRows)]
        i = 0
        for col in range(oldNumCols - 1, -1, -1):
            j = 0
            for row in range(oldNumRows):
                rotated[i][j] = piece[row][col]
                j += 1
            i += 1
        
        return rotated
    
    def get_resulting_state(self, app, action):
        """Simulate applying an action and return resulting state"""
        # Create a minimal app copy with only essential attributes to avoid deepcopy issues
        app_copy = type('AppCopy', (), {})()
        
        # Copy only essential attributes (avoiding tkinter objects)
        app_copy.rows = app.rows
        app_copy.cols = app.cols
        app_copy.emptyColor = app.emptyColor
        app_copy.board = [row[:] for row in app.board]  # Deep copy board
        app_copy.score = app.score
        app_copy.tetrisPieces = app.tetrisPieces  # Reference is fine, pieces don't change
        app_copy.tetrisPieceColors = app.tetrisPieceColors  # Reference is fine
        app_copy.isGameOver = False
        
        # Set piece state from action
        app_copy.fallingPiece = [row[:] for row in action['piece']]  # Deep copy the piece
        app_copy.fallingPieceRow = action['row']
        app_copy.fallingPieceCol = action['col']
        app_copy.fallingPieceColor = app.fallingPieceColor
        
        # Copy hold and next pieces safely
        if app.holdPiece is not None:
            app_copy.holdPiece = [row[:] for row in app.holdPiece]
        else:
            app_copy.holdPiece = None
        app_copy.holdPieceColor = app.holdPieceColor
        app_copy.nextPiecesIndices = app.nextPiecesIndices[:]
        
        # Simulate placing the piece
        original_score = app_copy.score
        placeFallingPiece(app_copy)
        score_change = app_copy.score - original_score
        
        # Calculate shaped reward
        reward = calculate_shaped_reward(app, app_copy, score_change)
        
        # Simulate getting a new piece to check game over
        newFallingPiece(app_copy)
        is_terminal = not fallingPieceIsLegal(app_copy, app_copy.fallingPieceRow, app_copy.fallingPieceCol)
        if is_terminal:
            app_copy.isGameOver = True
            reward += REWARDS['game_over']  # Add game over penalty
        
        # Get state representation of resulting state
        next_state = self.get_state_representation(app_copy)
        
        return next_state, reward, is_terminal
    
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
                # Get the resulting state from taking this action
                next_state, reward, is_terminal = self.get_resulting_state(app, action)
                board_tensor, piece_features = next_state
                
                # Calculate Q-value: immediate reward + discounted future value
                future_value = 0.0
                if not is_terminal:
                    future_value = self.model(board_tensor, piece_features).item()
                    
                q_value = reward + self.gamma * future_value
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return best_action
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        board_state, piece_features = state
        next_board_state, next_piece_features = next_state
        
        # Store the experience
        self.experience_buffer.add(board_state, action, reward, next_board_state, done)
    
    def train(self):
        """Sample batch from experience buffer and train model"""
        if len(self.experience_buffer) < self.batch_size:
            return None
        
        # Sample random batch
        board_states, actions, rewards, next_board_states, dones = self.experience_buffer.sample(self.batch_size)
        
        # Convert rewards and dones to tensors if they aren't already
        rewards = torch.tensor(rewards, device=self.device) if not isinstance(rewards, torch.Tensor) else rewards
        dones = torch.tensor(dones, device=self.device) if not isinstance(dones, torch.Tensor) else dones
        
        # Compute targets (Bellman equation)
        with torch.no_grad():
            # For each next state, get its predicted value
            next_values = torch.zeros_like(rewards)
            
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(next_board_states), batch_size):
                end = min(i + batch_size, len(next_board_states))
                batch_indices = list(range(i, end))
                
                # Get piece features for this batch
                next_piece_features_batch = torch.zeros((len(batch_indices), 7), device=self.device)
                for j, idx in enumerate(batch_indices):
                    action = actions[idx]
                    # Extract piece features from the action if available
                    if 'piece_features' in action:
                        next_piece_features_batch[j] = action['piece_features']
                
                # Get values for non-terminal states
                batch_values = self.model(next_board_states[batch_indices], next_piece_features_batch).squeeze()
                for j, idx in enumerate(batch_indices):
                    if not dones[idx]:
                        next_values[idx] = batch_values[j]
            
            # Compute target values using the Bellman equation
            targets = rewards + self.gamma * next_values
        
        # Compute current value predictions
        self.model.train()
        piece_features_batch = torch.zeros((len(board_states), 7), device=self.device)
        for i, action in enumerate(actions):
            if 'piece_features' in action:
                piece_features_batch[i] = action['piece_features']
                
        predictions = self.model(board_states, piece_features_batch).squeeze()
        
        # Compute loss and update
        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, filename):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load_model(self, filename):
        """Load the model from a file"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.model.eval()