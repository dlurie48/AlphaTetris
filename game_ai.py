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
        # Neural network that only uses the calculated state features
        self.layers = nn.Sequential(
            # Input layer for state features
            nn.Linear(22, 64),  # 6 scalar features + 10 column heights from calculate_state_features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for V-value
        )

    def forward(self, state_features):
        return self.layers(state_features)

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
        
        # Experience buffer setup with shape for feature vector
        self.experience_buffer = ExperienceBuffer(
            self.buffer_size, 
            (22,),  # 6 scalar features + 10 column heights
            self.device
        )
    
    def get_state_representation(self, app):
        # Get calculated state features
        features_dict = calculate_state_features(app.board, app.rows, app.cols, app.emptyColor)
        
        # Extract column heights
        column_heights = features_dict['column_heights']
        
        # Encode piece information
        current_piece_index = app.tetrisPieces.index(app.fallingPiece) if app.fallingPiece in app.tetrisPieces else -1
        hold_piece_index = app.tetrisPieces.index(app.holdPiece) if app.holdPiece is not None and app.holdPiece in app.tetrisPieces else -1
        
        # Get next piece indices, padding with -1 if fewer than 4 pieces
        next_pieces = app.nextPiecesIndices[:4] + [-1] * (4 - len(app.nextPiecesIndices))
        
        # Combine features
        feature_values = [
            features_dict['holes'],
            features_dict['bumpiness'],
            features_dict['aggregate_height'],
            features_dict['max_height'],
            features_dict['complete_lines'],
            features_dict.get('aggregate_height', 0) / app.rows,  # Normalized height
        ]
        
        # Add column heights and next pieces
        feature_values.extend(column_heights)
        feature_values.extend([current_piece_index, hold_piece_index])
        feature_values.extend(next_pieces)
        
        # Convert to tensor
        state_features = torch.tensor(feature_values, dtype=torch.float32, device=self.device).view(1, -1)
        
        return state_features
        
    def get_possible_actions(self, app):
        """Generate all valid final placements for the current piece and hold options"""
        possible_actions = []
        
        # Store original piece state
        original_piece = [row[:] for row in app.fallingPiece]
        original_row = app.fallingPieceRow
        original_col = app.fallingPieceCol
        
        # First, handle current piece actions
        self._add_piece_actions(app, app.fallingPiece, possible_actions, is_hold=False)
        
        # Then, handle hold piece actions if available and not already used this turn
        if not app.holdPieceUsed and app.holdPiece is not None:
            self._add_piece_actions(app, app.holdPiece, possible_actions, is_hold=True)
        # Or if we don't have a hold piece yet, simulate using the next piece
        elif not app.holdPieceUsed and app.holdPiece is None and app.nextPiecesIndices:
            next_piece_idx = app.nextPiecesIndices[0]
            next_piece = app.tetrisPieces[next_piece_idx]
            self._add_piece_actions(app, next_piece, possible_actions, is_hold=True)
        
        # Restore original piece state
        app.fallingPiece = original_piece
        app.fallingPieceRow = original_row
        app.fallingPieceCol = original_col
        
        return possible_actions
    
    def _add_piece_actions(self, app, piece, possible_actions, is_hold=False):
        """Helper method to add all possible actions for a specific piece"""
        # Store original piece state
        original_piece = app.fallingPiece
        original_row = app.fallingPieceRow
        original_col = app.fallingPieceCol
        
        # Temporarily set the app's falling piece to the piece we're evaluating
        app.fallingPiece = [row[:] for row in piece]
        
        # Get unique rotations for the given piece type
        max_rotations = 4
        for piece_idx, tetris_piece in enumerate(app.tetrisPieces):
            if self._pieces_equal(piece, tetris_piece):
                if piece_idx == 3:  # O piece
                    max_rotations = 1
                elif piece_idx == 0:  # I piece
                    max_rotations = 2
                break
        
        # Try all rotations
        for rotation in range(max_rotations):
            # Reset to original piece
            app.fallingPiece = [row[:] for row in piece]
            app.fallingPieceRow = 0
            app.fallingPieceCol = app.cols // 2
            
            # Apply rotations using the existing function
            for _ in range(rotation):
                rotatePieceWithoutChecking(app)
            
            # Get piece width after rotation
            piece_width = len(app.fallingPiece[0])
            
            # Try all valid columns
            for col in range(-1, app.cols - piece_width + 2):
                app.fallingPieceRow = 0
                app.fallingPieceCol = col
                
                # Skip if position is invalid
                if not fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol):
                    continue
                
                # Find final row by dropping the piece
                row = 0
                while fallingPieceIsLegal(app, row + 1, app.fallingPieceCol):
                    row += 1
                
                # Record action
                action = {
                    'piece': [row[:] for row in app.fallingPiece],
                    'rotation': rotation,
                    'col': app.fallingPieceCol,
                    'row': row,
                    'is_hold': is_hold
                }
                
                # Check if this is unique compared to existing actions
                is_unique = True
                for existing_action in possible_actions:
                    if (existing_action['col'] == action['col'] and 
                        existing_action['row'] == action['row'] and
                        existing_action['rotation'] == action['rotation'] and
                        existing_action['is_hold'] == action['is_hold']):
                        is_unique = False
                        break
                
                if is_unique:
                    possible_actions.append(action)
                    
        # Restore original piece state
        app.fallingPiece = original_piece
        app.fallingPieceRow = original_row
        app.fallingPieceCol = original_col

    def _pieces_equal(self, piece1, piece2):
        """Helper method to check if two pieces are equal"""
        if len(piece1) != len(piece2):
            return False
        
        for i in range(len(piece1)):
            if len(piece1[i]) != len(piece2[i]):
                return False
            
            for j in range(len(piece1[i])):
                if piece1[i][j] != piece2[i][j]:
                    return False
        
        return True
    
    def get_resulting_state(self, app, action):
        """Simulate applying an action and return resulting state"""
        # Create a minimal app copy with only essential attributes
        app_copy = type('AppCopy', (), {})()
        
        # Copy only essential attributes
        app_copy.rows = app.rows
        app_copy.cols = app.cols
        app_copy.emptyColor = app.emptyColor
        app_copy.board = [row[:] for row in app.board]  # Deep copy board
        app_copy.score = app.score
        app_copy.tetrisPieces = app.tetrisPieces  # Reference is fine
        app_copy.tetrisPieceColors = app.tetrisPieceColors  # Reference is fine
        app_copy.isGameOver = False
        app_copy.holdPieceUsed = app.holdPieceUsed
        
        # Copy hold and next pieces safely
        if app.holdPiece is not None:
            app_copy.holdPiece = [row[:] for row in app.holdPiece]
        else:
            app_copy.holdPiece = None
        app_copy.holdPieceColor = app.holdPieceColor
        app_copy.nextPiecesIndices = app.nextPiecesIndices[:]
        
        # Set up additional required properties for holdPiece function to work
        app_copy.fallingPiece = [row[:] for row in app.fallingPiece]
        app_copy.fallingPieceColor = app.fallingPieceColor
        app_copy.fallingPieceRow = app.fallingPieceRow
        app_copy.fallingPieceCol = app.fallingPieceCol
        app_copy.holdPieceUsed = app.holdPieceUsed
        
        # Handle hold action
        if action.get('is_hold', False):
            # Use the existing holdPiece function for consistent behavior
            from tetris_game import holdPiece
            holdPiece(app_copy)
        
        # Set piece state from action
        app_copy.fallingPiece = [row[:] for row in action['piece']]  # Deep copy the piece
        app_copy.fallingPieceRow = action['row']
        app_copy.fallingPieceCol = action['col']
        
        # Simulate placing the piece
        original_score = app_copy.score
        placeFallingPiece(app_copy)
        score_change = app_copy.score - original_score
        reward = score_change
        # Calculate shaped reward

        #reward = calculate_shaped_reward(self, app, action, score_change)
        
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
                
                # Calculate Q-value: immediate reward + discounted future value
                future_value = 0.0
                if not is_terminal:
                    future_value = self.model(next_state).item()
                    
                q_value = reward + self.gamma * future_value
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return best_action
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        # Store the experience
        self.experience_buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        """Sample batch from experience buffer and train model"""
        if len(self.experience_buffer) < self.batch_size:
            return None
        
        # Sample random batch
        states, actions, rewards, next_states, dones = self.experience_buffer.sample(self.batch_size)
        
        # Convert to tensors if they aren't already
        rewards = torch.tensor(rewards, device=self.device) if not isinstance(rewards, torch.Tensor) else rewards
        dones = torch.tensor(dones, device=self.device) if not isinstance(dones, torch.Tensor) else dones
        
        # Compute targets (Bellman equation)
        with torch.no_grad():
            # For each next state, get its predicted value
            next_values = torch.zeros_like(rewards)
            
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(next_states), batch_size):
                end = min(i + batch_size, len(next_states))
                batch_indices = list(range(i, end))
                
                # Get values for non-terminal states
                batch_values = self.model(next_states[batch_indices]).squeeze()
                for j, idx in enumerate(batch_indices):
                    if not dones[idx]:
                        next_values[idx] = batch_values[j]
            
            # Compute target values using the Bellman equation
            targets = rewards + self.gamma * next_values
        
        # Compute current value predictions
        self.model.train()
        predictions = self.model(states).squeeze()
        
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