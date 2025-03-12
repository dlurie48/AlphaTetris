import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tetris_game import *
from configs import GAME_CONFIG, AI_CONFIG, REWARDS
from helpers import calculate_state_features, calculate_shaped_reward, ExperienceBuffer

class TetrisNet(nn.Module):
    def __init__(self, rows=20, cols=10, feature_size=63):
        super(TetrisNet, self).__init__()
        
        # CNN for board state
        self.batch_norm = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=6, stride=1, padding=2)
        
        # Calculate CNN output features
        cnn_features = 128 * 2 + 64 * 2  # 384 (from max and avg pooling)
        
        # Fully connected layers for combined features
        self.fc1 = nn.Linear(cnn_features + feature_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Extract components
        board = x['board']  # [batch, 1, rows, cols]
        features = x['features']  # [batch, feature_size]
        
        # Process board
        board = self.batch_norm(board)
        
        # CNN Path 1
        c1 = F.relu(self.conv1(board))
        c1_max = torch.max(c1.view(c1.size(0), c1.size(1), -1), dim=2)[0]
        c1_avg = torch.mean(c1.view(c1.size(0), c1.size(1), -1), dim=2)
        
        # CNN Path 2
        c2 = F.relu(self.conv2(board))
        c2_max = torch.max(c2.view(c2.size(0), c2.size(1), -1), dim=2)[0]
        c2_avg = torch.mean(c2.view(c2.size(0), c2.size(1), -1), dim=2)
        
        # Combine CNN features with additional features
        combined = torch.cat([c1_max, c1_avg, c2_max, c2_avg, features], dim=1)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class TetrisAI:
    def __init__(self):
        # Load parameters from AI_CONFIG
        self.buffer_size = AI_CONFIG['buffer_size']
        self.batch_size = AI_CONFIG['batch_size']
        self.gamma = AI_CONFIG['gamma']
        self.epsilon = AI_CONFIG['epsilon']
        self.epsilon_decay = AI_CONFIG['epsilon_decay']
        self.epsilon_min = AI_CONFIG['epsilon_min']
        self.learning_rate = AI_CONFIG['learning_rate']
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set the fixed state dimension to 263 as calculated
        # 200 (board state) + 
        # 10 (column heights) + 10 (holes) + 1 (hold allowed) + 
        # 42 (current piece + hold piece + 4 next pieces) = 263
        self.state_dim = 263
        
        # Create network - ensure TetrisNet is a separate class
        self._create_model()
        
        # Initialize experience buffer with the fixed dimension
        self.experience_buffer = ExperienceBuffer(
        self.buffer_size,
        self.device
    )
    
    def _create_model(self):
        """Create the neural network model"""
        rows = 20  # Default rows
        cols = 10  # Default columns
        feature_size = 63  # Default feature size
        
        self.model = TetrisNet(rows, cols, feature_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.HuberLoss()
    
    def get_state_representation(self, app):
        """Convert game state to a tensor with 2D board and features"""
        # Convert board to 2D numerical array (0.0 for empty, 1.0 for filled)
        board_state = [[1.0 if app.board[row][col] != app.emptyColor else 0.0 
                    for col in range(app.cols)] for row in range(app.rows)]
        
        # Create feature vector
        features = []
        
        # Column heights (10 features)
        heights = []
        for col in range(app.cols):
            for row in range(app.rows):
                if app.board[row][col] != app.emptyColor:
                    heights.append((app.rows - row) / app.rows)  # Normalize
                    break
            else:
                heights.append(0.0)  # Empty column
        features.extend(heights)
        
        # Holes (10 features)
        holes = []
        for col in range(app.cols):
            found_block = False
            col_holes = 0
            for row in range(app.rows):
                if app.board[row][col] != app.emptyColor:
                    found_block = True
                elif found_block:
                    col_holes += 1
            holes.append(col_holes / app.rows)  # Normalize
        features.extend(holes)
        
        # Hold piece availability (1 feature)
        features.append(1.0 if not app.holdPieceUsed else 0.0)
        
        # Current, hold, and next pieces (42 features)
        # Current piece (7 features)
        current_piece = [0] * 7
        if app.fallingPiece in app.tetrisPieces:
            current_piece[app.tetrisPieces.index(app.fallingPiece)] = 1
        features.extend(current_piece)
        
        # Hold piece (7 features)
        hold_encoding = [0] * 7
        if app.holdPiece is not None and app.holdPiece in app.tetrisPieces:
            hold_encoding[app.tetrisPieces.index(app.holdPiece)] = 1
        features.extend(hold_encoding)
        
        # Next pieces (4x7=28 features)
        for i in range(min(4, len(app.nextPiecesIndices))):
            piece_idx = app.nextPiecesIndices[i]
            piece_encoding = [0] * 7
            piece_encoding[piece_idx] = 1
            features.extend(piece_encoding)
        
        # Pad with zeros if needed
        for _ in range(4 - min(4, len(app.nextPiecesIndices))):
            features.extend([0] * 7)
        
        # Convert to tensors
        board_tensor = torch.tensor([board_state], device=self.device, dtype=torch.float32).unsqueeze(1)
        feature_tensor = torch.tensor([features], device=self.device, dtype=torch.float32)
        
        return {
            'board': board_tensor,  # Shape: [1, 1, rows, cols]
            'features': feature_tensor  # Shape: [1, 63]
        }

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
        """
        Simulate applying an action and return resulting state
        For s', the 4th next piece is unknown and represented as [0,0,0,0,0,0,0]
        """
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
        
        # Copy next pieces, but for s', we'll only use the first 3 next pieces
        # The 4th next piece will be represented by zeros in get_state_representation
        if len(app.nextPiecesIndices) > 0:
            # For s', we only know 3 of the next pieces
            app_copy.nextPiecesIndices = app.nextPiecesIndices[:min(3, len(app.nextPiecesIndices))]
        else:
            app_copy.nextPiecesIndices = []
        
        # Set up additional required properties for holdPiece function to work
        app_copy.fallingPiece = [row[:] for row in app.fallingPiece]
        app_copy.fallingPieceColor = app.fallingPieceColor
        app_copy.fallingPieceRow = app.fallingPieceRow
        app_copy.fallingPieceCol = app.fallingPieceCol
        
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
        
        # Convert rewards and dones to tensors if they aren't already
        rewards = torch.tensor(rewards, device=self.device) if not isinstance(rewards, torch.Tensor) else rewards
        dones = torch.tensor(dones, device=self.device) if not isinstance(dones, torch.Tensor) else dones
        
        # Compute targets (Bellman equation)
        with torch.no_grad():
            # For each next state, get its predicted value
            next_values = torch.zeros_like(rewards)
            
            # Process in smaller batches to avoid memory issues
            batch_size = AI_CONFIG['batch_size']
            for i in range(0, len(next_states), batch_size):
                end = min(i + batch_size, len(next_states))
                
                # Get the subset of next_states for this mini-batch
                batch_next_states = next_states[i:end]
                
                # Get dictionary of batch states by collecting boards and features
                batch_dict = {
                    'board': torch.cat([s['board'] for s in batch_next_states]),
                    'features': torch.cat([s['features'] for s in batch_next_states])
                }
                
                # Get values for non-terminal states
                batch_values = self.model(batch_dict).squeeze()
                
                # Assign values to appropriate positions in next_values
                for j, idx in enumerate(range(i, end)):
                    if not dones[idx]:
                        next_values[idx] = batch_values[j]
            
            # Compute target values using the Bellman equation
            targets = rewards + self.gamma * next_values
        
        # Compute current value predictions
        self.model.train()
        
        # Prepare batch of current states
        batch_states = {
            'board': torch.cat([s['board'] for s in states]),
            'features': torch.cat([s['features'] for s in states])
        }
        
        predictions = self.model(batch_states).squeeze()
        
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
            'epsilon': self.epsilon,
            'rows': self.model.rows if hasattr(self.model, 'rows') else 20,
            'cols': self.model.cols if hasattr(self.model, 'cols') else 10,
            'feature_size': self.model.feature_size if hasattr(self.model, 'feature_size') else 63
        }, filename)

    def load_model(self, filename):
        """Load the model from a file"""
        checkpoint = torch.load(filename)
        
        # If the model hasn't been created yet, create it with saved dimensions
        if not hasattr(self, 'model') or self.model is None:
            rows = checkpoint.get('rows', 20)
            cols = checkpoint.get('cols', 10)
            feature_size = checkpoint.get('feature_size', 63)
            self.model = TetrisNet(rows, cols, feature_size).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.model.eval()