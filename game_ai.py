import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import copy
from tetris_game import *
from configs import GAME_CONFIG, AI_CONFIG, PATHS
from helpers import ExperienceBuffer, board_to_tensor

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
        conv_output_size = 64 * rows * cols
        
        # Feature embedding for pieces (current, hold, and next 4)
        self.piece_embedding = nn.Embedding(8, 32)  # 7 piece types + 1 for "no piece"
        piece_features_size = 32 * 6  # current + hold + 4 next pieces
        
        # Fully connected layers for the output
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + piece_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for V-value
        )
    
    def forward(self, board_state, piece_info):
        """Process board state and piece information to predict state value"""
        board_features = self.conv_layers(board_state)
        piece_features = self.piece_embedding(piece_info)
        piece_features = piece_features.view(piece_features.size(0), -1)
        combined_features = torch.cat([board_features, piece_features], dim=1)
        return self.fc_layers(combined_features)

class TetrisAI:
    def __init__(self):
        # Initialize parameters from config
        self.buffer_size = AI_CONFIG['buffer_size']
        self.batch_size = AI_CONFIG['batch_size']
        self.gamma = AI_CONFIG['gamma']
        self.epsilon = AI_CONFIG['epsilon']
        self.epsilon_decay = AI_CONFIG['epsilon_decay']
        self.epsilon_min = AI_CONFIG['epsilon_min']
        self.learning_rate = AI_CONFIG['learning_rate']
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=self.buffer_size)
        
        # Neural network model
        self.model = TetrisNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.HuberLoss()
        
        # Training metrics
        self.training_loss = 0.0
    
    def get_state_representation(self, app):
        """Convert game state to tensors for neural network input"""
        # Board representation
        board_tensor = board_to_tensor(app.board, app.rows, app.cols, app.emptyColor, self.device)
        
        # Get current piece index
        current_piece_idx = next((i for i, piece in enumerate(app.tetrisPieces) 
                                if app.fallingPiece == piece), 7)
        
        # Get hold piece index
        hold_piece_idx = 7  # Default: no piece
        if hasattr(app, 'holdPiece') and app.holdPiece is not None:
            hold_piece_idx = next((i for i, piece in enumerate(app.tetrisPieces) 
                                  if app.holdPiece == piece), 7)
        
        # Get next pieces indices
        next_pieces_idx = (app.nextPiecesIndices[:4] if hasattr(app, 'nextPiecesIndices') 
                           else []) + [7] * 4  # Pad with "no piece"
        next_pieces_idx = next_pieces_idx[:4]  # Ensure only 4 pieces
        
        # Create piece info tensor
        piece_info = [current_piece_idx, hold_piece_idx] + next_pieces_idx
        piece_info_tensor = torch.tensor([piece_info], dtype=torch.long, device=self.device)
        
        return board_tensor, piece_info_tensor
    
    def get_possible_actions(self, app):
        """Generate all possible actions including placements and hold"""
        possible_actions = []
        
        # Store original state
        original_state = (
            copy.deepcopy(app.fallingPiece),
            app.fallingPieceRow,
            app.fallingPieceCol
        )
        
        # Get placements for current piece
        possible_actions.extend(self._get_piece_placements(app, app.fallingPiece, "current"))
        
        # Consider hold piece if available and not already used
        if hasattr(app, 'holdPiece') and not getattr(app, 'holdPieceUsed', False):
            if app.holdPiece is not None:
                # Try placing the hold piece
                possible_actions.extend(self._get_piece_placements(app, app.holdPiece, "hold"))
            elif hasattr(app, 'nextPiecesIndices') and app.nextPiecesIndices:
                # If no hold piece, next piece becomes current after hold
                next_piece = app.tetrisPieces[app.nextPiecesIndices[0]]
                possible_actions.extend(self._get_piece_placements(app, next_piece, "hold"))
        
        # Restore original state
        app.fallingPiece, app.fallingPieceRow, app.fallingPieceCol = original_state
        
        return possible_actions
    
    def _get_piece_placements(self, app, piece, piece_source):
        """Get all possible placements for a specific piece"""
        placements = []
        
        # Store original piece state
        original_state = (app.fallingPiece, app.fallingPieceRow, app.fallingPieceCol)
        
        # Set the piece we're evaluating
        app.fallingPiece = copy.deepcopy(piece)
        
        # Try all rotations (maximum 4 for any piece)
        for rotation in range(4):
            piece_width = len(app.fallingPiece[0])
            
            # Try all columns
            for col in range(-1, app.cols - piece_width + 2):
                app.fallingPieceRow = 0
                app.fallingPieceCol = col
                
                if not fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol):
                    continue
                
                # Find drop position
                drop_row = app.fallingPieceRow
                while fallingPieceIsLegal(app, drop_row + 1, app.fallingPieceCol):
                    drop_row += 1
                
                # Store this action
                placements.append({
                    'piece': copy.deepcopy(app.fallingPiece),
                    'row': drop_row,
                    'col': app.fallingPieceCol,
                    'rotation': rotation,
                    'piece_source': piece_source
                })
            
            # Rotate piece for next iteration
            self._rotate_piece(app)
        
        # Restore original state
        app.fallingPiece, app.fallingPieceRow, app.fallingPieceCol = original_state
        
        return placements
        
    def _rotate_piece(self, app):
        """Helper method to rotate a piece without checking legality"""
        oldPiece = app.fallingPiece
        oldNumRows = len(oldPiece)
        oldNumCols = len(oldPiece[0])
        newNumRows, newNumCols = oldNumCols, oldNumRows
        
        rotatedFallingPiece = [([None] * newNumCols) for row in range(newNumRows)]
        i = 0
        for col in range(oldNumCols - 1, -1, -1):
            j = 0
            for row in range(oldNumRows):
                rotatedFallingPiece[i][j] = oldPiece[row][col]
                j += 1
            i += 1
        
        app.fallingPiece = rotatedFallingPiece
    
    def choose_action(self, app, train=True):
        """Choose the best action based on predicted state values"""
        possible_actions = self.get_possible_actions(app)
        if not possible_actions:
            return None
                
        # Explore: choose random action
        if train and random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Exploit: choose best action based on value
        best_action = None
        best_value = float('-inf')
            
        self.model.eval()
        with torch.no_grad():
            for action in possible_actions:
                # Get value of resulting state
                resulting_state, reward, _ = self.get_resulting_state(app, action)
                board_tensor, piece_info_tensor = resulting_state
                
                predicted_value = self.model(board_tensor, piece_info_tensor).item()
                total_value = reward + self.gamma * predicted_value
                    
                if total_value > best_value:
                    best_value = total_value
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
        
        # Calculate raw score change
        score_change = app_copy.score - original_score
        
        # Calculate shaped reward
        reward = self.calculate_shaped_reward(app_copy, action, score_change)
        
        # Get the state representation
        state_representation = self.get_state_representation(app_copy)
        
        return state_representation, reward, app_copy.isGameOver

    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        self.experience_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train model using experience replay"""
        if len(self.experience_buffer) < self.batch_size:
            return 0.0
        
        # Sample random batch
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Prepare batch data
        board_states, piece_infos = [], []
        next_board_states, next_piece_infos = [], []
        rewards, dones = [], []
        
        for state, _, reward, next_state, done in batch:
            board_state, piece_info = state
            next_board_state, next_piece_info = next_state
            
            board_states.append(board_state)
            piece_infos.append(piece_info)
            rewards.append(reward)
            next_board_states.append(next_board_state)
            next_piece_infos.append(next_piece_info)
            dones.append(done)
        
        # Convert to tensors
        board_states = torch.cat(board_states, dim=0)
        piece_infos = torch.cat(piece_infos, dim=0)
        rewards = torch.tensor(rewards, device=self.device)
        next_board_states = torch.cat(next_board_states, dim=0)
        next_piece_infos = torch.cat(next_piece_infos, dim=0)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Compute targets (r + γV(s'))
        self.model.train()
        with torch.no_grad():
            next_values = self.model(next_board_states, next_piece_infos).squeeze()
            next_values[dones] = 0.0
            targets = rewards + self.gamma * next_values
        
        # Compute predictions V(s)
        predictions = self.model(board_states, piece_infos).squeeze()
        
        # Compute loss and update model
        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_loss = loss.item()
        
        return loss.item()
    
    def save_model(self, filename):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_loss': self.training_loss
        }, filename)
    
    def load_model(self, filename):
        """Load model from file"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.training_loss = checkpoint.get('training_loss', 0.0)
        self.model.eval()

def train_tetris_ai(episodes=AI_CONFIG['episodes'], save_interval=AI_CONFIG['save_interval']):
    """Train the Tetris AI"""
    from helpers import log_metrics, print_stats
    
    class GameEnvironment:
        def __init__(self):
            self.reset()
        
        def reset(self):
            # Create app with Tetris game state
            self.app = AppWrapper()
            return self.app
        
        def step(self, action):
            if action is None:
                return self.app, 0, True
            
            # Store current score for reward calculation
            pre_score = self.app.score
            
            # Handle hold piece action
            if action['piece_source'] == 'hold':
                self._handle_hold_piece()
            
            # Place the piece
            self.app.fallingPiece = action['piece']
            self.app.fallingPieceRow = action['row']
            self.app.fallingPieceCol = action['col']
            placeFallingPiece(self.app)
            
            # Generate new piece if needed
            if action['piece_source'] == 'current':
                newFallingPiece(self.app)
            
            # Calculate reward and check for game over
            reward = self.app.score - pre_score
            done = not fallingPieceIsLegal(self.app, self.app.fallingPieceRow, self.app.fallingPieceCol)
            self.app.isGameOver = done
            
            return self.app, reward, done
        
        def _handle_hold_piece(self):
            """Handle swapping with hold piece"""
            if self.app.holdPiece is None:
                self.app.holdPiece = self.app.fallingPiece
                self.app.holdPieceColor = self.app.fallingPieceColor
                newFallingPiece(self.app)
            else:
                temp = (self.app.holdPiece, self.app.holdPieceColor)
                self.app.holdPiece, self.app.holdPieceColor = self.app.fallingPiece, self.app.fallingPieceColor
                self.app.fallingPiece, self.app.fallingPieceColor = temp
            
            self.app.holdPieceUsed = True
    
    # Initialize AI and environment
    ai = TetrisAI()
    env = GameEnvironment()
    
    # Training metrics
    rewards = []
    avg_rewards = []
    losses = []
    
    # Training loop
    start_time = time.time()
    
    for episode in range(episodes):
        # Reset environment
        app = env.reset()
        done = False
        total_reward = 0
        episode_start_time = time.time()
        
        # Episode loop
        while not done:
            # Get state and choose action
            state = ai.get_state_representation(app)
            action = ai.choose_action(app)
            
            # Take action
            next_app, reward, done = env.step(action)
            
            # Get next state and add experience
            next_state = ai.get_state_representation(next_app)
            ai.add_experience(state, action, reward, next_state, done)
            
            # Train model
            ai.train()
            
            # Update state and reward
            app = next_app
            total_reward += reward
        
        # Record metrics
        rewards.append(total_reward)
        avg_reward = sum(rewards[-100:]) / min(len(rewards), 100)
        avg_rewards.append(avg_reward)
        losses.append(ai.training_loss)
        
        # Log progress
        time_elapsed = time.time() - start_time
        log_metrics(episode, total_reward, ai.training_loss, time_elapsed,
                   f"{PATHS['logs_dir']}training_log.csv")
        
        # Print stats periodically
        if (episode + 1) % 10 == 0:
            print_stats(episode, episodes, total_reward, avg_reward, 
                       ai.training_loss, time_elapsed, ai.epsilon)
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            save_path = f"{PATHS['models_dir']}tetris_model_episode_{episode+1}.pt"
            ai.save_model(save_path)
    
    # Save final model
    ai.save_model(f"{PATHS['models_dir']}tetris_model_final.pt")
    
    return ai

# AppWrapper for testing and training without GUI
class AppWrapper:
    def __init__(self):
        # Initialize game state
        self.rows, self.cols, self.cellSize, self.margin = gameDimensions()
        self.isGameOver = False
        self.score = 0
        self.emptyColor = "blue"
        self.board = [([self.emptyColor] * self.cols) for row in range(self.rows)]
        
        # Initialize tetris pieces
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
        
        # Initialize hold piece
        self.holdPiece = None
        self.holdPieceColor = None
        self.holdPieceUsed = False
        
        # Initialize next pieces queue
        self.nextPiecesIndices = []
        for _ in range(4):
            self.nextPiecesIndices.append(random.randint(0, len(self.tetrisPieces) - 1))
        
        # Initialize falling piece
        self.fallingPieceRow = 0
        self.fallingPieceCol = 0
        newFallingPiece(self)

if __name__ == "__main__":
    train_tetris_ai()