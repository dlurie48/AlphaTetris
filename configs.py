# Configuration settings for Tetris AI

# AI parameters
AI_CONFIG = {
    'buffer_size': 50000,       # Experience replay buffer size
    'batch_size': 512,          # Batch size for training
    'gamma': 0.95,              # Discount factor
    'epsilon': 0.06,            # Exploration rate for ε-greedy policy
    'epsilon_decay': 0.999,     # Decay rate for epsilon
    'epsilon_min': 0.01,        # Minimum epsilon value
    'learning_rate': 0.001,     # Learning rate for optimizer
    'target_update': 10,        # Update target network every N episodes
    
    # Model hyperparameters
    'conv_filters': [32, 64],   # Number of filters in each convolutional layer
    'fc_units': [256, 128],     # Number of units in fully connected layers
    
    # Training settings
    'episodes': 1000,           # Number of episodes to train for
    'save_interval': 100,       # Save model every N episodes
    'max_memory': 100000,       # Maximum memory usage (in experiences)
    
    # Deep search parameters
    'search_depth': 3,          # Default tree search depth
    'search_best': 6,           # Number of best states to keep
    'search_random': 2,         # Number of random states to include
    
    # Reward shaping
    'game_over_penalty': -10.0, # Penalty for game over
    'reward_coef_tetris': 1.5,  # Reward coefficient for Tetris (4 lines)
    'reward_coef_triple': 1.3,  # Reward coefficient for Triple
    'reward_coef_double': 1.0,  # Reward coefficient for Double
    'reward_coef_single': 0.8,  # Reward coefficient for Single
}

# Game settings
GAME_CONFIG = {
    'ai_move_delay': 300,       # Milliseconds between AI moves
    'training_speed': 50,       # Speed multiplier during training
    'rows': 20,                 # Board rows
    'cols': 10,                 # Board columns
    'cell_size': 30,            # Pixel size of each cell
    'margin': 20,               # Margin around the board
}

# File paths
PATHS = {
    'models_dir': 'models/',
    'default_model': 'models/tetris_model.pt',
    'best_model': 'models/tetris_model_best.pt',
    'logs_dir': 'logs/',
}

# Features to extract from the game state
STATE_FEATURES = [
    'board_grid',               # The main board grid (0=empty, 1=filled)
    'column_heights',           # Height of each column
    'holes',                    # Number of holes (empty cells with filled cells above)
    'bumpiness',                # Sum of absolute differences between adjacent columns
    'completed_lines',          # Number of lines that would be completed
    'landing_height',           # Height where the piece would land
]

# Reward shaping
REWARDS = {
    'line_clear': 1.0,          # Base reward for clearing a line
    'multiple_lines': {         # Additional multipliers for multiple lines
        1: 1.0,                 # Single line: 1.0 × line_clear
        2: 2.5,                 # Double: 2.5 × line_clear
        3: 7.5,                 # Triple: 7.5 × line_clear
        4: 20.0,                # Tetris: 20.0 × line_clear
    },
    'game_over': -10.0,         # Penalty for game over
    'hole_created': -0.5,       # Penalty for creating a hole
    'height_penalty': -0.02,    # Penalty per height unit (discourages high stacks)
}