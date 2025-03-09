#!/usr/bin/env python3

import argparse
import os
from controller import TetrisWithAI
from game_ai import train_tetris_ai
from configs import AI_CONFIG, PATHS

def ensure_directories():
    """Ensure all required directories exist"""
    for path in PATHS.values():
        if path.endswith('/'):  # It's a directory
            os.makedirs(path, exist_ok=True)
        else:  # It's a file
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tetris with AI')
    
    parser.add_argument('--train', action='store_true',
                        help='Train the AI model without GUI')
    
    parser.add_argument('--episodes', type=int, default=AI_CONFIG['episodes'],
                        help=f'Number of episodes to train for (default: {AI_CONFIG["episodes"]})')
    
    parser.add_argument('--save-interval', type=int, default=AI_CONFIG['save_interval'],
                        help=f'Save model every N episodes (default: {AI_CONFIG["save_interval"]})')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load an existing model')
    
    parser.add_argument('--epsilon', type=float, default=AI_CONFIG['epsilon'],
                        help=f'Exploration rate (default: {AI_CONFIG["epsilon"]})')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    if args.train:
        print(f"Training AI for {args.episodes} episodes...")
        print(f"Models will be saved to {PATHS['models_dir']} every {args.save_interval} episodes")
        
        # Train the AI
        train_tetris_ai(episodes=args.episodes, save_interval=args.save_interval)
        
        print("Training complete!")
    else:
        # Run the game with GUI
        game = TetrisWithAI(width=600, height=700)
        
        # If a model was specified, load it
        if args.load_model:
            try:
                game.ai.load_model(args.load_model)
                print(f"Loaded model from {args.load_model}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Set epsilon if specified
        if args.epsilon != AI_CONFIG['epsilon']:
            game.ai.epsilon = args.epsilon
            print(f"Set exploration rate (epsilon) to {args.epsilon}")

if __name__ == "__main__":
    main()