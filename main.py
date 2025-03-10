#!/usr/bin/env python3

import argparse
import os
import threading
import time
from controller import TetrisWithAI
from game_ai import TetrisAI
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

def train_in_background(ai, episodes, save_interval, save_path):
    """Train the AI in a background thread without GUI"""
    print(f"Starting background training for {episodes} episodes...")
    
    # Create a simple environment for training
    from tetris_game import appStarted, placeFallingPiece, newFallingPiece, fallingPieceIsLegal
    
    # Simple environment class to simulate the game
    class Environment:
        def __init__(self):
            self.reset()
            
        def reset(self):
            # Initialize basic game state
            appStarted(self)
            self.episode_reward = 0
            return ai.get_state_representation(self)
        
        def step(self, action):
            if action is None:
                return self.reset(), 0, True
            
            # Apply action
            pre_score = self.score
            self.fallingPiece = action['piece']
            self.fallingPieceRow = action['row']
            self.fallingPieceCol = action['col']
            
            # Place piece and update game
            placeFallingPiece(self)
            reward = self.score - pre_score
            
            # Get new piece
            newFallingPiece(self)
            
            # Check if game over
            done = not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol)
            
            return ai.get_state_representation(self), reward, done
    
    env = Environment()
    state = env.reset()
    total_episodes = 0
    best_reward = 0
    start_time = time.time()
    
    # Training loop
    while total_episodes < episodes:
        # Select action
        action = ai.choose_action(env)
        
        # Execute action
        next_state, reward, done = env.step(action)
        
        # Store experience
        ai.add_experience(state, action, reward, next_state, done)
        
        # Update state
        state = next_state
        env.episode_reward += reward
        
        # Train model
        ai.train()
        
        # Check if episode ended
        if done:
            total_episodes += 1
            
            # Print progress
            elapsed = time.time() - start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            print(f"Episode {total_episodes}/{episodes} - Reward: {env.episode_reward:.1f} - Time: {time_str}")
            
            # Save model periodically
            if total_episodes % save_interval == 0 or env.episode_reward > best_reward:
                if env.episode_reward > best_reward:
                    best_reward = env.episode_reward
                    ai.save_model(f"{save_path}/best_model.pt")
                    print(f"Saved new best model with reward: {best_reward:.1f}")
                
                if total_episodes % save_interval == 0:
                    ai.save_model(f"{save_path}/model_episode_{total_episodes}.pt")
                    print(f"Saved checkpoint at episode {total_episodes}")
            
            # Reset environment
            state = env.reset()
    
    # Final save
    ai.save_model(f"{save_path}/final_model.pt")
    print(f"Training complete! Final model saved to {save_path}/final_model.pt")
    print(f"Best score achieved: {best_reward:.1f}")

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
        
        # Create AI instance
        ai = TetrisAI()
        
        # Load model if specified
        if args.load_model:
            try:
                ai.load_model(args.load_model)
                print(f"Loaded model from {args.load_model}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Set epsilon if specified
        if args.epsilon != AI_CONFIG['epsilon']:
            ai.epsilon = args.epsilon
            print(f"Set exploration rate (epsilon) to {args.epsilon}")
        
        # Train in background thread
        train_thread = threading.Thread(
            target=train_in_background,
            args=(ai, args.episodes, args.save_interval, PATHS['models_dir'])
        )
        train_thread.daemon = True
        train_thread.start()
        
        # Keep main thread alive
        try:
            while train_thread.is_alive():
                train_thread.join(1)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final model...")
            ai.save_model(f"{PATHS['models_dir']}/interrupted_model.pt")
            print(f"Model saved to {PATHS['models_dir']}/interrupted_model.pt")
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