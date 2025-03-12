import multiprocessing
import os
import time
import torch
import numpy as np
import pickle
from tetris_game import Game
from configs import GAME_CONFIG, AI_CONFIG, REWARDS, PATHS

def ensure_paths():
    """Ensure all required directories exist"""
    for path in PATHS.values():
        if path.endswith('/'):  # It's a directory
            os.makedirs(path, exist_ok=True)
        else:  # It's a file
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

def save_model_checkpoint(model, optimizer, epsilon, filename):
    """Save model checkpoint to file"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
    }, filename)

def load_model_checkpoint(model, optimizer, filename):
    """Load model checkpoint from file"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint.get('epsilon', 0.05)
        return epsilon
    return None

def save_experience_buffer(buffer, filename):
    """Save experience buffer to file"""
    # Convert buffer to a format suitable for pickling
    serializable_buffer = {
        'states': buffer.states,
        'actions': buffer.actions,
        'rewards': buffer.rewards.cpu().numpy(),
        'next_states': buffer.next_states,
        'dones': buffer.dones.cpu().numpy(),
        'count': buffer.count,
        'full': buffer.full
    }
    with open(filename, 'wb') as f:
        pickle.dump(serializable_buffer, f)

def load_experience_buffer(buffer, filename):
    """Load experience buffer from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Restore buffer contents
        buffer.states = data['states']
        buffer.actions = data['actions']
        buffer.rewards = torch.tensor(data['rewards'], device=buffer.device)
        buffer.next_states = data['next_states']
        buffer.dones = torch.tensor(data['dones'], device=buffer.device)
        buffer.count = data['count']
        buffer.full = data['full']
        return True
    return False

def training_loop(ai, episodes=1000, save_interval=50, batch_size=128):
    """Main training loop with periodic saves and evaluation"""
    ensure_paths()
    
    # Track metrics
    episode_rewards = []
    episode_losses = []
    best_reward = 0
    start_time = time.time()
    
    # Create environment for training
    env = Game()
    
    # Load previous buffer if available
    experience_file = os.path.join(PATHS['models_dir'], 'experience_buffer.pkl')
    loaded = load_experience_buffer(ai.experience_buffer, experience_file)
    if loaded:
        print("Loaded experience buffer from file")
    
    for episode in range(episodes):
        # Reset environment
        env.reset()
        episode_reward = 0
        state = ai.get_state_representation(env)
        step_count = 0
        losses = []
        
        # Update target network periodically
        if episode % AI_CONFIG['target_update'] == 0:
            ai.update_target_model()
            print(f"Episode {episode}: Updated target network")
        
        # Episode loop
        while not env.isGameOver and step_count < 1000:
            # Choose action using current policy
            action = ai.choose_action(env)
            if action is None:
                break
                
            # Apply action to environment
            next_env, next_state, reward, done = ai.get_resulting_state(env, action)
            
            # Update environment
            env.board = next_env.board
            env.score = next_env.score
            env.holdPiece = next_env.holdPiece
            env.holdPieceColor = next_env.holdPieceColor
            env.holdPieceUsed = next_env.holdPieceUsed
            env.fallingPiece = next_env.fallingPiece
            env.fallingPieceColor = next_env.fallingPieceColor
            env.fallingPieceRow = next_env.fallingPieceRow
            env.fallingPieceCol = next_env.fallingPieceCol
            env.isGameOver = next_env.isGameOver
            env.nextPiecesIndices = next_env.nextPiecesIndices.copy()
            
            # Store experience in buffer
            ai.add_experience(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Train the model multiple times per step
            for _ in range(4):  # Adjust based on your needs
                loss = ai.train(update_target=(episode % AI_CONFIG['target_update'] == 0))
                if loss is not None:
                    losses.append(loss)
            
            if done:
                break
        
        # End of episode processing
        ai.training_episodes += 1
        episode_rewards.append(episode_reward)
        if losses:
            episode_losses.append(np.mean(losses))
        
        # Print progress
        if episode % 1 == 0:
            elapsed = time.time() - start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
            
            print(f"Episode {episode}/{episodes} - Reward: {episode_reward:.1f} - " +
                  f"Avg Reward: {avg_reward:.1f} - Loss: {avg_loss:.6f} - " +
                  f"Steps: {step_count} - Epsilon: {ai.epsilon:.4f} - Time: {time_str}")
        
        # Save model and buffer periodically
        if (episode % save_interval == 0 and episode > 0) or episode_reward > best_reward:
            model_path = os.path.join(PATHS['models_dir'], f"model_episode_{episode}.pt")
            save_model_checkpoint(ai.model, ai.optimizer, ai.epsilon, model_path)
            
            # Save buffer
            save_experience_buffer(ai.experience_buffer, experience_file)
            
            print(f"Saved checkpoint at episode {episode}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = os.path.join(PATHS['models_dir'], "best_model.pt")
                save_model_checkpoint(ai.model, ai.optimizer, ai.epsilon, best_model_path)
                print(f"Saved new best model with reward: {best_reward:.1f}")
    
    # Final save
    final_model_path = os.path.join(PATHS['models_dir'], "final_model.pt")
    save_model_checkpoint(ai.model, ai.optimizer, ai.epsilon, final_model_path)
    save_experience_buffer(ai.experience_buffer, experience_file)
    
    print(f"Training complete! Final model saved to {final_model_path}")
    print(f"Best score achieved: {best_reward:.1f}")
    
    return episode_rewards, episode_losses

def parallel_training_data_collection(ai, episodes_per_worker=10, workers=4):
    """Collect training data using multiple parallel workers"""
    ensure_paths()
    
    # Set up multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    
    # Create temporary model file for workers to load
    temp_model_path = os.path.join(PATHS['models_dir'], 'temp_model.pt')
    save_model_checkpoint(ai.model, ai.optimizer, ai.epsilon, temp_model_path)
    
    # Start worker processes
    processes = []
    for worker_id in range(workers):
        p = multiprocessing.Process(
            target=worker_collect_data,
            args=(worker_id, temp_model_path, episodes_per_worker, result_queue)
        )
        processes.append(p)
        p.start()
    
    # Collect results
    all_experiences = []
    total_reward = 0
    completed_workers = 0
    
    while completed_workers < workers:
        try:
            worker_id, experiences, avg_reward = result_queue.get(timeout=600)
            all_experiences.extend(experiences)
            total_reward += avg_reward
            completed_workers += 1
            print(f"Worker {worker_id} completed: {len(experiences)} experiences, avg reward: {avg_reward:.1f}")
        except Exception as e:
            print(f"Error collecting from worker: {e}")
            break
    
    # Clean up processes
    for p in processes:
        p.join(timeout=1.0)
        if p.is_alive():
            p.terminate()
    
    # Clean up temporary file
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    # Return results
    avg_reward = total_reward / workers if workers > 0 else 0
    return all_experiences, avg_reward

def worker_collect_data(worker_id, model_path, episodes, result_queue):
    """Worker process for collecting training data"""
    from game_ai import TetrisNet, TetrisAI
    
    # Create AI instance for this worker
    ai = TetrisAI()
    
    # Load model
    checkpoint = torch.load(model_path)
    ai.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set exploration rate based on worker id (worker 0 exploits more)
    ai.epsilon = checkpoint.get('epsilon', 0.05)
    if worker_id > 0:
        ai.epsilon = min(0.3, ai.epsilon * (1 + worker_id * 0.5))
    
    # Create environment
    env = Game()
    
    # Collect experiences
    experiences = []
    total_reward = 0
    
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        state = ai.get_state_representation(env)
        
        while not env.isGameOver:
            # Choose action
            action = ai.choose_action(env)
            if action is None:
                break
            
            # Apply action
            next_env, next_state, reward, done = ai.get_resulting_state(env, action)
            
            # Store experience
            experiences.append((state, action, reward, next_state, done))
            
            # Update environment
            env.board = next_env.board
            env.score = next_env.score
            env.holdPiece = next_env.holdPiece
            env.holdPieceColor = next_env.holdPieceColor
            env.holdPieceUsed = next_env.holdPieceUsed
            env.fallingPiece = next_env.fallingPiece
            env.fallingPieceColor = next_env.fallingPieceColor
            env.fallingPieceRow = next_env.fallingPieceRow
            env.fallingPieceCol = next_env.fallingPieceCol
            env.isGameOver = next_env.isGameOver
            env.nextPiecesIndices = next_env.nextPiecesIndices.copy()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
    
    # Send results back through queue
    avg_reward = total_reward / episodes if episodes > 0 else 0
    result_queue.put((worker_id, experiences, avg_reward))