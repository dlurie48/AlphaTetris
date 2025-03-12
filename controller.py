from cmu_112_graphics import *
from tetris_game import *
from game_ai import TetrisAI
import time
import random
import torch
import threading
import os
from configs import GAME_CONFIG, AI_CONFIG, PATHS

# Ensure directories exist
def ensure_paths():
    for path in PATHS.values():
        if path.endswith('/'):  # It's a directory
            os.makedirs(path, exist_ok=True)
        else:  # It's a file
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

class TetrisWithAI(App):
    def appStarted(self):
        # Initialize the base Tetris game
        self.initTetrisGame()
        
        # Game modes: "human_player", "ai_player_watching", "ai_player_training"
        self.gameMode = "human_player"
        
        # AI settings
        self.ai = TetrisAI()
        self.aiMoveDelay = 300  # ms between AI moves in watching mode
        self.aiEpisodes = 0
        self.aiTrainingStart = None
        self.showAIDebug = False
        self.trainingStepsPerFrame = 50  # Increased from 10 to 50
        
        # AI search parameters
        self.searchDepth = 3  # Default search depth
        self.ai.search_depth = self.searchDepth
        
        # Training metrics
        self.episodeRewards = []
        self.episodeLosses = []
        self.avgReward = 0
        self.avgLoss = 0
        
        # Background training thread
        self.trainingThread = None
        self.stopTraining = False
        
        # Status message
        self.statusMessage = "Human Mode. Press 'M' to cycle through modes, 'O' for options"
        self.statusTimer = 3000  # 3 seconds
        
        # Options menu
        self.showOptions = False
    
    def initTetrisGame(self):
        # Initialize the tetris game using functions from tetris_game.py
        appStarted(self)
        
        # Training state tracking
        self.currentEpisodeReward = 0
        self.currentEpisodeSteps = 0
        self.gameOver = False
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0
        
        # Initialize pane dimensions for hold and next pieces display
        self.holdPaneWidth = 6 * self.cellSize
        self.holdPaneHeight = 6 * self.cellSize
        self.nextPaneWidth = 6 * self.cellSize
        self.nextPaneHeight = 20 * self.cellSize
    
    def keyPressed(self, event):
        # Mode switching
        if event.key.lower() == "m":
            self.cycleGameMode()
            return
        
        # Reset game
        if event.key == "r":
            self.appStarted()
            return
            
        # Options
        if event.key == "o":
            self.showOptions = not self.showOptions
            return
            
        # Debug info
        if event.key == "d":
            self.showAIDebug = not self.showAIDebug
            return
            
        # Model operations
        if event.key == "l" and self.gameMode != "ai_player_training":
            ensure_paths()
            path = os.path.join(PATHS['models_dir'], "best_model.pt")
            if os.path.exists(path):
                checkpoint = torch.load(path)
                self.ai.model.load_state_dict(checkpoint['model_state_dict'])
                if 'target_model_state_dict' in checkpoint:
                    self.ai.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                else:
                    self.ai.target_model.load_state_dict(checkpoint['model_state_dict'])
                self.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.ai.epsilon = checkpoint.get('epsilon', 0.05)
                self.statusMessage = f"Model loaded from {path}"
                self.statusTimer = 3000
            else:
                self.statusMessage = "No model file found"
                self.statusTimer = 3000
            return
            
        if event.key == "p":
            ensure_paths()
            path = os.path.join(PATHS['models_dir'], "saved_model.pt")
            self.ai.save_model(path)
            self.statusMessage = f"Model saved to {path}"
            self.statusTimer = 3000
            return
        
        # Search depth adjust
        if event.key == "]" and self.gameMode == "ai_player_watching":
            self.searchDepth = min(5, self.searchDepth + 1)
            self.ai.search_depth = self.searchDepth
            self.statusMessage = f"Search depth: {self.searchDepth}"
            self.statusTimer = 3000
            return
            
        if event.key == "[" and self.gameMode == "ai_player_watching":
            self.searchDepth = max(1, self.searchDepth - 1)
            self.ai.search_depth = self.searchDepth
            self.statusMessage = f"Search depth: {self.searchDepth}"
            self.statusTimer = 3000
            return
        
        # Training rate adjust
        if event.key == "+" and self.gameMode == "ai_player_training":
            self.trainingStepsPerFrame = min(200, self.trainingStepsPerFrame + 10)
            self.statusMessage = f"Training speed: {self.trainingStepsPerFrame} steps/frame"
            self.statusTimer = 3000
            return
            
        if event.key == "-" and self.gameMode == "ai_player_training":
            self.trainingStepsPerFrame = max(10, self.trainingStepsPerFrame - 10)
            self.statusMessage = f"Training speed: {self.trainingStepsPerFrame} steps/frame"
            self.statusTimer = 3000
            return
        
        # Game controls for human player mode only
        if self.gameMode == "human_player" and not self.isGameOver:
            if event.key == "Up":
                rotateFallingPiece(self)
            elif event.key == "Down":
                moveFallingPiece(self, 1, 0)
            elif event.key == "Left":
                moveFallingPiece(self, 0, -1)
            elif event.key == "Right":
                moveFallingPiece(self, 0, 1)
            elif event.key == "Space":
                hardDrop(self)
                placeFallingPiece(self)
                newFallingPiece(self)
                if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
                    self.isGameOver = True
            elif event.key.lower() == "c":
                holdPiece(self)
    
    def cycleGameMode(self):
        """Cycle through game modes and stop any running training threads"""
        # Stop training thread if running
        if self.trainingThread and self.trainingThread.is_alive():
            self.stopTraining = True
            self.trainingThread.join(timeout=1.0)
            self.trainingThread = None
            self.stopTraining = False
        
        # Cycle modes
        if self.gameMode == "human_player":
            self.gameMode = "ai_player_watching"
            self.statusMessage = f"AI Mode (Watching). Search depth: {self.searchDepth}"
            self.ai.search_depth = self.searchDepth
        elif self.gameMode == "ai_player_watching":
            self.gameMode = "ai_player_training"
            self.statusMessage = "AI Mode (Training). AI trains without visualization."
        elif self.gameMode == "ai_player_training":
            self.gameMode = "human_player"
            self.statusMessage = "Human Mode. Use arrow keys to play."
        
        # Reset game state and start training if needed
        self.initTetrisGame()
        if self.gameMode in ["ai_player_training", "ai_player_watching"]:
            self.startAITraining()
        
        self.statusTimer = 5000
    
    def startAITraining(self):
        """Initialize AI training"""
        self.aiEpisodes = 0
        self.aiTrainingStart = time.time()
        self.episodeRewards = []
        self.episodeLosses = []
        self.avgReward = 0
        self.avgLoss = 0
        
        # Start background training if in training mode
        if self.gameMode == "ai_player_training":
            self.stopTraining = False
            self.trainingThread = threading.Thread(target=self.backgroundTraining)
            self.trainingThread.daemon = True
            self.trainingThread.start()
    
    def backgroundTraining(self):
        """Background thread for efficient AI training"""
        try:
            while not self.stopTraining:
                # Process multiple training steps in a batch for efficiency
                for _ in range(100):
                    if self.stopTraining:
                        break
                    
                    # Execute a single step
                    self.executeAIMove(batch_mode=True)
                    
                    # If game over, record metrics and reset
                    if self.isGameOver:
                        self.episodeRewards.append(self.currentEpisodeReward)
                        
                        # Calculate moving averages
                        if len(self.episodeRewards) > 100:
                            self.episodeRewards.pop(0)
                            if self.episodeLosses:
                                self.episodeLosses.pop(0)
                        
                        if self.episodeRewards:
                            self.avgReward = sum(self.episodeRewards) / len(self.episodeRewards)
                        
                        if self.episodeLosses:
                            self.avgLoss = sum(self.episodeLosses) / len(self.episodeLosses)
                        
                        # Start new episode
                        self.initTetrisGame()
                        self.aiEpisodes += 1
                        self.ai.training_episodes += 1
                
                # Sleep briefly to prevent CPU overuse
                time.sleep(0.001)
        except Exception as e:
            print(f"Training error: {e}")
    
    def timerFired(self):
        # Update status message timer
        if self.statusTimer > 0:
            self.statusTimer -= self.timerDelay
        
        # Handle game logic based on mode
        if self.gameMode == "human_player":
            self.handleHumanPlayerMode()
        elif self.gameMode == "ai_player_watching":
            self.executeAIMove(batch_mode=False)
        elif self.gameMode == "ai_player_training" and not self.trainingThread:
            # Only run this if background training thread isn't active
            for _ in range(self.trainingStepsPerFrame):
                self.executeAIMove(batch_mode=True)
                if self.isGameOver:
                    self.aiEpisodes += 1
                    self.ai.training_episodes += 1
                    self.initTetrisGame()
                    break
    
    def handleHumanPlayerMode(self):
        """Handle game logic for human player mode"""
        if not self.isGameOver:
            if not moveFallingPiece(self, 1, 0):
                placeFallingPiece(self)
                newFallingPiece(self)
                if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
                    self.isGameOver = True
    
    def executeAIMove(self, batch_mode=False):
        """Execute a single AI move and update training data"""
        # Handle game over state - reset the board for AI watching mode
        if self.isGameOver and not batch_mode:
            self.aiEpisodes += 1
            self.ai.training_episodes += 1
            self.episodeRewards.append(self.currentEpisodeReward)
            
            # Calculate moving averages
            if len(self.episodeRewards) > 100:
                self.episodeRewards.pop(0)
                if self.episodeLosses:
                    self.episodeLosses.pop(0)
            
            if self.episodeRewards:
                self.avgReward = sum(self.episodeRewards) / len(self.episodeRewards)
            
            if self.episodeLosses:
                self.avgLoss = sum(self.episodeLosses) / len(self.episodeLosses)
            
            # Reset game
            self.initTetrisGame()
            return
        
        # Get current state
        current_state = self.ai.get_state_representation(self)
        
        # For training, handle experience storage
        if self.lastState is not None:
            # Calculate reward (score difference)
            reward = self.score - self.lastScore
            reward = self.ai.calculate_reward(reward, self.isGameOver)
            self.currentEpisodeReward += reward
            
            # Check if game is over after this action
            terminal = self.isGameOver
            
            # Add experience with proper terminal state handling
            self.ai.add_experience(self.lastState, self.lastAction, 
                                reward, current_state, terminal)
            
            if reward != 0 and not batch_mode:
                print(f"Got reward: {reward}, total episode reward: {self.currentEpisodeReward}")
            
            # Train the model
            update_target = (self.ai.training_episodes % AI_CONFIG['target_update'] == 0)
            loss = self.ai.train(update_target=update_target)
            if loss is not None:
                self.episodeLosses.append(loss)
        
        # Use AI policy to choose action
        action = self.ai.choose_action(self)
        
        # Save current state for next iteration
        self.lastState = current_state
        self.lastScore = self.score
        self.lastAction = action
        self.currentEpisodeSteps += 1
        
        # Handle game over or invalid action
        if action is None or self.isGameOver:
            self.isGameOver = True
            return
        
        # Execute hold action if requested
        if action.get('is_hold', False):
            # Execute hold action using the existing holdPiece function
            holdPiece(self)
        
        # Apply action
        self.fallingPiece = action['piece']
        self.fallingPieceRow = action['row']
        self.fallingPieceCol = action['col']
        
        # Place the piece and get a new one
        placeFallingPiece(self)
        newFallingPiece(self)
        
        # Check if game is over
        if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
            self.isGameOver = True
    
    def drawHoldPane(self, canvas):
        """Draw the hold piece panel"""
        holdLeftMargin = self.margin
        holdTopMargin = self.margin
        
        canvas.create_rectangle(holdLeftMargin, holdTopMargin,
                                holdLeftMargin + self.holdPaneWidth,
                                holdTopMargin + self.holdPaneHeight,
                                fill="black", width=2, outline="white")
        
        canvas.create_text(holdLeftMargin + self.holdPaneWidth // 2,
                          holdTopMargin + 20,
                          text="HOLD", fill="white",
                          font="Arial 16 bold")
        
        if self.holdPiece is not None:
            pieceRows = len(self.holdPiece)
            pieceCols = len(self.holdPiece[0])
            centerRow = (self.holdPaneHeight - pieceRows * self.cellSize) // 2
            centerCol = (self.holdPaneWidth - pieceCols * self.cellSize) // 2
            
            for row in range(pieceRows):
                for col in range(pieceCols):
                    if self.holdPiece[row][col]:
                        drawCell(self, canvas, 
                                 row + centerRow // self.cellSize + 1,
                                 col + centerCol // self.cellSize,
                                 self.holdPieceColor,
                                 holdLeftMargin, holdTopMargin)

    def drawNextPiecesPane(self, canvas):
        """Draw the next pieces panel"""
        nextLeftMargin = self.boardLeftMargin + self.cols * self.cellSize + self.margin
        nextTopMargin = self.margin
        
        canvas.create_rectangle(nextLeftMargin, nextTopMargin,
                                nextLeftMargin + self.nextPaneWidth,
                                nextTopMargin + self.nextPaneHeight,
                                fill="black", width=2, outline="white")
        
        canvas.create_text(nextLeftMargin + self.nextPaneWidth // 2,
                          nextTopMargin + 20,
                          text="NEXT", fill="white",
                          font="Arial 16 bold")
        
        for i in range(min(4, len(self.nextPiecesIndices))):
            pieceIndex = self.nextPiecesIndices[i]
            piece = self.tetrisPieces[pieceIndex]
            pieceColor = self.tetrisPieceColors[pieceIndex]
            
            pieceRows = len(piece)
            pieceCols = len(piece[0])
            centerCol = (self.nextPaneWidth - pieceCols * self.cellSize) // 2
            
            startRow = 2 + i * 5
            
            for row in range(pieceRows):
                for col in range(pieceCols):
                    if piece[row][col]:
                        drawCell(self, canvas,
                                 startRow + row,
                                 centerCol // self.cellSize + col,
                                 pieceColor,
                                 nextLeftMargin, nextTopMargin)

    def drawModeIndicator(self, canvas):
        """Draw the current mode indicator"""
        modeColors = {
            "human_player": "green",
            "ai_player_watching": "blue",
            "ai_player_training": "red"
        }
        color = modeColors.get(self.gameMode, "purple")
        
        canvas.create_rectangle(
            self.width - 120, 10, 
            self.width - 10, 30, 
            fill=color, width=1
        )
        
        modeText = {
            "human_player": "HUMAN",
            "ai_player_watching": "AI WATCH",
            "ai_player_training": "AI TRAIN"
        }
        text = modeText.get(self.gameMode, self.gameMode.upper())
        
        canvas.create_text(
            self.width - 65, 20,
            text=text, fill="white", font="Arial 12 bold"
        )

    def drawTrainingInfo(self, canvas):
        """Draw training metrics if in AI mode"""
        if self.gameMode not in ["ai_player_watching", "ai_player_training"]:
            return
            
        trainingTime = time.time() - self.aiTrainingStart if self.aiTrainingStart else 0
        hours, remainder = divmod(trainingTime, 3600)
        minutes, seconds = divmod(remainder, 60)
        timeStr = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        infoX = self.width - 150
        infoY = self.height - 130
        
        canvas.create_rectangle(
            infoX - 10, infoY - 10,
            self.width - 10, self.height - 60,
            fill="black", outline="gray"
        )
        
        canvas.create_text(
            infoX, infoY,
            text=f"Episodes: {self.aiEpisodes}", 
            anchor="nw", fill="white", font="Arial 10"
        )
        
        canvas.create_text(
            infoX, infoY + 20,
            text=f"Time: {timeStr}", 
            anchor="nw", fill="white", font="Arial 10"
        )
        
        canvas.create_text(
            infoX, infoY + 40,
            text=f"Avg Reward: {self.avgReward:.1f}", 
            anchor="nw", fill="white", font="Arial 10"
        )
        
        canvas.create_text(
            infoX, infoY + 60,
            text=f"Avg Loss: {self.avgLoss:.6f}", 
            anchor="nw", fill="white", font="Arial 10"
        )
        
        # Add search depth for watching mode
        if self.gameMode == "ai_player_watching":
            canvas.create_text(
                infoX, infoY + 80,
                text=f"Search Depth: {self.searchDepth}", 
                anchor="nw", fill="white", font="Arial 10"
            )
    
    def drawStatusMessage(self, canvas):
        """Draw temporary status messages"""
        if self.statusTimer <= 0:
            return
        
        alpha = min(1.0, self.statusTimer / 1000)
        fillColor = f"#{int(0 * alpha):02x}{int(0 * alpha):02x}{int(0 * alpha):02x}"
        textColor = f"#{int(255 * alpha):02x}{int(255 * alpha):02x}{int(255 * alpha):02x}"
        
        msgY = self.height - 40
        canvas.create_rectangle(
            20, msgY - 15,
            self.width - 180, msgY + 15,
            fill=fillColor, outline=f"#{int(100 * alpha):02x}{int(100 * alpha):02x}{int(100 * alpha):02x}"
        )
        
        canvas.create_text(
            self.width // 2 - 80, msgY,
            text=self.statusMessage, fill=textColor, font="Arial 10"
        )

    def drawOptions(self, canvas):
        """Draw options menu overlay"""
        if not self.showOptions:
            return
        
        canvas.create_rectangle(
            0, 0, self.width, self.height,
            fill="white", stipple="gray50"
        )
        
        centerX = self.width // 2
        centerY = self.height // 2
        boxWidth = 300
        boxHeight = 250
        
        canvas.create_rectangle(
            centerX - boxWidth//2, centerY - boxHeight//2,
            centerX + boxWidth//2, centerY + boxHeight//2,
            fill="black", outline="white", width=2
        )
        
        canvas.create_text(
            centerX, centerY - boxHeight//2 + 20,
            text="OPTIONS", fill="white", font="Arial 14 bold"
        )
        
        options = [
            "M - Change Mode",
            "R - Reset Game",
            "L - Load AI Model",
            "P - Save AI Model",
            "O - Close Options",
            "+/- - Adjust Training Speed",
            "[/] - Adjust Search Depth",
            "D - Toggle Debug Info"
        ]
        
        for i, text in enumerate(options):
            canvas.create_text(
                centerX, centerY - boxHeight//4 + i*25,
                text=text, fill="white", font="Arial 12"
            )
            
        # Display current AI settings
        settings = [
            f"Current Search Depth: {self.searchDepth}",
            f"Current Epsilon: {self.ai.epsilon:.4f}",
            f"Training Speed: {self.trainingStepsPerFrame}/frame"
        ]
        
        for i, text in enumerate(settings):
            canvas.create_text(
                centerX, centerY + boxHeight//4 + i*20,
                text=text, fill="#AAAAFF", font="Arial 10"
            )
    
    def drawTrainingOnlyScreen(self, canvas):
        """Draw simplified UI during training"""
        canvas.create_rectangle(0, 0, self.width, self.height, fill="black")
        
        trainingTime = time.time() - self.aiTrainingStart if self.aiTrainingStart else 0
        hours, remainder = divmod(trainingTime, 3600)
        minutes, seconds = divmod(remainder, 60)
        timeStr = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        centerX = self.width // 2
        centerY = self.height // 2
        
        # Training status
        canvas.create_text(
            centerX, centerY - 60,
            text="AI TRAINING IN PROGRESS",
            fill="yellow", font="Arial 16 bold"
        )
        
        # Stats
        canvas.create_text(
            centerX, centerY - 20,
            text=f"Episodes: {self.aiEpisodes}",
            fill="white", font="Arial 14"
        )
        
        canvas.create_text(
            centerX, centerY + 10,
            text=f"Average Reward: {self.avgReward:.1f}",
            fill="white", font="Arial 14"
        )
        
        canvas.create_text(
            centerX, centerY + 40,
            text=f"Training Time: {timeStr}",
            fill="white", font="Arial 14"
        )
        
        canvas.create_text(
            centerX, centerY + 70,
            text=f"Steps Per Frame: {self.trainingStepsPerFrame}",
            fill="white", font="Arial 14"
        )
        
        # Controls reminder
        canvas.create_text(
            centerX, centerY + 110,
            text="Press 'M' to change mode, '+/-' to adjust training speed",
            fill="white", font="Arial 12"
        )
        
        # Mode indicator
        self.drawModeIndicator(canvas)
    
    def redrawAll(self, canvas):
        # For fast training mode, show minimal UI
        if self.gameMode == "ai_player_training":
            self.drawTrainingOnlyScreen(canvas)
            return
        
        # For human and watching modes, draw the standard game
        canvas.create_rectangle(0, 0, self.width, self.height, fill="black")
        
        # Draw the standard game elements
        drawBoard(self, canvas)
        drawFallingPiece(self, canvas)
        
        # Draw hold and next pieces panes
        self.drawHoldPane(canvas)
        self.drawNextPiecesPane(canvas)
        
        # Draw mode indicator
        self.drawModeIndicator(canvas)
        
        # Draw score
        writeScore(self, canvas)
        
        # Add game over message if needed
        if self.isGameOver:
            writeGameOver(self, canvas)
        
        # Draw AI training info
        self.drawTrainingInfo(canvas)
        
        # Draw status message if active
        self.drawStatusMessage(canvas)
        
        # Draw options menu if active
        self.drawOptions(canvas)
        
        # Draw key instructions
        instructionY = self.height - self.margin // 2
        canvas.create_text(self.width // 2, instructionY,
                          text="Controls: ←→↓: Move, ↑: Rotate, Space: Drop, C: Hold, O: Options",
                          fill="white", font="Arial 12")
        
def runTetrisWithAI():
    """Run the Tetris game with AI capabilities"""
    rows, cols, cellSize, margin = gameDimensions()
    boardWidth = cols * cellSize
    holdPaneWidth = 6 * cellSize
    nextPaneWidth = 6 * cellSize
    
    windowWidth = holdPaneWidth + boardWidth + nextPaneWidth + 4 * margin
    windowHeight = rows * cellSize + 4 * margin
    
    ensure_paths()  # Make sure all necessary directories exist
    game = TetrisWithAI(width=windowWidth, height=windowHeight)