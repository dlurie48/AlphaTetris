from cmu_112_graphics import *
from tetris_game import *
from game_ai import TetrisAI
import copy
import time
import random

class TetrisWithAI(App):
    def appStarted(self):
        # Initialize the base Tetris game by using the functions from tetris_game.py
        self.initTetrisGame()
        
        # Game modes: "human_player", "ai_player_training", "ai_player_watching"
        self.gameMode = "human_player"
        
        # AI settings
        self.ai = TetrisAI()
        self.aiMoveDelay = 300  # ms between AI moves
        self.aiEpisodes = 0
        self.aiTrainingStart = None
        self.showAIDebug = False  # Whether to show AI debug info
        
        # Options menu
        self.showOptions = False
        
        # Status message
        self.statusMessage = "Human Mode. Press 'M' to cycle through modes, 'O' for options"
        self.statusTimer = 3000  # 3 seconds
    
    def initTetrisGame(self):
        # Initialize the tetris game using functions from tetris_game.py
        appStarted(self)
        
        # For AI training - tracking last state
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0
    
    def keyPressed(self, event):
        # Debugging output to console
        print(f"Key pressed: {event.key}, Current mode: {self.gameMode}")
        
        # Check for mode switching specifically (case insensitive)
        if event.key.lower() == "m":
            print("Cycling game mode...")
            self.cycleGameMode()
            return
        
        # Reset game
        if event.key == "r":
            self.appStarted()
            return
            
        # Other common controls across all modes
        if event.key == "o":
            self.showOptions = not self.showOptions
        elif event.key == "d":
            self.showAIDebug = not self.showAIDebug
        elif event.key == "l" and self.gameMode != "ai_player_training":
            self.loadAIModel()
        elif event.key == "p":
            self.saveAIModel()
        
        # Game controls for human player mode only
        elif self.gameMode == "human_player" and not self.isGameOver:
            # Call the keyPressed function from tetris_game.py
            # but handle its functionality directly here to avoid
            # appStarted being called again
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
    
    def cycleGameMode(self):
        """Cycle through game modes: human_player -> ai_player_watching -> ai_player_training -> human_player"""
        print(f"Cycling from current mode: {self.gameMode}")
        
        # Use a more defensive approach with explicit string comparison
        current_mode = self.gameMode
        
        if current_mode == "human_player":
            self.gameMode = "ai_player_watching"
            self.statusMessage = "AI Mode (Watching). AI plays with visualization."
            print(f"Changed to: {self.gameMode}")
        elif current_mode == "ai_player_watching":
            self.gameMode = "ai_player_training"
            self.statusMessage = "AI Mode (Training). AI trains without visualization."
            print(f"Changed to: {self.gameMode}")
        elif current_mode == "ai_player_training":
            self.gameMode = "human_player"
            self.statusMessage = "Human Mode. Use arrow keys to play."
            print(f"Changed to: {self.gameMode}")
        else:
            # Fallback in case of unexpected mode
            print(f"Unexpected mode: {current_mode}, defaulting to human_player")
            self.gameMode = "human_player"
            self.statusMessage = "Human Mode. Use arrow keys to play."
        
        # Reset game state and start training if needed
        self.initTetrisGame()  # Use our own init method instead of appStarted
        if self.gameMode in ["ai_player_training", "ai_player_watching"]:
            self.startAITraining()
        
        self.statusTimer = 5000  # 5 seconds to make message more visible
    
    def startAITraining(self):
        """Initialize AI training"""
        self.aiEpisodes = 0
        self.aiTrainingStart = time.time()
        
        # Reset AI training state to avoid stale data
        self.lastState = None
        self.lastAction = None
        self.lastScore = 0
        
        # Ensure AI is properly initialized
        if not hasattr(self, 'ai') or self.ai is None:
            from game_ai import TetrisAI
            self.ai = TetrisAI()
    
    def saveAIModel(self):
        """Save the current AI model"""
        self.ai.save_model("tetris_ai_model.pt")
        self.statusTimer = 3000
    
    def loadAIModel(self):
        """Load an AI model from file"""
        self.ai.load_model("tetris_ai_model.pt")
        self.statusTimer = 3000
    
    def timerFired(self):
        # Update status message timer
        if self.statusTimer > 0:
            self.statusTimer -= self.timerDelay
        
        # Handle game logic based on mode
        if self.gameMode == "human_player":
            self.handleHumanPlayerMode()
        elif self.gameMode == "ai_player_training":
            self.handleAITrainingMode(fast=True)
        elif self.gameMode == "ai_player_watching":
            self.handleAITrainingMode(fast=False)
    
    def handleHumanPlayerMode(self):
        """Handle game logic for human player mode using tetris_game.py functions"""
        if not self.isGameOver:
            # This is exactly what timerFired in tetris_game.py does
            if not moveFallingPiece(self, 1, 0):
                placeFallingPiece(self)
                newFallingPiece(self)
                if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
                    self.isGameOver = True
    
    def handleAITrainingMode(self, fast=False):
        """Handle game logic for AI training modes
        
        Args:
            fast (bool): If True, run without visualization delays (ai_player_training)
                        If False, run with visualization (ai_player_watching)
        """
        if fast:
            # Run multiple training steps per timer fired
            for _ in range(10):  # Adjust this number for faster training
                if self.gameMode != "ai_player_training":  # Check if mode changed during iteration
                    break
                self.executeAIMove()
        else:
            # Run a single step with visualization
            self.executeAIMove()
    
    def executeAIMove(self):
        """Execute a single AI move and update training data"""
        # Get current state
        currentState = self.ai.get_state_representation(self)
            
        # For training mode, handle experience storage
        if self.lastState is not None:
            reward = self.score - self.lastScore
            done = self.isGameOver
            self.ai.add_experience(self.lastState, self.lastAction, reward, 
                                currentState, done)
            self.ai.train()
            
        # Get all possible actions directly rather than using AI's choose_action method
        # to avoid the deepcopy operation that causes problems
        possible_actions = self.ai.get_possible_actions(self)
        if not possible_actions:
            action = None
        else:
            # Choose a random action for now (we'll improve this later)
            action = random.choice(possible_actions)
            
        # Save current state for next iteration
        self.lastState = currentState
        self.lastScore = self.score
        self.lastAction = action
            
        # Handle game over or invalid action
        if action is None or self.isGameOver:
            self.aiEpisodes += 1
            self.initTetrisGame()
            return
            
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

    def drawModeIndicator(self, canvas):
        """Draw current mode indicator in the top-right corner"""
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
        """Draw AI training information for AI modes"""
        if self.gameMode not in ["ai_player_watching", "ai_player_training"]:
            return
            
        trainingTime = time.time() - self.aiTrainingStart if self.aiTrainingStart else 0
        hours, remainder = divmod(trainingTime, 3600)
        minutes, seconds = divmod(remainder, 60)
        timeStr = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Draw training info in bottom-right area
        infoX = self.width - 150
        infoY = self.height - 80
        
        canvas.create_rectangle(
            infoX - 10, infoY - 10,
            self.width - 10, self.height - 10,
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
    
    def drawStatusMessage(self, canvas):
        """Draw temporary status message at the bottom of the screen"""
        if self.statusTimer <= 0:
            return
        
        # Opacity effect based on remaining time
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
        
        # Semi-transparent background
        canvas.create_rectangle(
            0, 0, self.width, self.height,
            fill="#white", stipple="gray50"
        )
        
        # Options box
        centerX = self.width // 2
        centerY = self.height // 2
        boxWidth = 250
        boxHeight = 180
        
        canvas.create_rectangle(
            centerX - boxWidth//2, centerY - boxHeight//2,
            centerX + boxWidth//2, centerY + boxHeight//2,
            fill="black", outline="white", width=2
        )
        
        # Title
        canvas.create_text(
            centerX, centerY - boxHeight//2 + 20,
            text="OPTIONS", fill="white", font="Arial 14 bold"
        )
        
        # Options
        options = [
            "M - Change Mode",
            "R - Reset Game",
            "L - Load AI Model",
            "P - Save AI Model",
            "O - Close Options"
        ]
        
        for i, text in enumerate(options):
            canvas.create_text(
                centerX, centerY - boxHeight//4 + i*25,
                text=text, fill="white", font="Arial 12"
            )
    
    def drawTrainingOnlyScreen(self, canvas):
        """Draw the simplified training-only screen"""
        canvas.create_rectangle(0, 0, self.width, self.height, fill="black")
        
        trainingTime = time.time() - self.aiTrainingStart if self.aiTrainingStart else 0
        hours, remainder = divmod(trainingTime, 3600)
        minutes, seconds = divmod(remainder, 60)
        timeStr = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        centerX = self.width // 2
        centerY = self.height // 2
        
        # Training status
        canvas.create_text(
            centerX, centerY - 40,
            text="AI TRAINING IN PROGRESS",
            fill="yellow", font="Arial 16 bold"
        )
        
        # Stats
        canvas.create_text(
            centerX, centerY,
            text=f"Episodes: {self.aiEpisodes}",
            fill="white", font="Arial 14"
        )
        
        canvas.create_text(
            centerX, centerY + 30,
            text=f"Training Time: {timeStr}",
            fill="white", font="Arial 14"
        )
        
        # Controls reminder
        canvas.create_text(
            centerX, centerY + 80,
            text="Press 'M' to change mode",
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
        
        # Draw mode indicator
        self.drawModeIndicator(canvas)
        
        # Draw score (using the function from tetris_game.py)
        writeScore(self, canvas)
        
        # Add game over message if needed
        if self.isGameOver:
            writeGameOver(self, canvas)
        
        # Draw AI training info (for AI modes)
        self.drawTrainingInfo(canvas)
        
        # Draw status message if active
        self.drawStatusMessage(canvas)
        
        # Draw options menu if active (this is an overlay)
        self.drawOptions(canvas)

def runTetrisWithAI():
    # Use the dimensions from tetris_game.py
    rows, cols, cellSize, margin = gameDimensions()
    game = TetrisWithAI(width=cols*cellSize+2*margin, height=rows*cellSize+2*margin)

if __name__ == "__main__":
    runTetrisWithAI()