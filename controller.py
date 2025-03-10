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

    def createInfoBox(self, canvas, x, y, width, height, title, textItems, titleFont="Arial 12 bold", textFont="Arial 10"):
        """
        Create a styled information box on the canvas with a title and multiple text items.
        """
        # Draw the background box with outline
        canvas.create_rectangle(x - 10, y - 10, x + width, y + height, 
                            fill="black", outline="white")
        
        # Draw the title
        canvas.create_text(x, y, text=title, anchor="nw", fill="white", font=titleFont)
        
        # Draw each text item
        for i, text in enumerate(textItems):
            canvas.create_text(x, y + 25 + i*20, text=text, anchor="nw", 
                            fill="white", font=textFont)

    def createCenteredBox(self, canvas, centerX, centerY, width, height, title, textItems, 
                        bgColor="black", titleColor="white", textColor="white", 
                        titleFont="Arial 16 bold", textFont="Arial 12"):
        """
        Create a centered box with title and text items.
        """
        # Draw the background box
        canvas.create_rectangle(centerX - width//2, centerY - height//2,
                            centerX + width//2, centerY + height//2,
                            fill=bgColor, outline="white", width=2)
        
        # Draw the title
        canvas.create_text(centerX, centerY - height//2 + 20, 
                        text=title, fill=titleColor, font=titleFont)
        
        # Draw each text item
        for i, text in enumerate(textItems):
            canvas.create_text(centerX, centerY - height//2 + 60 + i*25, 
                            text=text, fill=textColor, font=textFont)

    def createStatusMessage(self, canvas, message):
        """
        Create a status message at the bottom of the screen.
        """
        if self.statusTimer <= 0:
            return
        
        centerX = self.width // 2
        bottom = self.height - 30
        
        canvas.create_rectangle(centerX - 250, bottom - 20,
                            centerX + 250, bottom + 10,
                            fill="black", outline="white")
        
        canvas.create_text(centerX, bottom - 5, 
                        text=message, fill="white", font="Arial 10")
    
    def drawAIDebugInfo(self, canvas):
        """Draw AI debug information on the canvas"""
        if not self.showAIDebug:
            return
        
        # Prepare text items for the debug info
        textItems = []
        
        title = f"Mode: {self.gameMode}"
        
        if self.gameMode in ["ai_player_training", "ai_player_watching"]:
            trainingTime = time.time() - self.aiTrainingStart
            textItems.append(f"Episodes: {self.aiEpisodes}")
            textItems.append(f"Training time: {trainingTime:.1f}s")
        
        # Show memory usage
        if hasattr(self.ai, 'experience_buffer'):
            textItems.append(f"Replay buffer: {len(self.ai.experience_buffer)}/{self.ai.buffer_size}")
        
        # Draw the info box
        infoX = self.width - self.margin - 200
        infoY = self.margin
        self.createInfoBox(canvas, infoX, infoY, 200, 200, title, textItems)

    def drawOptions(self, canvas):
        """Draw options menu on the canvas"""
        if not self.showOptions:
            return
        
        # Options menu content
        optionsText = [
            "M - Cycle Game Modes",
            "R - Reset Game",
            "L - Load AI Model",
            "P - Save AI Model",
            "D - Toggle AI Debug Info",
            "O - Close Options"
        ]
        
        # Draw the centered options menu
        centerX = self.width // 2
        centerY = self.height // 2
        self.createCenteredBox(canvas, centerX, centerY, 300, 250, "OPTIONS", optionsText)

    def drawStatusMessage(self, canvas):
        """Draw status message on the canvas"""
        self.createStatusMessage(canvas, self.statusMessage)
        
    def redrawAll(self, canvas):
        # Always draw the current mode at the top for visibility
        modeText = f"CURRENT MODE: {self.gameMode.upper()}"
        canvas.create_rectangle(0, 0, self.width, 30, fill="purple")
        canvas.create_text(self.width // 2, 15, 
                          text=modeText, fill="white", font="Arial 14 bold")
        
        # Skip detailed drawing for fast training mode
        if self.gameMode == "ai_player_training":
            # Only draw minimal info
            canvas.create_rectangle(0, 30, self.width, self.height, fill="black")
            
            trainingTime = time.time() - self.aiTrainingStart if self.aiTrainingStart else 0
            infoText = [
                f"AI Training Mode (Fast)",
                f"Episodes: {self.aiEpisodes}",
                f"Training time: {trainingTime:.1f}s",
                f"Press 'M' to change mode"
            ]
            
            # Draw centered info
            centerX = self.width // 2
            centerY = self.height // 2
            self.createCenteredBox(canvas, centerX, centerY, 300, 200, "TRAINING IN PROGRESS", infoText)
            return
        
        # For human and watching modes, we use the drawing functions from tetris_game.py
        canvas.create_rectangle(0, 30, self.width, self.height, fill="black")
        
        # Draw the game board using functions from tetris_game.py
        drawBoard(self, canvas)
        drawFallingPiece(self, canvas)
        
        # Add game over message if needed
        if self.isGameOver:
            writeGameOver(self, canvas)
        
        # Draw score using function from tetris_game.py
        writeScore(self, canvas)
            
        # Draw AI debug info if enabled
        self.drawAIDebugInfo(canvas)
            
        # Draw options menu if active
        self.drawOptions(canvas)
            
        # Draw status message if active
        self.drawStatusMessage(canvas)

def runTetrisWithAI():
    # Use the dimensions from tetris_game.py
    rows, cols, cellSize, margin = gameDimensions()
    game = TetrisWithAI(width=cols*cellSize+2*margin, height=rows*cellSize+4*margin)

if __name__ == "__main__":
    runTetrisWithAI()