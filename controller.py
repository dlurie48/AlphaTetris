from cmu_112_graphics import *
from tetris_game import *
from game_ai import TetrisAI, train_tetris_ai
import copy
import time

class TetrisWithAI(App):
    def appStarted(self):
        # Initialize the base Tetris game
        self.initTetrisGame()
        
        # AI settings
        self.ai = TetrisAI()
        self.aiMode = False  # Start in human play mode
        self.aiThinking = False
        self.aiMoveDelay = 500  # ms between AI moves
        self.aiTraining = False
        self.aiEpisodes = 0
        self.aiTrainingStart = None
        self.showAIDebug = False  # Whether to show AI debug info
        
        # Options menu
        self.showOptions = False
        
        # Status message
        self.statusMessage = "Press 'A' to toggle AI, 'T' to train AI, 'O' for options"
        self.statusTimer = 0
    
    def initTetrisGame(self):
        # Get game dimensions
        (self.rows, self.cols, self.cellSize, self.margin) = gameDimensions()
        self.isGameOver = False
        self.score = 0
        
        # Create board with all empty color
        self.emptyColor = "blue"
        self.board = [([self.emptyColor] * self.cols) for row in range(self.rows)]
        
        # Tetris pieces (copied from tetris_game.py)
        iPiece = [[True, True, True, True]]
        jPiece = [[True, False, False], [True, True, True]]
        lPiece = [[False, False, True], [True, True, True]]
        oPiece = [[True, True], [True, True]]
        sPiece = [[False, True, True], [True, True, False]]
        tPiece = [[False, True, False], [True, True, True]]
        zPiece = [[True, True, False], [False, True, True]]
        
        self.tetrisPieces = [iPiece, jPiece, lPiece, oPiece, sPiece, tPiece, zPiece]
        self.tetrisPieceColors = ["red", "yellow", "magenta", "pink", "cyan", "green", "orange"]
        
        # Start with new falling piece
        newFallingPiece(self)
        
        # For AI training - tracking last state
        self.lastState = None
        self.lastAction = None
    
    def keyPressed(self, event):
        # Game controls
        if not self.aiMode and not self.aiTraining and not self.isGameOver:
            if event.key == "r":
                self.appStarted()
            elif event.key == "Up":
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
        
        # AI controls
        if event.key == "a" and not self.aiTraining:
            self.aiMode = not self.aiMode
            if self.aiMode:
                self.statusMessage = "AI Mode activated. AI will play automatically."
            else:
                self.statusMessage = "Human Mode activated. Use arrow keys to play."
            self.statusTimer = 3000  # 3 seconds
        
        elif event.key == "t" and not self.aiMode and not self.aiTraining:
            self.startAITraining()
        
        elif event.key == "s" and self.aiTraining:
            self.stopAITraining()
        
        elif event.key == "o":
            self.showOptions = not self.showOptions
        
        elif event.key == "d":
            self.showAIDebug = not self.showAIDebug
        
        elif event.key == "l" and not self.aiTraining:
            self.loadAIModel()
        
        elif event.key == "p":
            self.saveAIModel()
    
    def startAITraining(self):
        self.aiTraining = True
        self.appStarted()  # Reset the game
        self.aiEpisodes = 0
        self.aiTrainingStart = time.time()
        self.statusMessage = "AI Training started. Press 'S' to stop."
        self.statusTimer = 3000
    
    def stopAITraining(self):
        self.aiTraining = False
        trainingTime = time.time() - self.aiTrainingStart
        self.statusMessage = f"AI Training stopped after {self.aiEpisodes} episodes ({trainingTime:.1f}s)"
        self.statusTimer = 3000
        self.appStarted()  # Reset the game
    
    def saveAIModel(self):
        try:
            self.ai.save_model("tetris_ai_model.pt")
            self.statusMessage = "AI model saved successfully."
        except Exception as e:
            self.statusMessage = f"Error saving AI model: {str(e)}"
        self.statusTimer = 3000
    
    def loadAIModel(self):
        try:
            self.ai.load_model("tetris_ai_model.pt")
            self.statusMessage = "AI model loaded successfully."
        except Exception as e:
            self.statusMessage = f"Error loading AI model: {str(e)}"
        self.statusTimer = 3000
    
    def timerFired(self):
        # Update status message timer
        if self.statusTimer > 0:
            self.statusTimer -= self.timerDelay
        
        # Handle AI training
        if self.aiTraining:
            self.runAITrainingStep()
            return
        
        # Handle AI gameplay
        if self.aiMode and not self.isGameOver and not self.aiThinking:
            self.aiThinking = True
            self.makeAIMove()
            self.aiThinking = False
            return
        
        # Regular game logic for human play
        if not self.isGameOver and not self.aiMode:
            if not moveFallingPiece(self, 1, 0):
                # If moving down is illegal, place piece and get a new one
                placeFallingPiece(self)
                newFallingPiece(self)
                if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
                    self.isGameOver = True
    
    def makeAIMove(self):
        # Get current state representation
        currentState = self.ai.get_state_representation(self)
        
        # Choose an action
        action = self.ai.choose_action(self, currentState)
        
        if action is None:
            self.isGameOver = True
            return
        
        # Apply action
        self.fallingPiece = action['piece']
        self.fallingPieceRow = action['row']
        self.fallingPieceCol = action['col']
        
        # Place the piece
        placeFallingPiece(self)
        
        # Create new piece
        newFallingPiece(self)
        
        # Check if game is over
        if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
            self.isGameOver = True
    
    def runAITrainingStep(self):
        # Get current state
        currentState = self.ai.get_state_representation(self)
        
        if self.lastState is not None:
            # Calculate reward (score difference)
            reward = self.score - self.lastScore
            
            # Add experience to replay buffer
            done = self.isGameOver
            self.ai.add_experience(self.lastState, self.lastAction, reward, currentState, done)
            
            # Train the model
            self.ai.train()
        
        # Save current state
        self.lastState = currentState
        self.lastScore = self.score
        
        # Choose action
        action = self.ai.choose_action(self, currentState)
        self.lastAction = action
        
        if action is None or self.isGameOver:
            # Game over, start a new episode
            self.aiEpisodes += 1
            self.initTetrisGame()
            return
        
        # Apply action
        self.fallingPiece = action['piece']
        self.fallingPieceRow = action['row']
        self.fallingPieceCol = action['col']
        
        # Place the piece
        placeFallingPiece(self)
        
        # Create new piece
        newFallingPiece(self)
        
        # Check if game is over
        if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
            self.isGameOver = True
    
    def drawAIDebugInfo(self, canvas):
        if not self.showAIDebug:
            return
        
        # Draw AI debug information
        infoX = self.width - self.margin - 200
        infoY = self.margin
        
        canvas.create_rectangle(infoX - 10, infoY - 10, 
                               self.width - self.margin + 10, infoY + 200,
                               fill="black", outline="white")
        
        # Show AI state
        modeText = "AI Mode" if self.aiMode else "Human Mode"
        trainingText = "TRAINING" if self.aiTraining else ""
        canvas.create_text(infoX, infoY, text=f"{modeText} {trainingText}", 
                         anchor="nw", fill="white", font="Arial 12 bold")
        
        if self.aiTraining:
            trainingTime = time.time() - self.aiTrainingStart
            canvas.create_text(infoX, infoY + 25, 
                             text=f"Episodes: {self.aiEpisodes}", 
                             anchor="nw", fill="white", font="Arial 10")
            canvas.create_text(infoX, infoY + 45, 
                             text=f"Training time: {trainingTime:.1f}s", 
                             anchor="nw", fill="white", font="Arial 10")
        
        # Show memory usage
        if hasattr(self.ai, 'experience_buffer'):
            canvas.create_text(infoX, infoY + 70, 
                             text=f"Replay buffer: {len(self.ai.experience_buffer)}/{self.ai.buffer_size}", 
                             anchor="nw", fill="white", font="Arial 10")
    
    def drawOptions(self, canvas):
        if not self.showOptions:
            return
        
        # Draw options menu
        centerX = self.width // 2
        centerY = self.height // 2
        width = 300
        height = 250
        
        canvas.create_rectangle(centerX - width//2, centerY - height//2,
                              centerX + width//2, centerY + height//2,
                              fill="black", outline="white", width=2)
        
        canvas.create_text(centerX, centerY - height//2 + 20, 
                         text="OPTIONS", fill="white", font="Arial 16 bold")
        
        optionsText = [
            "A - Toggle AI Mode",
            "T - Start AI Training",
            "S - Stop AI Training",
            "L - Load AI Model",
            "P - Save AI Model",
            "D - Toggle AI Debug Info",
            "O - Close Options"
        ]
        
        for i, text in enumerate(optionsText):
            canvas.create_text(centerX, centerY - height//2 + 60 + i*25, 
                             text=text, fill="white", font="Arial 12")
    
    def drawStatusMessage(self, canvas):
        if self.statusTimer <= 0:
            return
        
        # Draw status message at the bottom of the screen
        centerX = self.width // 2
        bottom = self.height - 30
        
        canvas.create_rectangle(centerX - 250, bottom - 20,
                              centerX + 250, bottom + 10,
                              fill="black", outline="white")
        
        canvas.create_text(centerX, bottom - 5, 
                         text=self.statusMessage, fill="white", font="Arial 10")
    
    def redrawAll(self, canvas):
        # Draw the base Tetris game
        canvas.create_rectangle(0, 0, self.width, self.height, fill="black")
        
        # Draw the game area
        canvas.create_rectangle(self.margin - 5, self.margin - 5,
                              self.margin + self.cols * self.cellSize + 5,
                              self.margin + self.rows * self.cellSize + 5,
                              fill="orange", width=2)
        
        # Draw the standard game elements
        drawBoard(self, canvas)
        drawFallingPiece(self, canvas)
        
        # Draw score
        scoreText = f"Score: {self.score}"
        canvas.create_text(self.margin, 10, text=scoreText,
                         anchor="nw", fill="white", font="Arial 14 bold")
        
        # Draw game over message if needed
        if self.isGameOver:
            centerX = self.margin + (self.cols * self.cellSize) // 2
            centerY = self.margin + (self.rows * self.cellSize) // 2
            
            canvas.create_rectangle(centerX - 100, centerY - 30,
                                  centerX + 100, centerY + 30,
                                  fill="black", outline="white", width=2)
            
            canvas.create_text(centerX, centerY,
                             text="GAME OVER", fill="white", font="Arial 20 bold")
            
            canvas.create_text(centerX, centerY + 25,
                            text=f"Final Score: {self.score}", fill="white", font="Arial 12")
        
        # Draw AI debug info if enabled
        self.drawAIDebugInfo(canvas)
        
        # Draw options menu if active
        self.drawOptions(canvas)
        
        # Draw status message if active
        self.drawStatusMessage(canvas)

def runTetrisWithAI():
    game = TetrisWithAI(width=600, height=700)

if __name__ == "__main__":
    runTetrisWithAI()