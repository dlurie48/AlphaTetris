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
        appStarted(self)
        
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
            self.executeAIMove(training = True)
            return
        
        # Handle AI gameplay
        elif self.aiMode and not self.isGameOver and not self.aiThinking:
            self.aiThinking = True
            self.executeAIMove(training = False)
            self.aiThinking = False
            return
        
        # Regular game logic for human play
        elif not self.isGameOver and not self.aiMode:
            if not moveFallingPiece(self, 1, 0):
                # If moving down is illegal, place piece and get a new one
                placeFallingPiece(self)
                newFallingPiece(self)
                if not fallingPieceIsLegal(self, self.fallingPieceRow, self.fallingPieceCol):
                    self.isGameOver = True
    
    def executeAIMove(self, training=False):
        # Get current state
        currentState = self.ai.get_state_representation(self)
        
        # For training mode, handle experience storage
        if training and self.lastState is not None:
            reward = self.score - self.lastScore
            done = self.isGameOver
            self.ai.add_experience(self.lastState, self.lastAction, reward, 
                                currentState, done)
            self.ai.train()
        
        # Choose action
        action = self.ai.choose_action(self, currentState)
        
        # For training mode, save current state
        if training:
            self.lastState = currentState
            self.lastScore = self.score
            self.lastAction = action
        
        # Handle game over or invalid action
        if action is None or self.isGameOver:
            if training:
                self.aiEpisodes += 1
                self.initTetrisGame()
            else:
                self.isGameOver = True
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
        
        Args:
            canvas: The tkinter canvas to draw on
            x, y: Top-left coordinates of the text (box will extend around this)
            width, height: Dimensions of the box
            title: Title text to display at the top
            textItems: List of strings to display as separate lines
            titleFont: Font specification for the title
            textFont: Font specification for the text items
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
        
        Args:
            canvas: The tkinter canvas to draw on
            centerX, centerY: Center coordinates of the box
            width, height: Dimensions of the box
            title: Title text to display at the top
            textItems: List of strings to display as separate lines
            bgColor: Background color of the box
            titleColor: Color of the title text
            textColor: Color of the text items
            titleFont: Font specification for the title
            textFont: Font specification for the text items
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
        
        Args:
            canvas: The tkinter canvas to draw on
            message: The status message to display
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
        if not self.showAIDebug:
            return
        
        # Prepare text items for the debug info
        textItems = []
        
        modeText = "AI Mode" if self.aiMode else "Human Mode"
        trainingText = "TRAINING" if self.aiTraining else ""
        title = f"{modeText} {trainingText}"
        
        if self.aiTraining:
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
        if not self.showOptions:
            return
        
        # Options menu content
        optionsText = [
            "A - Toggle AI Mode",
            "T - Start AI Training",
            "S - Stop AI Training",
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
        self.createStatusMessage(canvas, self.statusMessage)
        
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