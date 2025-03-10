import decimal
from cmu_112_graphics import *
from configs import GAME_CONFIG
import copy
import numpy as np
import random

#################################################
# Helper functions
#################################################


def almostEqual(d1, d2, epsilon=10**-7):
    # note: use math.isclose() outside 15-112 with Python version 3.5 or later
    return (abs(d2 - d1) < epsilon)


def roundHalfUp(d):
    # Round to nearest with ties going away from zero.
    rounding = decimal.ROUND_HALF_UP
    # See other rounding options here:
    # https://docs.python.org/3/library/decimal.html#rounding-modes
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))


#################################################
# AI Interaction Functions
#################################################

def getBoardState(app):
    """
    Converts the board state to a format suitable for NN processing
    Returns a dictionary containing:
    - board_grid: current state of the board
    - current_piece: type of current piece
    - hold_piece: type of hold piece (if any)
    - next_pieces: list of next pieces
    """
    # Create a numerical representation of the board
    # 0 for empty, 1 for filled
    boardGrid = []
    for row in range(app.rows):
        boardRow = []
        for col in range(app.cols):
            if app.board[row][col] == app.emptyColor:
                boardRow.append(0)
            else:
                boardRow.append(1)
        boardGrid.append(boardRow)
    
    # Get current piece type
    currentPieceIndex = app.tetrisPieces.index(app.fallingPiece)
    currentPiece = currentPieceIndex
    
    # Get hold piece type if it exists
    holdPiece = None
    if app.holdPiece is not None:
        holdPieceIndex = app.tetrisPieces.index(app.holdPiece)
        holdPiece = holdPieceIndex
    
    # Get next pieces
    nextPieces = [app.nextPiecesIndices[i] for i in range(min(4, len(app.nextPiecesIndices)))]
    
    return {
        'board_grid': boardGrid,
        'current_piece': currentPiece,
        'hold_piece': holdPiece,
        'next_pieces': nextPieces
    }

def getColumnHeights(app):
    """
    Calculate the height of each column on the board
    """
    heights = [0] * app.cols
    for col in range(app.cols):
        for row in range(app.rows):
            if app.board[row][col] != app.emptyColor:
                heights[col] = app.rows - row
                break
    return heights

def getHoles(app):
    """
    Find the positions of all holes in the board
    A hole is an empty cell with a filled cell above it
    """
    holes = []
    for col in range(app.cols):
        foundBlock = False
        for row in range(app.rows):
            if app.board[row][col] != app.emptyColor:
                foundBlock = True
            elif foundBlock and app.board[row][col] == app.emptyColor:
                holes.append((row, col))
    return holes

def generateAllPossiblePlacements(app):
    """
    Generates all possible valid placements for the current piece
    Returns a list of (position, rotation, resulting_board, score) tuples
    """
    possiblePlacements = []
    originalPiece = copy.deepcopy(app.fallingPiece)
    originalRow = app.fallingPieceRow
    originalCol = app.fallingPieceCol
    
    # Try all rotations
    for rotation in range(4):  # Most pieces have at most 4 unique rotations
        app.fallingPiece = copy.deepcopy(originalPiece)
        for _ in range(rotation):
            _rotatePieceWithoutChecking(app)
        
        # Try all columns
        for col in range(-len(app.fallingPiece[0]), app.cols):
            app.fallingPieceRow = 0
            app.fallingPieceCol = col
            
            # Check if this placement is valid
            if fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol):
                # Simulate dropping the piece
                simulatedApp = copy.deepcopy(app)
                hardDrop(simulatedApp)
                
                # Calculate score for this move
                originalScore = app.score
                placeFallingPiece(simulatedApp)
                scoreChange = simulatedApp.score - originalScore
                
                # Save this placement
                resultingBoard = copy.deepcopy(simulatedApp.board)
                possiblePlacements.append((col, rotation, resultingBoard, scoreChange))
    
    # Restore original piece and position
    app.fallingPiece = originalPiece
    app.fallingPieceRow = originalRow
    app.fallingPieceCol = originalCol
    
    return possiblePlacements

def _rotatePieceWithoutChecking(app):
    """
    Rotates the falling piece without checking if it's legal.
    Used during placement generation.
    """
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

def executePlacement(app, col, rotation=None, action_dict=None):
    """
    Executes a placement selected by the AI
    
    Can work in two modes:
    1. Legacy mode: Takes col and rotation values
    2. Action dict mode: Takes a dictionary from choose_action with piece, row, col
    
    Args:
        app: The Tetris app instance
        col: Column position (legacy mode)
        rotation: Rotation value (legacy mode)
        action_dict: Dictionary from choose_action with piece, row, col (new mode)
    """
    if action_dict is not None:
        # New mode: Place the piece directly at the specified location
        app.fallingPiece = action_dict['piece']
        app.fallingPieceRow = action_dict['row']
        app.fallingPieceCol = action_dict['col']
        
        # Place the piece and get a new one
        placeFallingPiece(app)
        newFallingPiece(app)
    else:
        # Legacy mode: Manually move and rotate the piece
        # Reset position
        app.fallingPieceRow = 0
        app.fallingPieceCol = app.cols // 2 - len(app.fallingPiece[0]) // 2
        
        # Rotate to desired orientation
        for _ in range(rotation):
            rotateFallingPiece(app)
        
        # Move to desired column
        while app.fallingPieceCol > col:
            moveFallingPiece(app, 0, -1)
        while app.fallingPieceCol < col:
            moveFallingPiece(app, 0, 1)
        
        # Drop the piece
        hardDrop(app)
        placeFallingPiece(app)
        newFallingPiece(app)

#################################################
# Tetris Game
#################################################

#starts app with 
def appStarted(app):
    #grid and game dimension properties
    (app.rows, app.cols, app.cellSize, app.margin) = gameDimensions()
    
    #window layout properties
    app.boardLeftMargin = app.width // 2 - (app.cols * app.cellSize) // 2
    app.boardTopMargin = app.margin
    app.holdPaneWidth = 6 * app.cellSize
    app.holdPaneHeight = 6 * app.cellSize
    app.nextPaneWidth = 6 * app.cellSize
    app.nextPaneHeight = 20 * app.cellSize
    
    app.isGameOver = False
    app.score = 0
    app.canHold = True  # Can only hold once per piece
    
    #create board with all empty color
    app.emptyColor = "blue"
    app.board = [([app.emptyColor] * app.cols) for row in range(app.rows)]
    
    #pieces
    iPiece = [
        [  True,  True,  True,  True ]
    ]
    jPiece = [
        [  True, False, False ],
        [  True,  True,  True ]
    ]
    lPiece = [
        [ False, False,  True ],
        [  True,  True,  True ]
    ]
    oPiece = [
        [  True,  True ],
        [  True,  True ]
    ]
    sPiece = [
        [ False,  True,  True ],
        [  True,  True, False ]
    ]
    tPiece = [
        [ False,  True, False ],
        [  True,  True,  True ]
    ]
    zPiece = [
        [  True,  True, False ],
        [ False,  True,  True ]
    ]
    app.tetrisPieces = [iPiece, jPiece, lPiece, oPiece, sPiece, tPiece, zPiece]
    app.tetrisPieceColors = ["red", "yellow", "magenta", "pink",
                            "cyan", "green", "orange"]
    
    # Initialize hold and next pieces
    app.holdPiece = None
    app.holdPieceColor = None
    app.holdPieceUsed = False
    
    # Generate next pieces (keep 4 pieces in queue)
    app.nextPiecesIndices = []
    for _ in range(4):
        app.nextPiecesIndices.append(random.randint(0, len(app.tetrisPieces) - 1))
    
    #start game with new falling piece
    newFallingPiece(app)

#set dimensions of board to create grid dimensions and size, app size, margins
def gameDimensions():
    rows = GAME_CONFIG['rows']
    cols = GAME_CONFIG['cols']
    cellSize = GAME_CONFIG['cell_size']
    margin = GAME_CONFIG['margin']
    
    # Return tuple of dimensions
    return (rows, cols, cellSize, margin)

#starts new game with "r", moves piece with arrow keys, hard drops with space
def keyPressed(app, event):
    if event.key=="r":
        appStarted(app)
    #if game is over, stop piece-moving
    if app.isGameOver==False:
        if event.key=="Up":
            rotateFallingPiece(app)
        elif event.key=="Down":
            moveFallingPiece(app,1,0)
        elif event.key=="Left":
            moveFallingPiece(app,0,-1)
        elif event.key=="Right":
            moveFallingPiece(app,0,1)
        elif event.key=="Space":
            hardDrop(app)
            placeFallingPiece(app)
            newFallingPiece(app)
            if not fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol):
                app.isGameOver = True
        elif event.key.lower() == "c":  # Hold piece (using 'c' key)
            holdPiece(app)

#moves falling piece down one row with each timer firing
def timerFired(app):
    #only move pieces if game is ongoing
    if app.isGameOver==False:
        if (moveFallingPiece(app,1,0)==False):
            #if moving the falling piece down one row is illegal, place it in
            #the board and have a new piece fall
            placeFallingPiece(app)
            newFallingPiece(app)
            #if the piece we're trying to start falling would be illegal, the 
            #game is over
            if not(fallingPieceIsLegal(app,
                        app.fallingPieceRow,app.fallingPieceCol)):
                        app.isGameOver=True

# receives and performs DRL move
def runAIMove(app):
    """Handle AI's decision making process"""
    if not hasattr(app, '_ai_instance'):
        from game_ai import TetrisAI
        app._ai_instance = TetrisAI()
    
    # Get all possible placements
    possiblePlacements = generateAllPossiblePlacements(app)
    
    # Get the board state for AI
    boardState = getBoardState(app)
    
    # Let AI make the decision
    # In reinforcement learning, we would sometimes explore randomly
    # based on the epsilon value
    if random.random() < app.epsilon:
        # Explore - make a random move
        placement = random.choice(possiblePlacements)
        col, rotation, _, _ = placement
        executePlacement(app, col, rotation)
    else:
        # Exploit - use the AI to make the best move
        action = app._ai_instance.choose_action(app, boardState)
        executePlacement(app, None, None, action_dict=action)

#draws board and pieces on canvas
def drawCell(app, canvas, row, col, color, leftMargin=None, topMargin=None):
    # Use the provided margins, or default to the board margins
    leftMargin = leftMargin if leftMargin is not None else app.boardLeftMargin
    topMargin = topMargin if topMargin is not None else app.boardTopMargin
    
    canvas.create_rectangle(leftMargin + col * app.cellSize,
                            topMargin + row * app.cellSize,
                            leftMargin + (col + 1) * app.cellSize,
                            topMargin + (row + 1) * app.cellSize,
                            fill=color)

#draws board 
def drawBoard(app, canvas):
    # Draw grid background
    for row in range(app.rows):
        for col in range(app.cols):
            drawCell(app, canvas, row, col, app.board[row][col])
    
    # Draw grid border
    canvas.create_rectangle(app.boardLeftMargin, app.boardTopMargin,
                            app.boardLeftMargin + app.cols * app.cellSize,
                            app.boardTopMargin + app.rows * app.cellSize,
                            width=2, outline="white")

# Draw the hold piece panel
def drawHoldPane(app, canvas):
    # Draw hold panel background
    holdLeftMargin = app.margin
    holdTopMargin = app.margin
    
    # Draw hold panel border
    canvas.create_rectangle(holdLeftMargin, holdTopMargin,
                            holdLeftMargin + app.holdPaneWidth,
                            holdTopMargin + app.holdPaneHeight,
                            fill="black", width=2, outline="white")
    
    # Draw "HOLD" text
    canvas.create_text(holdLeftMargin + app.holdPaneWidth // 2,
                      holdTopMargin + 20,
                      text="HOLD", fill="white",
                      font="Arial 16 bold")
    
    # Draw the hold piece if it exists
    if app.holdPiece is not None:
        # Center the piece in the hold panel
        pieceRows = len(app.holdPiece)
        pieceCols = len(app.holdPiece[0])
        centerRow = (app.holdPaneHeight - pieceRows * app.cellSize) // 2
        centerCol = (app.holdPaneWidth - pieceCols * app.cellSize) // 2
        
        for row in range(pieceRows):
            for col in range(pieceCols):
                if app.holdPiece[row][col]:
                    drawCell(app, canvas, 
                             row + centerRow // app.cellSize + 1,  # +1 to account for the "HOLD" text
                             col + centerCol // app.cellSize,
                             app.holdPieceColor,
                             holdLeftMargin, holdTopMargin)

# Draw the next pieces panel
def drawNextPiecesPane(app, canvas):
    # Calculate next pieces panel position
    nextLeftMargin = app.boardLeftMargin + app.cols * app.cellSize + app.margin
    nextTopMargin = app.margin
    
    # Draw next pieces panel border
    canvas.create_rectangle(nextLeftMargin, nextTopMargin,
                            nextLeftMargin + app.nextPaneWidth,
                            nextTopMargin + app.nextPaneHeight,
                            fill="black", width=2, outline="white")
    
    # Draw "NEXT" text
    canvas.create_text(nextLeftMargin + app.nextPaneWidth // 2,
                      nextTopMargin + 20,
                      text="NEXT", fill="white",
                      font="Arial 16 bold")
    
    # Draw the next 4 pieces
    for i in range(min(4, len(app.nextPiecesIndices))):
        pieceIndex = app.nextPiecesIndices[i]
        piece = app.tetrisPieces[pieceIndex]
        pieceColor = app.tetrisPieceColors[pieceIndex]
        
        # Center each piece horizontally in the next panel
        pieceRows = len(piece)
        pieceCols = len(piece[0])
        centerCol = (app.nextPaneWidth - pieceCols * app.cellSize) // 2
        
        # Space pieces vertically, starting after the "NEXT" text
        startRow = 2 + i * 5  # 2 rows for "NEXT" text, then 5 rows per piece
        
        for row in range(pieceRows):
            for col in range(pieceCols):
                if piece[row][col]:
                    drawCell(app, canvas,
                             startRow + row,
                             centerCol // app.cellSize + col,
                             pieceColor,
                             nextLeftMargin, nextTopMargin)

#creates new falling piece after previous piece is placed on board
def newFallingPiece(app):
    # Get the next piece from the queue
    if not app.nextPiecesIndices:
        # If queue is empty (shouldn't happen normally), add new pieces
        app.nextPiecesIndices.append(random.randint(0, len(app.tetrisPieces) - 1))
    
    randomIndex = app.nextPiecesIndices.pop(0)
    app.fallingPiece = app.tetrisPieces[randomIndex]
    app.fallingPieceColor = app.tetrisPieceColors[randomIndex]
    
    # Add a new piece to the end of the queue
    app.nextPiecesIndices.append(random.randint(0, len(app.tetrisPieces) - 1))
    
    # Reset hold usage flag
    app.holdPieceUsed = False
    
    #starts at the top
    app.fallingPieceRow = 0
    numFallingPieceCols = len(app.fallingPiece[0])
    #roughly centered halfway across board
    app.fallingPieceCol = app.cols // 2 - numFallingPieceCols // 2

# Hold the current piece
def holdPiece(app):
    # Can only hold once per piece
    if app.holdPieceUsed:
        return
    
    # Get current piece index
    currentPieceIndex = -1
    for i, piece in enumerate(app.tetrisPieces):
        if piece == app.fallingPiece:
            currentPieceIndex = i
            break
    
    if currentPieceIndex == -1:
        return  # Shouldn't happen if pieces are properly set up
    
    # If no piece is being held, swap with the next piece
    if app.holdPiece is None:
        app.holdPiece = app.tetrisPieces[currentPieceIndex]
        app.holdPieceColor = app.tetrisPieceColors[currentPieceIndex]
        newFallingPiece(app)
    else:
        # Swap the current piece with the hold piece
        tempPiece = app.holdPiece
        tempColor = app.holdPieceColor
        
        app.holdPiece = app.tetrisPieces[currentPieceIndex]
        app.holdPieceColor = app.tetrisPieceColors[currentPieceIndex]
        
        # Get the index of the hold piece
        holdPieceIndex = -1
        for i, piece in enumerate(app.tetrisPieces):
            if piece == tempPiece:
                holdPieceIndex = i
                break
        
        if holdPieceIndex != -1:
            app.fallingPiece = app.tetrisPieces[holdPieceIndex]
            app.fallingPieceColor = app.tetrisPieceColors[holdPieceIndex]
            
            # Reset position
            app.fallingPieceRow = 0
            numFallingPieceCols = len(app.fallingPiece[0])
            app.fallingPieceCol = app.cols // 2 - numFallingPieceCols // 2
    
    # Mark that hold has been used for this piece
    app.holdPieceUsed = True

#draws falling piece on board
def drawFallingPiece(app, canvas):
    for row in range(len(app.fallingPiece)):
        for col in range(len(app.fallingPiece[0])):
            #draws each cell location of piece over board
            if app.fallingPiece[row][col] == True:
                drawCell(app, canvas,
                        row + app.fallingPieceRow, col + app.fallingPieceCol,
                        app.fallingPieceColor)

#moves falling piece down, left, or right if it is legal, then returns if the
#move was legal or not
def moveFallingPiece(app, drow, dcol):
    app.fallingPieceRow += drow
    app.fallingPieceCol += dcol
    #if the requested move isn't legal, undo the changes
    if fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol) == False:
        app.fallingPieceRow -= drow
        app.fallingPieceCol -= dcol
        return False
    return True

#checks if the piece is trying to move off the screen or onto a placed piece
def fallingPieceIsLegal(app, row, col):
    for pieceRow in range(len(app.fallingPiece)):
        for pieceCol in range(len(app.fallingPiece[0])):
            if app.fallingPiece[pieceRow][pieceCol] == True:
                boardRow = pieceRow + row
                boardCol = pieceCol + col
                #check if the row or column is out of range
                if (boardRow < 0 
                or boardRow >= app.rows
                or boardCol < 0 
                or boardCol >= app.cols
                #or if the location that the piece cell is about to go is 
                #already occupied by a placed piece
                or app.board[boardRow][boardCol] != app.emptyColor):
                    return False
    return True

#rotates falling piece counterclockwise
def rotateFallingPiece(app):
    #set old piece properties to temporary variables in case the rotation is 
    #illegal to set the piece back to its original location and orientation
    oldPiece = app.fallingPiece
    oldRow, oldCol = app.fallingPieceRow, app.fallingPieceCol
    oldNumRows = len(oldPiece)
    oldNumCols = len(oldPiece[0])
    #new piece will have switched row and column dimensions
    newNumRows, newNumCols = oldNumCols, oldNumRows
    #initialize new falling piece
    rotatedFallingPiece = [([None] * newNumCols) for row in range(newNumRows)]
    #iterate across each column starting from the end, moving down rows to 
    #rotate piece
    i = 0
    for col in range(oldNumCols - 1, -1, -1):
        j = 0
        for row in range(oldNumRows):
            rotatedFallingPiece[i][j] = oldPiece[row][col]
            j += 1
        i += 1
    #set the piece to the new dimensions and location
    app.fallingPiece = rotatedFallingPiece
    newRow = oldRow + oldNumRows // 2 - newNumRows // 2
    newCol = oldCol + oldNumCols // 2 - newNumCols // 2
    app.fallingPieceRow, app.fallingPieceCol = newRow, newCol
    #if the move was illegal, set the piece properties back to how it was first
    if not(fallingPieceIsLegal(app, app.fallingPieceRow, app.fallingPieceCol)):
        app.fallingPiece = oldPiece
        app.fallingPieceRow = oldRow
        app.fallingPieceCol = oldCol

#places a falling piece on the board
def placeFallingPiece(app):
    #iterating through each element in the piece, set the new board to it
    for row in range(len(app.fallingPiece)):
        for col in range(len(app.fallingPiece[0])):
            if app.fallingPiece[row][col] == True:
                (app.board[row + app.fallingPieceRow]
                [col + app.fallingPieceCol]) = app.fallingPieceColor
    #check if placing the falling piece resulted in a filled row that should be
    #removed
    removeFullRows(app)

#removes rows filled with placed pieces and adds to score
def removeFullRows(app):
    #create a temporary board
    tempBoard = [([app.emptyColor] * app.cols) for row in range(app.rows)]
    newRow = app.rows - 1
    rowsAdded = 0
    for row in range(app.rows - 1, -1, -1):
        for col in range(app.cols):
            #iterating backwards so we always add rows in the right order, 
            #we know we should keep a row if there is any column with the 
            #set empty color of the grid
            if app.board[row][col] == app.emptyColor:
                #we have now added (kept) one more row
                rowsAdded += 1
                tempBoard[newRow] = app.board[row]
                newRow -= 1
                #break so we don't repeat this process for multiple columns in
                #each row we want to keep
                break
    #add to app score and set the board to the temporary board
    app.score += (app.rows - rowsAdded) ** 2
    app.board = tempBoard

#drops falling piece to the lowest (visually) possible row
def hardDrop(app):
    while moveFallingPiece(app, 1, 0):
            continue

#draws rectangle with words "game over" written over it
def writeGameOver(app, canvas):
    canvas.create_rectangle(app.boardLeftMargin, app.boardTopMargin,
                           app.boardLeftMargin + app.cols * app.cellSize,
                           app.boardTopMargin + 2 * app.cellSize,
                           fill="yellow")
    canvas.create_text(app.boardLeftMargin + (app.cols * app.cellSize) // 2,
                      app.boardTopMargin + app.cellSize,
                      text="Game Over", fill="black",
                      font="Helvetica 26 bold")

#writes score at top of canvas
def writeScore(app, canvas):
    canvas.create_text(app.width // 2, app.margin // 2,
                      text=f"Score: {app.score}",
                      fill="white", font="Helvetica 16 bold")

#draws all elements in game
def redrawAll(app, canvas):
    # Draw background
    canvas.create_rectangle(0, 0, app.width, app.height, fill="black")
    
    # Draw game elements
    drawBoard(app, canvas)
    drawFallingPiece(app, canvas)
    drawHoldPane(app, canvas)
    drawNextPiecesPane(app, canvas)
    writeScore(app, canvas)
    
    # Draw game over message if needed
    if app.isGameOver:
        writeGameOver(app, canvas)

    # Draw key instructions
    instructionY = app.height - app.margin // 2
    canvas.create_text(app.width // 2, instructionY,
                      text="Controls: ←→↓: Move, ↑: Rotate, Space: Drop, C: Hold",
                      fill="white", font="Arial 12")

#runs app
def playTetris():
    # Calculate window width to accommodate the board, hold pane, and next pieces pane
    rows, cols, cellSize, margin = gameDimensions()
    boardWidth = cols * cellSize
    holdPaneWidth = 6 * cellSize
    nextPaneWidth = 6 * cellSize
    
    # Calculate window dimensions with extra space for margins between panes
    windowWidth = holdPaneWidth + boardWidth + nextPaneWidth + 4 * margin
    windowHeight = rows * cellSize + 2 * margin
    
    runApp(width=windowWidth, height=windowHeight)

#################################################
# main
#################################################

def main():
    playTetris()

if __name__ == '__main__':
    main()