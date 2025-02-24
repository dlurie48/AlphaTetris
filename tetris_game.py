from cmu_112_graphics import *

#################################################
# Helper functions
#################################################

def almostEqual(d1, d2, epsilon=10**-7):
    # note: use math.isclose() outside 15-112 with Python version 3.5 or later
    return (abs(d2 - d1) < epsilon)

import decimal
def roundHalfUp(d):
    # Round to nearest with ties going away from zero.
    rounding = decimal.ROUND_HALF_UP
    # See other rounding options here:
    # https://docs.python.org/3/library/decimal.html#rounding-modes
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))


#################################################
# Tetris
#################################################

#starts app with 
def appStarted(app):
    #grid and game dimension properties
    (app.rows,app.cols,app.cellSize,app.margin)=gameDimensions()
    app.isGameOver=False
    app.score=0
    #create board with all empty color
    app.emptyColor="blue"
    app.board=[([app.emptyColor]*app.cols) for row in range(app.rows)]
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
    app.tetrisPieces=[iPiece,jPiece,lPiece,oPiece,sPiece,tPiece,zPiece]
    app.tetrisPieceColors=["red","yellow","magenta","pink",
                            "cyan","green","orange"]
    #start game with new falling piece
    newFallingPiece(app)

#set dimensions of board to create grid dimensions and size, app size, margins
def gameDimensions():
    return (25,30,40,30)

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

#draws board and pieces on canvas
def drawCell(app,canvas,row,col,color):
    canvas.create_rectangle(app.margin+col*app.cellSize,
                            app.margin+row*app.cellSize,
                            app.margin+(col+1)*app.cellSize,
                            app.margin+(row+1)*app.cellSize,
                            fill=color)

#draws board 
def drawBoard(app,canvas):
    for row in range(app.rows):
        for col in range(app.cols):
            drawCell(app,canvas,row,col,app.board[row][col])

#creates new falling piece after previous piece is placed on board
def newFallingPiece(app):
    import random
    #randomly chooses type and color of piece
    randomIndex=random.randint(0,len(app.tetrisPieces)-1)
    app.fallingPiece=app.tetrisPieces[randomIndex]
    app.fallingPieceColor=app.tetrisPieceColors[randomIndex]
    #starts at the top
    app.fallingPieceRow=0
    numFallingPieceCols=len(app.fallingPiece[0])
    #roughly centered halfway across board
    app.fallingPieceCol=app.cols//2-numFallingPieceCols//2

#draws falling piece on board
def drawFallingPiece(app,canvas):
    for row in range(len(app.fallingPiece)):
        for col in range(len(app.fallingPiece[0])):
            #draws each cell location of piece over board
            if app.fallingPiece[row][col]==True:
                drawCell(app,canvas,
                        row+app.fallingPieceRow,col+app.fallingPieceCol,
                        app.fallingPieceColor)

#moves falling piece down, left, or right if it is legal, then returns if the
#move was legal or not
def moveFallingPiece(app,drow,dcol):
    app.fallingPieceRow+=drow
    app.fallingPieceCol+=dcol
    #if the requested move isn't legal, undo the changes
    if fallingPieceIsLegal(app,app.fallingPieceRow,app.fallingPieceCol)==False:
        app.fallingPieceRow-=drow
        app.fallingPieceCol-=dcol
        return False
    return True

#checks if the piece is trying to move off the screen or onto a placed piece
def fallingPieceIsLegal(app,row,col):
    for row in range(len(app.fallingPiece)):
        for col in range(len(app.fallingPiece[0])):
            if app.fallingPiece[row][col]==True:
                #check if the row or column is out of range
                if (row+app.fallingPieceRow<0 
                or row+app.fallingPieceRow>=app.rows
                or col+app.fallingPieceCol<0 
                or col+app.fallingPieceCol>=app.cols
                #or if the location that the piece cell is about to go is 
                #already occupied by a placed piece
                or app.board[row+app.fallingPieceRow]
                [col+app.fallingPieceCol]!=app.emptyColor):
                    return False
    return True

#rotates falling piece counterclockwise
def rotateFallingPiece(app):
    #set old piece properties to temporary variables in case the rotation is 
    #illegal to set the piece back to its original location and orientation
    oldPiece=app.fallingPiece
    oldRow,oldCol=app.fallingPieceRow,app.fallingPieceCol
    oldNumRows=len(oldPiece)
    oldNumCols=len(oldPiece[0])
    #new piece will have switched row and column dimensions
    newNumRows,newNumCols=oldNumCols,oldNumRows
    #initialize new falling piece
    rotatedFallingPiece=[([None]*newNumCols) for row in range(newNumRows)]
    #iterate across each column starting from the end, moving down rows to 
    #rotate piece
    i=0
    for col in range(oldNumCols-1,-1,-1):
        j=0
        for row in range(oldNumRows):
            rotatedFallingPiece[i][j]=oldPiece[row][col]
            j+=1
        i+=1
    #set the piece to the new dimensions and location
    app.fallingPiece=rotatedFallingPiece
    newRow=oldRow+oldNumRows//2-newNumRows//2
    newCol=oldCol+oldNumCols//2-newNumCols//2
    app.fallingPieceRow,app.fallingPieceCol=newRow,newCol
    #if the move was illegal, set the piece properties back to how it was first
    if not(fallingPieceIsLegal(app,app.fallingPieceRow,app.fallingPieceCol)):
        app.fallingPiece=oldPiece
        app.fallingPieceRow=oldRow
        app.fallingPieceCol=oldCol

#places a falling piece on the board
def placeFallingPiece(app):
    #iterating through each element in the piece, set the new board to it
    for row in range(len(app.fallingPiece)):
        for col in range(len(app.fallingPiece[0])):
            if app.fallingPiece[row][col]==True:
                (app.board[row+app.fallingPieceRow]
                [col+app.fallingPieceCol])=app.fallingPieceColor
    #check if placing the falling piece resulted in a filled row that should be
    #removed
    removeFullRows(app)

#removes rows filled with placed pieces and adds to score
def removeFullRows(app):
    #create a temporary board
    tempBoard=[([app.emptyColor]*app.cols) for row in range(app.rows)]
    newRow=app.rows-1
    rowsAdded=0
    for row in range(app.rows-1,-1,-1):
        for col in range(app.cols):
            #iterating backwards so we always add rows in the right order, 
            #we know we should keep a row if there is any column with the 
            #set empty color of the grid
            if app.board[row][col]==app.emptyColor:
                #we have now added (kept) one more row
                rowsAdded+=1
                tempBoard[newRow]=app.board[row]
                newRow-=1
                #break so we don't repeat this process for multiple columns in
                #each row we want to keep
                break
    #add to app score and set the board to the temporary board
    app.score+=(app.rows-rowsAdded)**2
    app.board=tempBoard

#drops falling piece to the lowest (visually) possible row
def hardDrop(app):
    while moveFallingPiece(app,1,0):
            continue

#draws rectangle with words "game over" written over it
def writeGameOver(app,canvas):
    (rows,cols,cellSize,margin)=gameDimensions()
    if app.isGameOver:
        canvas.create_rectangle(0,margin,cols*cellSize+2*margin,
                                2*cellSize+margin,fill="yellow")
        canvas.create_text((cols*cellSize+2*margin)//2,margin, 
                            text="Game Over",anchor="n",
                            fill="black", font='Helvetica 26 bold')

#writes score at top of canvas
def writeScore(app,canvas):
    (rows,cols,cellSize,margin)=gameDimensions()
    canvas.create_text((cols*cellSize+2*margin)//2,0,text=f"Score:{app.score}",
                        anchor="n",fil="white",font="Helvetica 16 bold")

#draws all elements in game
def redrawAll(app, canvas):
    canvas.create_rectangle(0,0,app.cols*app.cellSize+2*app.margin,
                            app.rows*app.cellSize+2*app.margin,fill="orange")
    drawBoard(app,canvas)
    drawFallingPiece(app,canvas)
    writeGameOver(app,canvas)
    writeScore(app,canvas)

#runs app
def playTetris():
    (rows,cols,cellSize,margin)=gameDimensions()
    runApp(width=cols*cellSize+2*margin,height=rows*cellSize+2*margin)

#################################################
# main
#################################################

def main():
    #cs112_s22_week6_linter.lint()
    #testAll()
    #s22MidtermAnimation_()
    playTetris()

if __name__ == '__main__':
    main()
