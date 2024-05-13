function drawPhotodiode(color, window)
global black white photoSquareSize

switch color
    case 'black'
        squareColor = black;
    case 'white'
        squareColor = white;
end

baseRect = [0 0 photoSquareSize photoSquareSize];
Screen('FillRect', window, squareColor, baseRect);
end