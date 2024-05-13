function [onsetTime] = showStimulus(trialType, window)
global squareSize circleDiameter xCenter yCenter white

% Draw the shape
switch trialType
    case 'square'
        baseRect = [0 0 squareSize squareSize];
        centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
        Screen('FillRect', window, white, centeredRect);
    case 'circle'
        baseRect = [0 0 circleDiameter circleDiameter];
        centeredRect = CenterRectOnPointd(baseRect, xCenter, yCenter);
        Screen('FillOval', window, white, centeredRect);
end

% Draw the photodiode:
drawPhotodiode('white', window)

% Flip to the screen
[~, onsetTime, ~, ~, ~] = Screen('Flip', window,[],1);

end