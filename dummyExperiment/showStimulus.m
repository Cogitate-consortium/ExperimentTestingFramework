function [onsetTime] = showStimulus(texture, window, stimWidth, stimHeight, xCenter, yCenter)
global isTestMode
% Stimulus coordinates:
stimRect = [0, 0, stimWidth, stimHeight];
destRect = CenterRectOnPointd(stimRect, xCenter, yCenter);

% Draw the shape
Screen('DrawTexture', window, texture, [], destRect, 0);

% Draw the photodiode:
if isTestMode
    drawPhotodiode('white', window)
end
% Flip to the screen
[~, onsetTime, ~, ~, ~] = Screen('Flip', window,[],1);

end