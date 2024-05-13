% Clear the workspace and the screen
clear;
close all;
sca;

% Set the global variables:
global squareSize circleDiameter xCenter yCenter white black screenYpixels screenXpixels photoSquareSize

%% Setup Psychtoolbox:
PsychDefaultSetup(1);
screenNumber = max(Screen('Screens'));

% Define white, black, and grey
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
grey = white / 2;

% Open the screen
Screen('Preference', 'SkipSyncTests', 1);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);

% Get the size of the screen
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% Get the center coordinates of the window
[xCenter, yCenter] = RectCenter(windowRect);

% Get the refresh rate:
hz  = Screen('NominalFrameRate', window); % Screen refresh rate
ifi = 1 / hz;                             % Inter frame interval

% Text settings
Screen('TextSize',  window, 40);
Screen('TextFont', window, 'Arial');

%% Design parameters
shapes        = {'square', 'circle'};  % Shapes to presented
durations     = round([1.0, 1.5] / ... % Duration of the stimuli, adjusted to match the refresh rate
    ifi) * ifi;            
trialDuration = 2.0;                   % Total duration of each trial
numTrials     = 10;                    % Number of trials for each condition
totalTrials   = numTrials * ...        % Total number of trials
    length(shapes) * length(durations);

% Generate the trial list
trials = repmat...                          % Create numbers from 1 to 4 encoding each condition and repeat numTrials times                  
    ((1:length(shapes) * length(durations))', numTrials, 1);
trials = trials(randperm(length(trials)));  % Randomize the trials order

% Define elements sizes:
squareSize      = 300;    % Size of the square
circleDiameter  = 300;    % Diameter of the circle

% Photodiode parameters
photodiodeDur = 0.032;    % Duration for which the photodiode should remain on
photoSquareSize = 20;     % Size of the photodiode square

% Text:
welcomeText  = 'Welcome to the dummy experiment!';
endText      = 'Congrats, you have completed the dummy experiment! The data were saved successfully';
continueText = 'Press space to continue';

%% Prepare the log file:
stimShapes    = cell(totalTrials, 1);  % Log the planned shape of each stimulus
stimDurations = zeros(totalTrials, 1); % Log the planned duration of each stimulus
stimOnset     = zeros(totalTrials, 1); % Log the PTB stimulus onset
stimOffset    = zeros(totalTrials, 1); % Log the PTB stimulus offset
rts           = zeros(totalTrials, 1); % Log the reaction times
itis          = zeros(totalTrials, 1); % Log the inter trial intervals

%% Trials loop:

% Show an intruction screen:
[~, ~, ~]     = DrawFormattedText(window, welcomeText, 'center', 'center', white);
[~, ~, ~] = DrawFormattedText(window, continueText, 'center', screenYpixels - 50, white);
Screen('Flip', window);
% Wait for space key press
KbName('UnifyKeyNames');
spaceKey = KbName('space');
while 1
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && keyCode(spaceKey)
        break;
    end
end
WaitSecs(rand + 0.5);  % Add a random jitter

% Loop through each unique trial
for i = 1:totalTrials

    % Determine current trial parameters
    [shapeIndex, durationIndex] = ind2sub([length(shapes), ... % Use indexing to get the trials parameters
        length(durations)], trials(i));
    shape                       = shapes{shapeIndex};          % Shape of the stimulus to present in this trial
    duration                    = durations(durationIndex);    % Duration for which to present the stimulus
    iti                         = round((rand + 0.5) / ...     % Random inter-trial interval between 0.5 and 1.5, adjusted to match the refresh rate
        ifi) * ifi;                  
    % Initiate time counters:
    elapsedTime      = 0; % Initiate time counter
    elapsedTimePhoto = 0; % Time counter for the photodiode
    rt               = 0;

    % Show the current stimulus:
    [onsetTime] = showStimulus(shape, window);   % Show the stimulus

    % Set the flags
    photoFlag   = 1;         % The photodiode is turned on when the stimulus appears                                 
    photoOnset  = onsetTime; % At the same time as the stimulus
    stimFlag    = 1;         % mark that the stimulus is on
    hasResp     = 0;         % Check whether a key was pressed
    escapeKey   = 0;

    %% Time loop:
    while elapsedTime < trialDuration + iti
        
        % -----------------------------------------------------------------
        if stimFlag && elapsedTime > duration
            % If the stimulus is still present but the elapsed time since
            % onset exceeds the stimulus duration, clear the screen
            Screen('FillRect', window, grey); % Draw gray background
            drawPhotodiode('white', window);  % Draw photodiode to mark the event
            [~, offsetTime, ~, ~, ~] = Screen('Flip', window); % Flip the screen
            photoOnset               = offsetTime;             % Reset the photodiode time counter
            photoFlag                = 1;                      % Mark photodiode as on
            stimFlag                 = 0;                      % Mark that the stimulus was removed
        end
        
        % -----------------------------------------------------------------
        if photoFlag && elapsedTimePhoto > photodiodeDur
            % If the photodiode is still on and has been on for more than
            % it should, draw it off
            drawPhotodiode('black', window);  % Set the photodiode square to black
            [~, offsetTime, ~, ~, ~] = Screen('Flip', ... % Flip the screen
                window, [], 1);
            photoFlag                = 0;                 % Reset the photodiode flag
            elapsedTimePhoto         = 0;                 % Reset the photodiode time counter
        end
        
        % -----------------------------------------------------------------
        if ~hasResp
            % If no response was registered in this trial, check if a key
            % is pressed
            [KeyIsDown, respTime, keyCode] = KbCheck();
            if KeyIsDown
                % If a key was pressed
                hasResp = 1;                     % Set response flag
                rt      = respTime - onsetTime;  % Calculate reaction time
           
                if ismember('ESCAPE', KbName(keyCode))
                    % Abort the experiment if the escape key was pressed
                    escapeKey = true;
                    break;
                end
            end
        end

        % -----------------------------------------------------------------
        elapsedTime = (GetSecs) - onsetTime;
        if photoFlag
            elapsedTimePhoto = GetSecs - photoOnset;
        end

    end

    % Abort the experiment if the escape key was pressed
    if escapeKey
        break;
    end

    %% Log the past trial events
    stimShapes{i}    = shape;       % Shape of the previous trial
    stimDurations(i) = duration;    % Duration of the previous trial
    stimOnset(i)     = onsetTime;   % Onset of the stimulus in the previous trial
    stimOffset(i)    = offsetTime;  % Offset of the stimulus in the previous trial
    rts(i)           = rt;          % Reaction time in the previous trial
    itis(i)        = iti;           % Duration of the inter-trial interval

end

% Combine all trials info into a log file:
logFile = array2table([(1:totalTrials)', stimDurations, ...
    stimOnset, stimOffset, rts, itis], ...
    "VariableNames", {'trialNumber', 'duration','stimOnset', ...
    'stimOffset', 'rt', 'iti'});
% Add the stimShapes cell array as a table column
logFile.shape = stimShapes;

% Save to a csv:
writetable(logFile, 'sub-101_ses-1_task-Dummy_events.csv')

% Show an intruction screen:
[~, ~, textBounds]     = DrawFormattedText(window, endText, 'center', 'center', white);
[~, ~, continueBounds] = DrawFormattedText(window, continueText, 'center', screenYpixels - 50, white);
Screen('Flip', window);
% Wait for space key press
KbName('UnifyKeyNames');
spaceKey = KbName('space');
while 1
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && keyCode(spaceKey)
        break;
    end
end

% Clear the screen
sca;


