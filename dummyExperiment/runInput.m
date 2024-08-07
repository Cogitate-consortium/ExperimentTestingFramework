function [subjectName, isTestMode, distanceFromScreen] = runInput()
    % Initialize output variables
    subjectName = '';
    isTestMode = false;
    distanceFromScreen = 0;

    % Create a figure for the input dialog
    fig = figure('Position', [100, 100, 350, 180], 'MenuBar', 'none', 'Name', 'Experiment Setup', 'NumberTitle', 'off', 'Resize', 'off');

    % Create a uicontrol for the subject name input
    uicontrol('Style', 'text', 'Position', [10, 130, 330, 20], 'String', 'Enter the subject name (SX123):');
    subjectNameInput = uicontrol('Style', 'edit', 'Position', [10, 110, 330, 20], 'String', '');

    % Create a uicontrol for the distance from the screen input
    uicontrol('Style', 'text', 'Position', [10, 80, 330, 20], 'String', 'Enter the distance from the screen (in cm):');
    distanceInput = uicontrol('Style', 'edit', 'Position', [10, 60, 330, 20], 'String', '');

    % Create a checkbox for test mode
    testModeCheckbox = uicontrol('Style', 'checkbox', 'Position', [10, 30, 330, 20], 'String', 'Test Mode');

    % Create OK and Cancel buttons
    uicontrol('Style', 'pushbutton', 'Position', [70, 10, 60, 20], 'String', 'OK', 'Callback', @okButtonCallback);
    uicontrol('Style', 'pushbutton', 'Position', [140, 10, 60, 20], 'String', 'Cancel', 'Callback', @cancelButtonCallback);

    % Make the dialog modal
    set(fig, 'WindowStyle', 'modal');
    
    % Wait for the user to close the dialog
    uiwait(fig);
    
    % Callback function for the OK button
    function okButtonCallback(~, ~)
        subjectName = get(subjectNameInput, 'String');
        isTestMode = get(testModeCheckbox, 'Value');
        distanceFromScreen = str2double(get(distanceInput, 'String'));
        uiresume(fig);
        delete(fig);
    end

    % Callback function for the Cancel button
    function cancelButtonCallback(~, ~)
        uiresume(fig);
        delete(fig);
    end
end
