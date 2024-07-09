function [stimWidthPx, stimHeightPx] = convertDVAToPixels(stimWidthDVA, stimHeightDVA, screenWidthPx, screenWidthcm, distanceFromScreen)

    % Convert screen dimensions from cm to mm
    screenWidthmm = screenWidthcm * 10;

    % Calculate the pixel density (pixels per mm)
    pxPerMmX = screenWidthPx / screenWidthmm;

    % Calculate the size of the stimulus in mm using the formula:
    % size_mm = 2 * distance_mm * tan(deg/2 * pi/180)
    stimWidthMm = 2 * (distanceFromScreen * 10) * tan(stimWidthDVA / 2 * pi / 180);
    stimHeightMm = 2 * (distanceFromScreen * 10) * tan(stimHeightDVA / 2 * pi / 180);

    % Convert stimulus size from mm to pixels
    stimWidthPx = round(stimWidthMm * pxPerMmX);
    stimHeightPx = round(stimHeightMm * pxPerMmX);
end