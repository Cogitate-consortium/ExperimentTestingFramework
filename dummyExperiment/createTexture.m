function [texture] = createTexture(fl, w)
[img, ~, alpha] = imread(fl);
[~, ~, t3, ~] = size(img);
if t3 == 1 % this means its a monochrome image
    img(:, :, 2) = alpha;
else
    img(:, :, 4) = alpha;
end
texture = Screen('MakeTexture', w, img);
end
