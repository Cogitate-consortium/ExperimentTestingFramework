

data_file = "C:\Users\alexander.lepauvre\Documents\PhD\Experimental_testing_framework\data\simulation\bids\derivatives\erp\population\figure\unknown\jitter_summary.csv";
data = readtable(data_file);

% Get the x and y:
x = unique(data.jitterDuration);
y = unique(data.jitterProportion);
[X,Y] = meshgrid(x, y);
thresh = zeros([length(y), length(x)]) + 1.96;
c = zeros([length(y), length(x), 3]) + 0.5;
s = surf(X,Y,thresh, c);
s.EdgeColor = 'none';
hold on
% Extract effect sizes:
fsizes = unique(data.fsize);
for ind = 1:length(fsizes)
    % Get the data from this effect size:
    fsize_tstat = data(data.fsize == fsizes(ind), :);
    tstat = fsize_tstat.tStatistic;
    tstat = reshape(tstat, [length(y), length(x)]);
    c = zeros([length(y), length(x)]) + fsizes(ind);
    s = surf(X,Y,tstat, c,'FaceAlpha',0.5);
    s.EdgeColor = 'none';
    hold on
    colormap(hot)
end
 