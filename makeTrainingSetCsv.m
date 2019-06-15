function makeTrainingSetCsv

% This function exports a training set CSV from labeles
% Created by Rick Warren
% https://github.com/rwarren2163/locomotionAnalysis

folder = fullfile(getenv('OBSDATADIR'), 'tracking', 'trainingData', 'pawContact');
files = dir(fullfile(folder, '*.mat'));

allSessionsData = cell(1,length(files));

for i = 1:length(files)
    
    load(fullfile(folder, files(i).name), 'classNames', 'classes', 'session')
    skipBin = contains(classNames, 'skip');
    analyzedBins = all(~isnan(classes),1) & classes(skipBin,:)~=1;
    
    sessionData = cell2table(cell(sum(analyzedBins),3), 'VariableNames', {'session', 'frameNum', 'class'});
    sessionData.session = repmat({session}, height(sessionData), 1);
    sessionData.frameNum = find(analyzedBins)';
    sessionData.class = classNames(vec2ind(classes(:,analyzedBins)))';
    
    allSessionsData{i} = sessionData;
    
end

allSessionsData = vertcat(allSessionsData{:}); % combinee into one big ass table
writetable(allSessionsData, fullfile(folder, 'contactData.csv'));
