clear;
close all;

%% =========================================================================
% PRELIMINARY
cd ..
global conf;

conf.baseDir = pwd();
addpath(fullfile(conf.baseDir, 'commonFunctions'));

conf.db_name = 'PF-PASCAL';

% class name
conf.classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'};

% algorithm
conf.algorithm = {'DeepFlow', 'GMK', 'SIFTflow', 'DSP', 'Proposal Flow'};

% path to warping results
conf.algDir = cell(length(conf.algorithm),1);
for i=1:length(conf.algorithm)
    if strcmp(conf.algorithm{i}, 'Proposal Flow')
        conf.algDir{i} = fullfile(conf.baseDir,'_BM-PF-PASCAL','RP','RP-1000','dense-flow');
    else
        conf.algDir{i} = fullfile(conf.baseDir,'_comparision','eva_PASCAL', sprintf('%s', conf.algorithm{i}));
    end
end


% path to the images and webpages
conf.webpDir = fullfile(conf.baseDir,'_comparision','vis_PASCAL');
if isempty(dir(conf.webpDir))
    mkdir(conf.webpDir);
end
conf.webp_cls_path = fullfile(conf.webpDir,'html');
if isempty(dir(conf.webp_cls_path))
    mkdir(conf.webp_cls_path);
end

% path to dataset
conf.datasetDir = fullfile(conf.baseDir, 'PF-dataset-PASCAL');
% path to source and target images
conf.imageDir = fullfile(conf.datasetDir, 'JPEGImages');

% visualization setting
conf.iwidth_img  = 250;				% image width in webpages

%% =========================================================================
% INDEX.HTML

fprintf('Writing html documents\n');

fout = fopen(fullfile(conf.webpDir, 'index.html'), 'w');
fprintf(fout, ['<html><head><title>Results on ', conf.db_name, '</title></head>\n']);
fprintf(fout, ['<h1>Visualization of dense flow field on the ', conf.db_name, 'dataset.</h1>\n']);
fprintf(fout, '<h2># Object classes</h2>\n');

cnt=0;
% table start
fprintf(fout, '<table border="1">\n');
fprintf(fout, '<tr>\n');
for cidx = 1 : numel(conf.classes)
    cls_name = conf.classes{cidx};
    
    if mod(cnt,5)==0
        fprintf(fout, '<tr>\n');
    end
    
    fprintf(fout, '<td style="width: 100px;">');
    
    % results per class: hyperlinks
    fprintf(fout, ['<font size=5><a href="', fullfile(conf.webp_cls_path,cls_name), '.html">', cls_name, '</a></font>\n']);
    fprintf(fout, '</td>');
    
    cnt=cnt+1;
end
% table end
fprintf(fout, '</tr>\n');
fprintf(fout, '</table>\n');
fprintf(fout, '</html>');
fclose(fout);

%% =========================================================================
% HTML PAGE PER CLASS

% ext matching pairs ---------------------------------------------------
load(fullfile(conf.datasetDir, 'parsePascalVOC.mat'));

annotation{1} = 'Source image';
annotation{2} = 'Target image';
for al=1:length(conf.algorithm)
    annotation{al+2}= conf.algorithm{al};
end

for cidx = 1 : numel(conf.classes)
    cls_name = conf.classes{cidx};
    
    % set matching pair
    classInd = pascalClassIndex(cls_name);
    pair = PascalVOC.pair{classInd};
    
    % webpage content
    fout = fopen(fullfile(conf.webp_cls_path, [cls_name, '.html']), 'w');
    fprintf(fout, ['<html><head><title>Results on ', conf.db_name, '</title></head>\n']);
    fprintf(fout, ['<h1><a href="', fullfile(conf.webpDir, 'index.html') ,'">Visualization of dense flow field</a> / ', cls_name, '</h1>\n']);
    fprintf(fout, ['<h2># Visual comparisons (', num2str(length(pair)), ' pairs) </h2>\n']);
    
    % table start
    fprintf(fout, '<table border="0", cellspacing="1">\n');
    
    for i=1:length(pair)
        
        fnames = cell(2+length(conf.algorithm),1);
        
        % load the sorce and target image files and results.
        imgA_name = cell2mat(strcat(pair(i,1)));
        fnames{2} = fullfile(conf.imageDir,[imgA_name '.jpg']);
        
        imgB_name = cell2mat(strcat(pair(i,2)));
        fnames{1} = fullfile(conf.imageDir,[imgB_name '.jpg']);
        
        for al=1:length(conf.algorithm)
            if strcmp(conf.algorithm{al}, 'Proposal Flow')
                fnames{al+2} = fullfile(conf.algDir{al}, cls_name, 'HOG', [ imgA_name '-' imgB_name '_RP_HOG_LOM.jpg']);
            else
                fnames{al+2} = fullfile(conf.algDir{al}, cls_name, [ imgA_name '-' imgB_name '_' conf.algorithm{al} '.jpg']);
            end
        end
        
        cnt=0;
        fprintf(fout, '<tr>\n');
        for fi=1:length(fnames)
            if cnt == 2
                fprintf(fout, '</tr>\n');
                fprintf(fout, '<tr>\n');
            end
            
            fprintf(fout, '<td valign=bottom>\n');
            fprintf(fout, ['<img src="', fnames{fi}, '" width="', num2str(conf.iwidth_img), '" border="1"></a>\n']);
            fprintf(fout, '<br>\n');
            fprintf(fout, ['<font size=3>', annotation{fi}, '</font>\n']);
            
            cnt = cnt + 1;
        end
        fprintf(fout, '</tr>\n');
        fprintf(fout, '<tr><td><br/><br/><br/></td></tr>\n');
    end
    % table end
    fprintf(fout, '</table>\n');
    fprintf(fout, '</html>');
    fclose(fout);
end






