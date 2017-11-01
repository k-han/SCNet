% script for making GT data based on the annotations and the proposals
% written by Bumsub Ham, Inria - WILLOW / ENS, Paris, France

function cal_ratio_AP_PASCAL(proposal, num_op)


global conf;

% dataset
conf.datasetDir = '../PF-dataset-PASCAL/';

% keypoint
conf.annoKPDir = [conf.datasetDir 'Annotations/'];

% object classes
conf.class = dir(conf.annoKPDir) ;
conf.class = {conf.class(4:end).name};

% benchmark dir
conf.benchmarkDir = fullfile('../_BM-PF-PASCAL', sprintf('%s-%s', proposal, num2str(num_op)));

sum=0;
for ci = 1:numel(conf.class)
    fprintf('processing %s...\n',conf.class{ci});
    % load the indices of valid object proposals and corresponding object bounding box
    load(fullfile(conf.benchmarkDir,sprintf('AP_%s.mat',conf.class{ci})), 'AP');
    ratio = mean(AP.num_op_active./AP.num_op_all);
    sum=sum + ratio;
end
avg = sum/numel(conf.class);
fprintf('\n%s-%s: ratio = %.4f\n', proposal, num2str(num_op), avg);
end
