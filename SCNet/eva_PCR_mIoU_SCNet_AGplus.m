function [ PCR ] = eva_PCR_mIoU_SCNet_AGplus()
useGPU = 0;
if useGPU
 gpuDevice([1]);
end
visulize_bbox = 0;
run(fullfile('../utils', 'vlfeat-0.9.20','toolbox', 'vl_setup.m')) ;
addpath(genpath('../utils/'));
run(fullfile(fileparts(mfilename('fullpath')),...
  '../..', 'matlab', 'vl_setupnn.m'));

%revise modelPath and imdbPath accordingly
modelPath = '../data/trained_models/PASCAL-RP/SCNet-AG+.mat';
imdbPath = fullfile('../data', 'PF-PASCAL-RP-1000.mat');

load(modelPath);
net = dagnn.DagNN.loadobj(net);
imdb = load(imdbPath);

val_idx = find(imdb.data.set==3);
tbinv_PCR = linspace(0,1,101);
tbinv_mIoU = linspace(1,100,100);
pcr = zeros(numel(val_idx), numel(tbinv_PCR));
time_cost = zeros(numel(val_idx),1);

max_boxNumer = 1000;
miou_all = ones(numel(val_idx), max_boxNumer).*(-1);

for i = 1:numel(val_idx)
batch = val_idx(i);
image_mean = imdb.data.image_mean;
image_std = imdb.data.image_std;
images_A = imdb.data.images{batch,1} ;
images_A = (single(images_A) - image_mean)./image_std;
images_B = imdb.data.images{batch,2} ;
images_B = (single(images_B) - image_mean)./image_std;
proposals_A = imdb.data.proposals{batch,1};
proposals_B = imdb.data.proposals{batch,2};
proposals_A = [ones(size(proposals_A,1), 1).*1 proposals_A];
proposals_B = [ones(size(proposals_B,1), 1).*2 proposals_B];
idx_for_active_opA = imdb.data.idx_for_active_opA{batch,1};
IoU2GT = imdb.data.IoU2GT{batch,1};
if useGPU
  inputs = {'b1_input', gpuArray(im2single(images_A)), 'b2_input', gpuArray(im2single(images_B)), 'b1_rois', gpuArray(single(proposals_A')), 'b2_rois', gpuArray(single(proposals_B')), 'idx_for_active_opA', gpuArray(single(idx_for_active_opA)), 'IoU2GT', gpuArray(single(IoU2GT))} ;
  net.move('gpu');
else
  inputs = {'b1_input', im2single(images_A), 'b2_input', im2single(images_B), 'b1_rois', single(proposals_A'), 'b2_rois', single(proposals_B'), 'idx_for_active_opA', single(idx_for_active_opA), 'IoU2GT', single(IoU2GT)} ;
end
net.conserveMemory = false ;

tic
net.eval(inputs);
time_cost(i) = toc;

confidenceMap = net.vars(net.getVarIndex('AG_out')).value;

%matching confidence
[ confidenceMax, idx ] = max(confidenceMap(idx_for_active_opA,:),[],2);

%eval IOU
i_IoU = zeros(numel(idx_for_active_opA), 1);
for ii = 1:numel(idx_for_active_opA)
i_IoU(ii) = IoU2GT(ii,idx(ii));
end

if numel(idx_for_active_opA)>0
    k = 0;
    for t = tbinv_PCR
        k = k+1;
        pcr(i,k) = numel(find(i_IoU < t))/numel(idx_for_active_opA);
    end 
end

%mIoU@k
IoU2GT = 1 - IoU2GT;
[~, idx_st ] = sort(confidenceMax, 'descend');

for k = 1:numel(idx_for_active_opA)
miou_all(i, k) = IoU2GT(idx_st(k), idx(idx_st(k)));
end

linewidth = 1;
if visulize_bbox
    proposals_l = RectLTRB2LTWH(proposals_A(:, 2:end));
    proposals_r = RectLTRB2LTWH(proposals_B(:, 2:end));
    for k = 1:numel(idx_for_active_opA);
        figure(1)
        imshow(images_A);
        hold on;
        rectangle('Position', proposals_l(k,:), 'LineWidth', linewidth, 'EdgeColor', 'g');
        figure(2)
        imshow(images_B);
        hold on;
        rectangle('Position', proposals_r(idx(k),:), 'LineWidth', linewidth, 'EdgeColor', 'r');
        rectangle('Position', proposals_r(k,:), 'LineWidth', linewidth, 'EdgeColor', 'g');
        pause
    end
end

end

PCR = mean(pcr,1);
mIoU = zeros(1, numel(tbinv_mIoU));
for j = 1:numel(tbinv_mIoU)
    valid_idx = find(miou_all(:,1:tbinv_mIoU(j)) > -1);
    mIoU(j) = sum(miou_all(valid_idx))./numel(valid_idx);
end

SCNet_AGplus.PCR = PCR;
SCNet_AGplus.mIoU = mIoU;
SCNet_AGplus.time_cost = time_cost;
figure 
plot(tbinv_PCR, PCR);
xlabel('IoU threshold');
ylabel('PCR');
axis([0, 1, 0, 1]);
save('eva_SCNet_AGplus.mat', 'SCNet_AGplus');
end