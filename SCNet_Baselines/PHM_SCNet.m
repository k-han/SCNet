useGPU = 1;
gpuDevice([4]);
visulize_bbox = 0;
run(fullfile('../utils', 'vlfeat-0.9.20','toolbox', 'vl_setup.m')) ;
addpath(genpath('../utils/'));
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

modelPath = '../data/trained_models/PASCAL-RP/SCNet-A.mat';
load(modelPath);
net = dagnn.DagNN.loadobj(net);

addpath(genpath('../utils/feature/'))
addpath(genpath('../utils/commonFunctions/'))
%load data
imdbPath = '../data/PF-PASCAL-RP-1000.mat';
imdb = load(imdbPath);

val_idx = find(imdb.data.set==3);

tbinv_PCR = linspace(0,1,101);
tbinv_mIoU = linspace(1,100,100);
    
pcr = zeros(numel(val_idx), numel(tbinv_PCR));
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
net.eval(inputs);

feat_b1 = net.vars(net.getVarIndex('b1_L2NM')).value;
feat_b1 = reshape(feat_b1, size(feat_b1, 3), size(feat_b1, 4));
feat_b1 = feat_b1';
feat_b2 = net.vars(net.getVarIndex('b2_L2NM')).value;
feat_b2 = reshape(feat_b2, size(feat_b2, 3), size(feat_b2, 4));
feat_b2 = feat_b2';

if useGPU
    feat_b1 = gather(feat_b1);
    feat_b2 = gather(feat_b2);
end

feat_A.img = imdb.data.images{batch,1};
feat_A.boxes = imdb.data.proposals{batch,1};
feat_A.hist = feat_b1;
feat_B.img = imdb.data.images{batch,2};
feat_B.boxes = imdb.data.proposals{batch,2};
feat_B.hist = feat_b2;

tic;
fprintf(' - %s matching... ', 'PHM');

% options for matching
opt.bDeleteByAspect = true;
opt.bDensityAware = false;
opt.bSimVote = false;
opt.bVoteExp = true;
opt.feature = 'LPF';

viewA = vl_getView2(feat_A.boxes);
viewA.desc = feat_b1';
viewA.img = feat_A.img;
viewB = vl_getView2(feat_B.boxes);
viewB.desc = feat_b2';
viewB.img = feat_B.img;

confidenceMap = PHM( viewA, viewB, opt );

% PCR
[ confidenceMax, idx ] = max(confidenceMap(idx_for_active_opA,:),[],2);
fprintf('   took %.2f secs\n',toc);


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

end

PCR = mean(pcr,1);
mIoU = zeros(1, numel(tbinv_mIoU));
for j = 1:numel(tbinv_mIoU)
    valid_idx = find(miou_all(:,1:tbinv_mIoU(j)) > -1);
    mIoU(j) = sum(miou_all(valid_idx))./numel(valid_idx);
end

PHM_SCNet.PCR = PCR;
PHM_SCNet.mIoU = mIoU;

figure 
plot(tbinv_PCR, PCR);
xlabel('IoU threshold');
ylabel('PCR');
axis([0, 1, 0, 1]);

save('PHM_SCNet-AG+fixA-1000.mat', 'PHM_SCNet');