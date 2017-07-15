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
img_A = imdb.data.images{batch,1} ;
img_B = imdb.data.images{batch,2} ;
op_A.coords = imdb.data.proposals{batch,1};
op_B.coords = imdb.data.proposals{batch,2};
proposals_A = imdb.data.proposals{batch,1};
proposals_B = imdb.data.proposals{batch,2};
proposals_A = [ones(size(proposals_A,1), 1).*1 proposals_A];
proposals_B = [ones(size(proposals_B,1), 1).*2 proposals_B];
idx_for_active_opA = imdb.data.idx_for_active_opA{batch,1};
IoU2GT = imdb.data.IoU2GT{batch,1};
%extract features
feat_A = extract_segfeat_hog(img_A,op_A); % HOG
feat_B = extract_segfeat_hog(img_B,op_B); % HOG

tic;
fprintf(' - %s matching... ', 'HOG-SCNet-AG');

A_out = feat_A.hist*feat_B.hist';
G_out = vl_nnhoughvote(A_out, proposals_A', proposals_B', img_A, img_B, []);

confidenceMap = A_out.*G_out;

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


AG.PCR = PCR;
AG.mIoU = mIoU;

figure 
plot(tbinv_PCR, PCR);
xlabel('IoU threshold');
ylabel('PCR');
axis([0, 1, 0, 1]);

save('HOG-SCNet-AG-eva.mat', 'AG');