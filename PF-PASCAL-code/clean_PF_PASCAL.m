function clean_PF_PASCAL()
% remove outlier data without GT pairs.

global conf;

load(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data');

mean_std = 0;

idx = find(data.set == 1 | data.set == 2 | data.set == 3);


outlier_idx = [];

for i = 1:numel(idx)
if (size(data.proposals_GT{idx(i), 1}, 1)< 1)
    outlier_idx = [outlier_idx; idx(i)];
end
end

idx = setdiff(idx, outlier_idx);

%update images + proposals + set
data.set = data.set(idx);
data.images = data.images(idx,:);
data.proposals = data.proposals(idx, :);
data.proposals_GT = data.proposals_GT(idx, :);
data.idx_for_active_opA = data.idx_for_active_opA(idx, :);
data.IoU2GT = data.IoU2GT(idx, :);
data.bbox = data.bbox(idx, :);
data.part_x = data.part_x(idx, :);
data.part_y = data.part_y(idx, :);
%update mean, std
if mean_std
    images_mat = cat(4,data.images{:,1},data.images{:,2});
    image_mean = mean(images_mat, 4);
    image_std = std(single(images_mat),1,4);
    data.image_mean = image_mean;
    data.image_std = image_std;
end

save(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data', '-v7.3');
