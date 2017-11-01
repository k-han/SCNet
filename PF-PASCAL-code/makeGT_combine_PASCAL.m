%by khan

function makeGT_combine_PASCAL()
global conf;

mean_std = 0;
visualize_output = 0;

linewidth = 2;
images = [];
proposals = [];
proposals_GT = [];
idx_for_active_opA = [];
IoU2GT = [];
set = [];
bbox = [];
part_x = [];
part_y = [];
% combine all images and proposals together
for ci = 1:numel(conf.class)
    load(fullfile(conf.matchGTDir,conf.class{ci},[ conf.class{ci} '.mat' ]), 'data');
    pair_num = size(data.images,1);
    train_num = ceil(pair_num * 7/13);
    val_num = ceil((pair_num - train_num)*0.5);
    test_num = pair_num - train_num - val_num;
    cur_set = [ones(train_num,1); ones(val_num,1)*2;   ones(test_num,1)*3];
    set = [set; cur_set];
    images = [images; data.images];
    proposals = [proposals; data.proposals];
    proposals_GT = [proposals_GT; data.proposals_GT];
    idx_for_active_opA = [idx_for_active_opA; data.idx_for_active_opA];
    IoU2GT = [IoU2GT; data.IoU2GT];
    bbox = [bbox; data.bbox];
    part_x = [part_x; data.part_x];
    part_y = [part_y; data.part_y];
end

data.images = images;
data.proposals = proposals;
data.set = set;
data.proposals_GT = proposals_GT;
data.idx_for_active_opA = idx_for_active_opA;
data.IoU2GT = IoU2GT;
data.bbox = bbox;
data.part_x = part_x;
data.part_y = part_y;
% get mean and std for training data
if mean_std
    train_idx = find(data.set == 1);
    images_mat = cat(4,data.images{train_idx,1},data.images{train_idx,2});
    image_mean = mean(images_mat, 4);
    image_std = std(single(images_mat),1,4);

    data.image_mean = image_mean;
    data.image_std = image_std;
end

if visualize_output
    for i = 1:size(images, 1)
        for j = 1:size(proposals{i,1},1);
        figure(1)
        imshow(images{i,1});
        hold on;
        drawboxline(proposals{i,1}(j,:)', 'LineWidth', linewidth, 'color', 'g');
        figure(2)
        imshow(images{i,2});
        hold on;
        drawboxline(proposals{i,2}(j,:)', 'LineWidth', linewidth, 'color', 'g');
        pause;
        end
    end
end

save(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data', '-v7.3');
end