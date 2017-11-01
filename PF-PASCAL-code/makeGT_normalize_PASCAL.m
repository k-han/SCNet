%by khan

function makeGT_normalize_PASCAL()
global conf;

resize = 1;
mean_std = 1;
visualize_output = 0;

load(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data');

linewidth = 2;
images = data.images;
proposals = data.proposals;
proposals_GT = data.proposals_GT;
set = data.set;
bbox = data.bbox;
part_x = data.part_x;
part_y = data.part_y;
% idx_for_active_opA = data.idx_for_active_opA;
% IoU2GT = data.IoU2GT;
% set = data.set;




% resize images and proposals
if resize
new_dim = [224,224];
    for i = 1:size(images, 1)
        img1_size = size(images{i,1});
        ratio_1 = new_dim./img1_size(1:2);

        img2_size = size(images{i,2});
        ratio_2 = new_dim./img2_size(1:2);

        images{i,1} = imresize(images{i,1}, new_dim, 'bicubic','antialiasing', false) ;
        images{i,2} = imresize(images{i,2}, new_dim, 'bicubic','antialiasing', false) ;

        proposals{i,1} = bsxfun(@times,proposals{i,1}, [ratio_1(2) ratio_1(1) ratio_1(2) ratio_1(1)]);
        proposals{i,2} = bsxfun(@times,proposals{i,2}, [ratio_2(2) ratio_2(1) ratio_2(2) ratio_2(1)]);
        proposals_GT{i,1} = bsxfun(@times,proposals_GT{i,1}, [ratio_1(2) ratio_1(1) ratio_1(2) ratio_1(1)]);
        proposals_GT{i,2} = bsxfun(@times,proposals_GT{i,2}, [ratio_2(2) ratio_2(1) ratio_2(2) ratio_2(1)]);
        bbox{i, 1} = bsxfun(@times,bbox{i,1}, [ratio_1(2) ratio_1(1) ratio_1(2) ratio_1(1)]);
        bbox{i, 2} = bsxfun(@times,bbox{i,2}, [ratio_1(2) ratio_1(1) ratio_1(2) ratio_1(1)]);
        part_x{i,1} = part_x{i,1}.*ratio_1(2);
        part_x{i,2} = part_x{i,2}.*ratio_2(2);
        part_y{i,1} = part_y{i,1}.*ratio_1(1);
        part_y{i,2} = part_y{i,2}.*ratio_2(1);
    end
end

% get mean and std for training data
if mean_std
    train_idx = find(set == 1);
    images_mat = cat(4,images{train_idx,1},images{train_idx,2});
    image_mean = mean(images_mat, 4);
    image_std = std(single(images_mat),1,4);

    data.image_mean = image_mean;
    data.image_std = image_std;
end
data.images = images;
data.proposals = proposals;
data.proposals_GT = proposals_GT;
data.bbox = bbox;
data.part_x = part_x;
data.part_y = part_y;
if visualize_output
    for i = 1:size(images, 1)
        for j = 1:size(proposals_GT{i,1},1);
        figure(1)
        imshow(images{i,1});
        hold on;
        drawboxline(proposals_GT{i,1}(j,:)', 'LineWidth', linewidth, 'color', 'g');
        figure(2)
        imshow(images{i,2});
        hold on;
        drawboxline(proposals_GT{i,2}(j,:)', 'LineWidth', linewidth, 'color', 'g');
        pause;
        end
    end
end

save(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data', '-v7.3');
end