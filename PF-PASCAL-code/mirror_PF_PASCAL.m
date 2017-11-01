function mirror_PF_PASCAL()
global conf;
visualize_output = 0;
linewidth = 2;
load(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data');
images = data.images;
proposals = data.proposals;
proposals_GT = data.proposals_GT;
idx_for_active_opA = data.idx_for_active_opA;
IoU2GT = data.IoU2GT;
set = data.set;
bbox = data.bbox;
part_x = data.part_x;
part_y = data.part_y;

train_idx = find(data.set == 1);

for i = 1:numel(train_idx)
% flip images
images{train_idx(i), 1} = flip(images{train_idx(i), 1}, 2);
images{train_idx(i), 2} = flip(images{train_idx(i), 2}, 2);

% flip proposals
proposals{train_idx(i), 1}(:,1) = 224 - proposals{train_idx(i), 1}(:,1);
proposals{train_idx(i), 1}(:,3) = 224 - proposals{train_idx(i), 1}(:,3);
proposals{train_idx(i), 2}(:,1) = 224 - proposals{train_idx(i), 2}(:,1);
proposals{train_idx(i), 2}(:,3) = 224 - proposals{train_idx(i), 2}(:,3);

% flip proposals_GT
proposals_GT{train_idx(i), 1}(:,1) = 224 - proposals_GT{train_idx(i), 1}(:,1);
proposals_GT{train_idx(i), 1}(:,3) = 224 - proposals_GT{train_idx(i), 1}(:,3);
proposals_GT{train_idx(i), 2}(:,1) = 224 - proposals_GT{train_idx(i), 2}(:,1);
proposals_GT{train_idx(i), 2}(:,3) = 224 - proposals_GT{train_idx(i), 2}(:,3);

%flip bbox
bbox{train_idx(i), 1}(:,1) = 224 - bbox{train_idx(i), 1}(:,1);
bbox{train_idx(i), 1}(:,3) = 224 - bbox{train_idx(i), 1}(:,3);
bbox{train_idx(i), 2}(:,1) = 224 - bbox{train_idx(i), 2}(:,1);
bbox{train_idx(i), 2}(:,3) = 224 - bbox{train_idx(i), 2}(:,3);


% flip part_x, party
part_x{train_idx(i), 1}(:,1) = 224 - part_x{train_idx(i), 1}(:,1);
part_x{train_idx(i), 1}(:,3) = 224 - part_x{train_idx(i), 1}(:,3);
part_x{train_idx(i), 2}(:,1) = 224 - part_x{train_idx(i), 2}(:,1);
part_x{train_idx(i), 2}(:,3) = 224 - part_x{train_idx(i), 2}(:,3);

part_y{train_idx(i), 1}(:,1) = 224 - part_y{train_idx(i), 1}(:,1);
part_y{train_idx(i), 1}(:,3) = 224 - part_y{train_idx(i), 1}(:,3);
part_y{train_idx(i), 2}(:,1) = 224 - part_y{train_idx(i), 2}(:,1);
part_y{train_idx(i), 2}(:,3) = 224 - part_y{train_idx(i), 2}(:,3);

end

new_set = ones(numel(train_idx),1);

%update images + proposals + propsosals_GT + idx_for_active_opA + IoU2GT
%+set
data.set = [data.set; new_set];
data.images = [data.images; images(train_idx,:)];
data.proposals = [data.proposals; proposals(train_idx,:)];
data.proposals_GT = [data.proposals_GT; proposals_GT(train_idx,:)];
data.idx_for_active_opA = [data.idx_for_active_opA; idx_for_active_opA(train_idx,:)];
data.IoU2GT = [data.IoU2GT; IoU2GT(train_idx,:)];
data.bbox = [data.bbox; bbox(train_idx,:)];
data.part_x = [data.part_x; part_x(train_idx,:)];
data.part_y = [data.part_y; part_y(train_idx,:)];
% update mean and std for training data
train_idx = find(data.set == 1);
images_mat = cat(4,data.images{train_idx,1},data.images{train_idx,2});
image_mean = mean(images_mat, 4);
image_std = std(single(images_mat),1,4);
data.image_mean = image_mean;
data.image_std = image_std;

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

save(fullfile(conf.matchGTDir,'PF-PASCAL.mat'), 'data','-v7.3');
