function prepKP_PASCAL()
% transform raw keypoint annotations into my data
% by Bumsub Ham, Inria - WILLOW / ENS

global conf;

classes = conf.class;

for ci = 1:length(classes)
    
    anno = dir(fullfile(conf.annoKPDir, classes{ci}, '*.mat'));
    
    fprintf('processing %s...',classes{ci});
    
    image_dir = cell(1,1); image_name = cell(1,1); image2anno = cell(1,1); anno2image = [];
    part_list = cell(1,1); 
    
    load(fullfile(conf.annoKPDir, classes{ci}, anno(1).name), 'kps');
    
    npart = size(kps,1);
    
    for fp = 1:npart
        part_list{fp} = [ 'part_' char(64+fp) ];
    end
    
    for fi = 1:length(anno)
        image_dir{fi} = classes{ci};
        image_name{fi} = [anno(fi).name(1:end-4) '.jpg'];
        image2anno{fi} = fi;
        anno2image(fi) = fi;
    end
    
    part_visible = true(length(part_list), length(anno));
    bbox = zeros(4, length(anno),'single');
    part_x = zeros(length(part_list), length(anno),'single');
    part_y = zeros(length(part_list), length(anno),'single');
    part_z = zeros(length(part_list), length(anno),'single');
    
    for fi = 1:length(anno)
        load(fullfile(conf.annoKPDir, classes{ci}, anno(fi).name), 'kps');
        kps=kps';
        bbox(:,fi) = [ min(kps(1,:)) min(kps(2,:))...
            max(kps(1,:)) max(kps(2,:)) ]';
        
        part_visible(:,fi) = ~isnan(kps(1,:))';
        
        if size(kps,2) == npart
            part_x(:, fi) = kps(1,:)';
            part_y(:, fi) = kps(2,:)';
            part_z(:, fi) = zeros(size(kps,2),1);
        else
            nt = size(kps,2);
            part_x(1:nt, fi) = kps(1,:)';
            part_y(1:nt, fi) = kps(2,:)';
            part_z(1:nt, fi) = zeros(size(kps,2),1);
            fprintf('%s file error!\n',anno(fi).name);
        end
        
    end
    
    KP.image_dir = image_dir;
    KP.image_name = image_name;
    KP.image2anno = image2anno;
    KP.anno2image = anno2image;
    KP.part_list = part_list;
    KP.part_visible = part_visible;
    KP.bbox = bbox;
    KP.part_x = part_x;
    KP.part_y = part_y;
    KP.part_z = part_z;
    save(fullfile(conf.benchmarkDir,sprintf('KP_%s.mat',classes{ci})), 'KP');
    fprintf('%d annotations processed\n',fi);
end



