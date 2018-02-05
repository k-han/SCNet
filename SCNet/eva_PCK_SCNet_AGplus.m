function [ PCK ] = eva_PCK_SCNet_AGplus()
useGPU = 1;
if useGPU
 gpuDevice([1]);
end
visualize = 0;
%run(fullfile('../utils', 'vlfeat-0.9.20','toolbox', 'vl_setup.m'));
addpath(genpath('../utils/'));
%run(fullfile(fileparts(mfilename('fullpath')), '../..', 'matlab', 'vl_setupnn.m'));

% parameter for TPS warping
tps_interp.method = 'invdist'; %'nearest'; %'none' % interpolation method
tps_interp.radius = 5; % radius or median filter dimension
tps_interp.power = 2; %power for inverse wwighting interpolation method

% parameter for SD-filtering (SDF)
sdf.nei= 0;                 % 0: 4-neighbor 1: 8-neighbor
sdf.lambda = 20;            % smoothness parameter
sdf.sigma_g = 30;           % bandwidth for static guidance
sdf.sigma_u = 15;           % bandwidth for dynamic guidance
sdf.itr=2;                  % number of iterations
sdf.issparse=true;          % is the input sparse or not

%normalization flag
normalization = 'same size';%'different sizes', 'bounding boxes'

%revise modelPath and imdbPath accordingly
modelPath = '../data/trained_models/PASCAL-RP/SCNet-AGplus.mat';
%imdbPath = fullfile(vl_rootnn, 'data', 'pf', 'PF-PASCAL-RP-500.mat');
imdbPath = fullfile('../data', 'PF-PASCAL-RP-1000.mat');


load(modelPath);
net = dagnn.DagNN.loadobj(net);
load(imdbPath);
removeLayer(net, 'loss');
net=net.saveobj;
%net.layers(56).inputs{4}='b1_input' 
%net.layers(56).inputs{5}='b2_input'
net = dagnn.DagNN.loadobj(net);
net.conserveMemory = false ;

val_idx = find(data.set==3);
nsets = numel(val_idx);
tbinv_PCK = linspace(0,1,101);
pck = zeros(numel(val_idx), numel(tbinv_PCK));

outputname = 'AG_out';

%opA = struct();
%opB = struct();
PCK=[]

tbinv = linspace(0,1,101);
histo_pck = zeros(numel(tbinv),1);
            
for i = 1:numel(val_idx)
    %% pmatch: Proposal Matching
    batch = val_idx(i);
    disp(batch)
    images_A = data.images{batch,1} ;
    images_B = data.images{batch,2} ;
    image_mean = data.image_mean;
    image_std = data.image_std;
    images_A = (single(images_A) - image_mean)./image_std;
    images_B = (single(images_B) - image_mean)./image_std;
    proposals_A = data.proposals{batch,1};
    proposals_B = data.proposals{batch,2};
    proposals_A = [ones(size(proposals_A,1), 1).*1 proposals_A];
    proposals_B = [ones(size(proposals_B,1), 1).*2 proposals_B];
      
    if exist(['flow/AG1000_2/' num2str(i) '.mat'])==0

        if useGPU
            inputs = {'b1_input', gpuArray(im2single(images_A)), 'b2_input', gpuArray(im2single(images_B)), 'b1_rois', gpuArray(single(proposals_A')), 'b2_rois', gpuArray(single(proposals_B'))} ;
            net.move('gpu');
        else
            inputs = {'b1_input', im2single(images_A), 'b2_input', im2single(images_B), 'b1_rois', single(proposals_A'), 'b2_rois', single(proposals_B')} ;
        end


        
        % get confidenceMap
        net.eval(inputs);

        confidenceMap = net.vars(net.getVarIndex(outputname)).value;
        %[ confidenceA, max_idx ] = max(confidenceMap(idx_for_active_opA,:),[],2);
        [ confidenceA, max_id ] = max(confidenceMap,[],2);
        %[anchor_confA, anchor_idA]=sort(confidenceA,'descend');
        %anchor_idB=max_id(anchor_idA);

        viewA = vl_getView2_ignacio(data.proposals{batch,1});
        viewB = vl_getView2_ignacio(data.proposals{batch,2});
        opA_=frame2box(viewA.frame);
        opA_xywh = [opA_(1,:);opA_(2,:);opA_(3,:)-opA_(1,:)+1;opA_(4,:)-opA_(2,:)+1];
        [viewA_H, viewA_W,~] = size(images_A);
        [viewB_H, viewB_W,~] = size(images_B);


        tic;
        [anchor_confA, anchor_idA]=sort(confidenceA,'descend');
        anchor_idB=max_id(anchor_idA);
        opB_=frame2box(viewB.frame);
        opB_xywh = [opB_(1,:);opB_(2,:);opB_(3,:)-opB_(1,:)+1;opB_(4,:)-opB_(2,:)+1];
        % dense warping field (x,y)
        WarpCoordXY = NaN(viewA_H,viewA_W,2);
        Conf_Dense = zeros(viewA_H,viewA_W);
        idxValid = zeros(viewA_H,viewA_W);
        % =========================================================================
        % initial dense warping field
        % find the matches having highest matching confidence, and then estimate dense correspondence
        % by interpolation.
        % =========================================================================
        for k=1:numel(anchor_idA)
            idxA=anchor_idA(k);
            idxB=anchor_idB(k);
            anchor_conf = anchor_confA(k);
            cand_opA=round(opA_(:,idxA));
            cand_opB=round(opB_(:,idxB));
            cand_opA_xywh=round(opA_xywh(:,idxA));
            cand_opB_xywh=round(opB_xywh(:,idxB));
            idxValid_temp=idxValid(cand_opA(2):cand_opA(4),cand_opA(1):cand_opA(3));
            if numel(find(idxValid_temp==0)) == 0
                continue;
            else
                % warped coordinate (x,y) for four points in
                % rectangle
                CellGrid = NaN(round(cand_opA_xywh(4)),round(cand_opA_xywh(3)),2);
                CellGrid(1,1,1:2)=cat(3,round(cand_opB(1)),round(cand_opB(2)));
                CellGrid(1,end,1:2)=cat(3,round(cand_opB(3)),round(cand_opB(2)));
                CellGrid(end,1,1:2)=cat(3,round(cand_opB(1)),round(cand_opB(4)));
                CellGrid(end,end,1:2)=cat(3,round(cand_opB(3)),round(cand_opB(4)));
                CellGridX = CellGrid(:,:,1);
                CellGridY = CellGrid(:,:,2);
                if size(CellGrid,1)==1 || size(CellGrid,2)==1
                    continue;
                else
                    [y,x] = find(~isnan(CellGridX));
                    indexes = sub2ind(size(CellGridX),y,x);
                    interpolator = scatteredInterpolant(y,x,double(CellGridX(indexes)), 'linear');
                    [X,Y] = meshgrid(1:size(CellGridX,2),1:size(CellGridX,1));
                    interpolated_CellGridX = interpolator(Y,X);
                    interpolator = scatteredInterpolant(y,x,double(CellGridY(indexes)), 'linear');
                    interpolated_CellGridY = interpolator(Y,X);
                end
                clear CellGridX;clear CellGridY;
                for p=1:cand_opA_xywh(4)
                    for q=1:cand_opA_xywh(3)
                        if idxValid(cand_opA(2)+p-1,cand_opA(1)+q-1)==0
                            WarpCoordXY(cand_opA(2)+p-1,cand_opA(1)+q-1,1)=interpolated_CellGridX(p,q);
                            WarpCoordXY(cand_opA(2)+p-1,cand_opA(1)+q-1,2)=interpolated_CellGridY(p,q);
                            Conf_Dense(cand_opA(2)+p-1,cand_opA(1)+q-1)=gather(anchor_conf);
                            idxValid(cand_opA(2)+p-1,cand_opA(1)+q-1) =1;
                        else
                            continue;
                        end
                    end
                end
            end

        end
        WarpCoordXY = round(WarpCoordXY);
        WarpCoordXY(:,:,1)=max(min(WarpCoordXY(:,:,1),viewB_W),1);
        WarpCoordXY(:,:,2)=max(min(WarpCoordXY(:,:,2),viewB_H),1);
        p=1:viewA_W;
        q=1:viewA_H;
        p=repmat(p,viewA_H,1);
        q=repmat(q',1,viewA_W);
        vx=WarpCoordXY(:,:,1)-p;
        vy=WarpCoordXY(:,:,2)-q;
        % =========================================================================
        % filtering outliers in initial dense warping field
        % using matching confidence
        % =========================================================================
        Buffer_for_WarpCoordXY = NaN(viewB_H,viewB_W,2);
        Buffer_for_Conf = NaN(viewB_H,viewB_W);
        for p=1:viewA_H
            for q=1:viewA_W
                WarpCoordX=vx(p,q)+q;
                WarpCoordY=vy(p,q)+p;
                if isnan(Buffer_for_Conf(WarpCoordY,WarpCoordX))
                    Buffer_for_Conf(WarpCoordY,WarpCoordX) = Conf_Dense(p,q);
                    Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,1)=q;
                    Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,2)=p;
                else
                    if  Buffer_for_Conf(WarpCoordY,WarpCoordX) < Conf_Dense(p,q)
                        Buffer_for_Conf(WarpCoordY,WarpCoordX) = Conf_Dense(p,q);
                        vx(Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,2)...
                            ,Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,1))=nan;
                        vy(Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,2)...
                            ,Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,1))=nan;
                        Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,1)=q;
                        Buffer_for_WarpCoordXY(WarpCoordY,WarpCoordX,2)=p;
                        Conf_Dense(p,q)=0;
                    else
                        vx(p,q)=nan;
                        vy(p,q)=nan;
                        Conf_Dense(p,q)=0;
                    end
                end
            end
        end
        % =========================================================================
        % dense field regularization
        % SD Filtering
        % =========================================================================
        Mask_for_Reg = isnan(vx) | isnan(vy);
        Mask_for_Reg=1-Mask_for_Reg;
        vx=vx.*Mask_for_Reg;
        vy=vy.*Mask_for_Reg;
        u0=ones(viewA_H,viewA_W);
        fprintf('  > SD Filtering for x offset\n');
        vx = sdfilter(im2double(images_A),u0,vx,Mask_for_Reg,...
        sdf.nei,sdf.lambda,sdf.sigma_g,sdf.sigma_u,sdf.itr,sdf.issparse);
        fprintf('  > SD Filtering for y offset\n');
        vy = sdfilter(im2double(images_A),u0,vy,Mask_for_Reg,...
        sdf.nei,sdf.lambda,sdf.sigma_g,sdf.sigma_u,sdf.itr,sdf.issparse);
        p=1:viewA_W;
        q=1:viewA_H;
        p=repmat(p,viewA_H,1);
        q=repmat(q',1,viewA_W);
        WarpCoordXY(:,:,1)=vx+p;
        WarpCoordXY(:,:,2)=vy+q;
        WarpCoordXY = round(WarpCoordXY);
        
        
        if 0
            dmatch.vx = round(vx);
            dmatch.vy = round(vy);
            if isempty(dir(fullfile(conf.flowDir,conf.feature{ft})))
                mkdir(fullfile(conf.flowDir,conf.feature{fa}));
            end
            imgWarping=warpImage(im2double(images_B),dmatch.vx,dmatch.vy);

            save(fullfile(conf.flowDir,conf.feature{ft},...
                [ 'pair_' num2str(fi) '_' func2str(conf.proposal) '_' ...
                conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.mat' ]), 'dmatch');

            imwrite(imgWarping, fullfile(conf.flowDir, conf.feature{ft},...
               [ 'pair_' num2str(fi) '_' func2str(conf.proposal) '_' ...
               conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.jpg' ]));
        end

        save(['flow/AG1000_2/' num2str(i) '.mat'],'vx','vy')
    else
        load(['flow/AG1000_2/' num2str(i) '.mat'],'vx','vy')
    end

    %% Evaluation of Dense Match
    %tbinv = linspace(0,1,101);
    %histo_pck = zeros(numel(tbinv),1);
    H = size(data.images{1}, 1);
    part_x = data.part_x(batch,:); 
    part_y = data.part_y(batch,:);
    vis_part = (part_x{1} <Inf).*(part_x{2} <Inf);
    vis_part_idx = find(vis_part == 1);

    PCK2GT = zeros(size(vis_part_idx));

    %vx=dmatch.vx;
    %vy=dmatch.vy;
    for k=1:numel(PCK2GT)
        p_idx = vis_part_idx(k);
        px=round(part_x{1}(p_idx));
        py=round(part_y{1}(p_idx));

        PCK2GT(k) = (part_x{1}(p_idx)+vx(py,px)-part_x{2}(p_idx))^2+(part_y{1}(p_idx)+vy(py,px)-part_y{2}(p_idx))^2;
    end
    PCK2GT=sqrt(PCK2GT);
    switch normalization
	case 'same size'
	    PCK2GT=PCK2GT./H;  % size(images_B,1);
	case 'different sizes'
	    PCK2GT=PCK2GT./sqrt(size(images_B,1)^2+size(images_B,2)^2);
	case 'bounding boxes'
	    PCK2GT=PCK2GT./max(data.bbox{batch,2}(3:4)-data.bbox{batch,2}(1:2));
    end

    bin_PCK = vl_binsearch(tbinv, double(PCK2GT));
    for p=1:numel(bin_PCK)
        histo_pck(bin_PCK(p)) = histo_pck(bin_PCK(p)) + 1.0/numel(PCK2GT)/nsets;
    end

end
    
if visualize
    %figure;
    %set_figure
    % set figure
    fig_width = 6; % inch
    fig_height = 6; % inch
    margin_vert_u = 0.35; % inch
    margin_vert_d = 0.5; % inch
    margin_horz_l = 0.70; % inch
    margin_horz_r = 0.15; % inch
    [hFig,hAxe] = figmod(fig_width,fig_height,margin_vert_u,margin_vert_d,margin_horz_l,margin_horz_r);

    linewidth=3;
    fontsize = 16;

    hold on;
    plot( tbinv(1:end-1), cumsum(histo_pck(1:end-1)),...
         '-', 'LineWidth', 1, 'MarkerSize', 1);
    legend('SCNet-AG+','Location','northwest');
    xlabel(hAxe,'\tau','FontName', 'helvetica','FontSize',fontsize);
    ylabel(hAxe,'PCK','FontName', 'helvetica','FontSize',fontsize);
    %title(['Results for ' data.db_name ' ' func2str(conf.proposal) '-' num2str(conf.num_op)]);
end

PCK = cumsum(histo_pck(1:end-1));
     
end
