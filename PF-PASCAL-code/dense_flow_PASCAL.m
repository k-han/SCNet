% script for computing dense flow field
% written by Bumsub Ham, Inria - WILLOW / ENS, Paris, France

function dense_flow_PASCAL()
% set_path;
% evalc(['set_conf_', db_name]);

global conf;
global sdf;

%temp
conf.feature = conf.feature(1);

bShowMatch = false;
bShowDenseMatch = false;

% load matching pair
load(fullfile(conf.datasetDir,'parsePascalVOC.mat'), 'PascalVOC');

for ft = 1:numel(conf.feature)
    
    % measure time
    time = struct([]);
    for fa = 1:numel(conf.algorithm)
        time(fa).method = func2str(conf.algorithm{fa});
        time(fa).sec    = [];
    end
    
    for ci = 1:length(conf.class)
        
        fprintf('Processing %s...',conf.class{ci});
        
        % load the annotation file
        load(fullfile(conf.benchmarkDir,sprintf('KP_%s.mat',conf.class{ci})), 'KP');
        
        % set matching pair
        classInd = pascalClassIndex(conf.class{ci});
        pair = PascalVOC.pair{classInd};
        
        for i=1:length(pair)
            
            % load object proposal and feature of image A
            imgA_name = cell2mat(strcat(pair(i,1)));
            imgA=imread(fullfile(conf.imageDir,[imgA_name '.jpg']));
            imgA_idx = find(strcmp(KP.image_name,[imgA_name '.jpg']));
            
            load(fullfile(conf.proposalDir,KP.image_dir{i},[ imgA_name...
                '_' func2str(conf.proposal) '.mat' ]), 'op');
            load(fullfile(conf.featureDir,KP.image_dir{i},conf.feature{ft},[ imgA_name...
                '_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ]), 'feat');
            viewA = load_view(imgA,op,feat,'conf', conf);
            
            opA=frame2box(viewA.frame);
            opA_xywh = [opA(1,:);opA(2,:);opA(3,:)-opA(1,:)+1;opA(4,:)-opA(2,:)+1];
            [viewA_H, viewA_W,~] = size(viewA.img);
            
            
            % load object proposal and feature of image B
            imgB_name = cell2mat(strcat(pair(i,2)));
            imgB=imread(fullfile(conf.imageDir,[imgB_name '.jpg']));
            imgB_idx = find(strcmp(KP.image_name,[imgB_name '.jpg']));
            
            load(fullfile(conf.proposalDir,KP.image_dir{i},[ imgB_name...
                '_' func2str(conf.proposal) '.mat' ]), 'op');
            load(fullfile(conf.featureDir,KP.image_dir{i},conf.feature{ft},[ imgB_name...
                '_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ]), 'feat');
            viewB = load_view(imgB, op, feat, 'conf', conf);
            
            [viewB_H, viewB_W,~] = size(viewB.img);
            
            fprintf('\n========== %s-(%03d/%03d) ==========\n',conf.class{ci}, i, length(pair));
            fprintf('+ features: %s\n', conf.feature{ft} );
            fprintf('+ object proposal: %s\n', func2str(conf.proposal) );
            fprintf('+ number of proposals: A %d => B %d\n', size(viewA.desc,2), size(viewB.desc,2) );
            
            % load matching results
            for fa = 1:numel(conf.algorithm)
                tic;
                fprintf(' - %s matching... \n', func2str(conf.algorithm{fa}));
                load(fullfile(conf.matchDir,KP.image_dir{i},conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.mat' ]), 'pmatch');
                
                tic;
                
                [ confidenceA, max_id ] = max(pmatch.confidence,[],2);
                
                [anchor_confA, anchor_idA]=sort(confidenceA,'descend');
                anchor_idB=max_id(anchor_idA);
                
                opB=frame2box(viewB.frame);
                opB_xywh = [opB(1,:);opB(2,:);opB(3,:)-opB(1,:)+1;opB(4,:)-opB(2,:)+1];
                
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
                    
                    cand_opA=opA(:,idxA);
                    cand_opB=opB(:,idxB);
                    
                    cand_opA_xywh=opA_xywh(:,idxA);
                    cand_opB_xywh=opB_xywh(:,idxB);
                    
                    idxValid_temp=idxValid(cand_opA(2):cand_opA(4),cand_opA(1):cand_opA(3));
                    if numel(find(idxValid_temp==0)) == 0
                        continue;
                    else
                        % warped coordinate (x,y) for four points in
                        % rectangle
                        CellGrid = NaN(cand_opA_xywh(4),cand_opA_xywh(3),2);
                        CellGrid(1,1,1:2)=cat(3,cand_opB(1),cand_opB(2));
                        CellGrid(1,end,1:2)=cat(3,cand_opB(3),cand_opB(2));
                        CellGrid(end,1,1:2)=cat(3,cand_opB(1),cand_opB(4));
                        CellGrid(end,end,1:2)=cat(3,cand_opB(3),cand_opB(4));
                        
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
                                    Conf_Dense(cand_opA(2)+p-1,cand_opA(1)+q-1)=anchor_conf;
                                    idxValid(cand_opA(2)+p-1,cand_opA(1)+q-1) =1;
                                else
                                    continue;
                                end
                            end
                        end
                    end
                    
                    if bShowMatch
                        hFig_match = figure(4); clf;
                        imgInput = appendimages( viewA.img, viewB.img, 'h' );
                        imshow(rgb2gray(imgInput)); hold on; iptsetpref('ImshowBorder','tight');
                        h1=rectangle('Position',cand_opA_xywh(:),'EdgeColor', 'y');
                        h2=rectangle('Position',cand_opB_xywh(:)+[viewA_W;0;0;0],'EdgeColor', 'r');
                        h1.LineWidth=4;h2.LineWidth=4;
                        
                        fprintf('Cell #: %d',k);
                        fprintf(' + Confidence: %f\n',anchor_conf);
                        pause;
                        clear h1;clear h2;clear h3;clear h4;
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
                vx = sdfilter(im2double(viewA.img),u0,vx,Mask_for_Reg,...
                    sdf.nei,sdf.lambda,sdf.sigma_g,sdf.sigma_u,sdf.itr,sdf.issparse);
                fprintf('  > SD Filtering for y offset\n');
                vy = sdfilter(im2double(viewA.img),u0,vy,Mask_for_Reg,...
                    sdf.nei,sdf.lambda,sdf.sigma_g,sdf.sigma_u,sdf.itr,sdf.issparse);
                
                p=1:viewA_W;
                q=1:viewA_H;
                p=repmat(p,viewA_H,1);
                q=repmat(q',1,viewA_W);
                WarpCoordXY(:,:,1)=vx+p;
                WarpCoordXY(:,:,2)=vy+q;
                WarpCoordXY = round(WarpCoordXY);
                
                dmatch.vx = round(vx);
                dmatch.vy = round(vy);
                
                t_match = toc;
                time(fa).sec  = [time(fa).sec; t_match];
                
                if isempty(dir(fullfile(conf.flowDir,conf.class{ci},conf.feature{ft})))
                    mkdir(fullfile(conf.flowDir,conf.class{ci},conf.feature{ft}));
                end
                
                imgWarping=warpImage(im2double(imgB),dmatch.vx,dmatch.vy);
                
                save(fullfile(conf.flowDir, KP.image_dir{i},conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.mat' ]), 'dmatch');
                
                imwrite(imgWarping, fullfile(conf.flowDir, KP.image_dir{i}, conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.jpg' ]));
                
                
                % =========================================================================
                % visualization dense flow field
                % show PCK averaged over the number of keypoints
                % =========================================================================
                
                if bShowDenseMatch
                    colorCode = makeColorCode(100);
                    
                    clf(figure(2),'reset');
                    imgInput = appendimages( viewA.img, viewB.img, 'h' );
                    figure(2);imshow(imgInput);hold on;
                    
                    
                    annoA = KP.image2anno{imgA_idx};
                    part_x_A = KP.part_x(:,annoA);
                    part_y_A = KP.part_y(:,annoA);
                    % delete invisible parts
                    part_x_A = part_x_A(~isnan(part_x_A));
                    part_y_A = part_y_A(~isnan(part_y_A));
                    
                    annoB = KP.image2anno{imgB_idx};
                    part_x_B = KP.part_x(:,annoB);
                    part_y_B = KP.part_y(:,annoB);
                    % delete invisible parts
                    part_x_B = part_x_B(~isnan(part_x_B));
                    part_y_B = part_y_B(~isnan(part_y_B));
                    
                    
                    for kp=1:length(part_x_A)
                        plot(part_x_A(kp),part_y_A(kp),'o','MarkerEdgeColor','k',...
                            'MarkerFaceColor',colorCode(:,kp),'MarkerSize', 10);
                        
                        plot(WarpCoordXY(round(part_y_A(kp)),round(part_x_A(kp)),1)+viewA_W...
                                ,WarpCoordXY(round(part_y_A(kp)),round(part_x_A(kp)),2),'s','MarkerEdgeColor','k',...
                                'MarkerFaceColor',colorCode(:,kp),'MarkerSize', 10);
                    end
                    
                    for kp=1:length(part_x_B)
                        plot(part_x_B(kp)+size(imgA,2),part_y_B(kp),'o','MarkerEdgeColor','k',...
                            'MarkerFaceColor',colorCode(:,kp),'MarkerSize', 10);
                    end
                    
                    PCK2GT = zeros(min(numel(part_x_A),numel(part_x_B)),1);
                    for k=1:length(PCK2GT)
                        PCK2GT(k) = (WarpCoordXY(round(part_y_A(k)),round(part_x_A(k)),1)-part_x_B(k))^2+...
                            (WarpCoordXY(round(part_y_A(k)),round(part_x_A(k)),2)-part_y_B(k))^2;
                    end
                    PCK2GT=sqrt(PCK2GT);
                    PCK2GT=PCK2GT./max(KP.bbox(3:4,imgB_idx) - KP.bbox(1:2,imgB_idx));
                    PCK2GT = PCK2GT<=0.1;
                    fprintf('  * Average PCK: %f\n',sum(PCK2GT)/length(PCK2GT));
                    
                    clf(figure(3),'reset');figure(3);imshow(flowToColor(cat(3,vx,vy)));%Flow =cat(3,vx,vy);cquiver(Flow(1:10:end,1:10:end,:));
                    WarpedImg=warpImage(im2double(viewB.img),vx,vy);
                    clf(figure(4),'reset');figure(4);imshow(WarpedImg);
                    pause;
                end
            end
        end
    end
    
    for fa = 1:numel(conf.algorithm)
        time(fa).avg = mean(time(fa).sec);
        time(fa).std = std(time(fa).sec);
    end
    save( fullfile(conf.timeDir, [ 'PF_DF_time_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ] ), 'time');
    
end
