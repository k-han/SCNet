% script for computing dense flow field
% written by Bumsub Ham, Inria - WILLOW / ENS, Paris, France

function dense_flow_ft_PASCAL()
% set_path;
% evalc(['set_conf_', db_name]);

global conf;

%temp
conf.feature = conf.feature(1);

bShowMatch = false;
bShowDenseMatch = false;

% SiftFlow
cellsize=3;
gridspacing=1;

SIFTflowpara.alpha=2*255;
SIFTflowpara.d=40*255;
SIFTflowpara.gamma=0.005*255;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=2;
SIFTflowpara.topwsize=10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations= 30;


nIterationArray=round(linspace(SIFTflowpara.nIterations,SIFTflowpara.nIterations,SIFTflowpara.nlevels));

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
            imgA=im2double(imread(fullfile(conf.imageDir,[imgA_name '.jpg'])));
            imgA_idx = find(strcmp(KP.image_name,[imgA_name '.jpg']));
            [viewA_H, viewA_W,~] = size(imgA);
            siftA = mexDenseSIFT(imgA,cellsize,gridspacing);
            
            
            % load object proposal and feature of image B
            imgB_name = cell2mat(strcat(pair(i,2)));
            imgB=im2double(imread(fullfile(conf.imageDir,[imgB_name '.jpg'])));
            imgB_idx = find(strcmp(KP.image_name,[imgB_name '.jpg']));
            [viewB_H, viewB_W,~] = size(imgB);
            siftB = mexDenseSIFT(imgB,cellsize,gridspacing);
            
            fprintf('\n========== %s-(%03d/%03d) ==========\n',conf.class{ci}, i, length(pair));
            fprintf('+ features: %s\n', conf.feature{ft} );
            fprintf('+ object proposal: %s\n', func2str(conf.proposal) );
            
            % load matching results
            for fa = 1:numel(conf.algorithm)
                tic;
                fprintf(' - %s matching... \n', func2str(conf.algorithm{fa}));
                
                load(fullfile(conf.flowDir,KP.image_dir{i},conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.mat' ]), 'dmatch');
                
                
                winSizeX=ones(viewA_H,viewA_W)*(SIFTflowpara.wsize);
                winSizeY=ones(viewA_H,viewA_W)*(SIFTflowpara.wsize);
                
                tic;
                
                [flow,foo]=mexDiscreteFlow(siftA,siftB,[SIFTflowpara.alpha,SIFTflowpara.d,SIFTflowpara.gamma,...
                    nIterationArray(1),SIFTflowpara.nlevels-1,SIFTflowpara.wsize],dmatch.vx,dmatch.vy,winSizeX,winSizeY);
                
                t_match = toc;
                time(fa).sec  = [time(fa).sec; t_match];
                
                vx=flow(:,:,1);
                vy=flow(:,:,2);
                
                dmatch_ft.vx=vx;
                dmatch_ft.vy=vy;
                
                
                if isempty(dir(fullfile(conf.flowDir,conf.class{ci},conf.feature{ft})))
                    mkdir(fullfile(conf.flowDir,conf.class{ci},conf.feature{ft}));
                end
                
                imgWarping=warpImage(im2double(imgB),dmatch_ft.vx,dmatch_ft.vy);
                
                save(fullfile(conf.flowDir, KP.image_dir{i},conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '_ft.mat' ]), 'dmatch_ft');
                
                imwrite(imgWarping, fullfile(conf.flowDir, KP.image_dir{i}, conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '_ft.jpg' ]));
                
                
                % =========================================================================
                % visualization dense flow field
                % show PCK averaged over the number of keypoints
                % =========================================================================
                
                if bShowDenseMatch
                    WarpCoordXY = zeros(viewA_H,viewA_W,2);
                    p=1:viewA_W;
                    q=1:viewA_H;
                    p=repmat(p,viewA_H,1);
                    q=repmat(q',1,viewA_W);
                    WarpCoordXY(:,:,1)=vx+p;
                    WarpCoordXY(:,:,2)=vy+q;
                    WarpCoordXY = round(WarpCoordXY);
                    
                    colorCode = makeColorCode(100);
                    
                    clf(figure(2),'reset');
                    imgInput = appendimages( imgA, imgB, 'h' );
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
                    clf(figure(4),'reset');figure(4);imshow(imgWarping);
                    pause;
                end
            end
        end
    end
    
    for fa = 1:numel(conf.algorithm)
        time(fa).avg = mean(time(fa).sec);
        time(fa).std = std(time(fa).sec);
    end
    save( fullfile(conf.timeDir, [ 'PF_FT_time_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ] ), 'time');
    
end
