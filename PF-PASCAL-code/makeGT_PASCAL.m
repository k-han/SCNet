% script for making GT data based on the annotations and the proposals
% written by Bumsub Ham, Inria - WILLOW / ENS, Paris, France

function makeGT_PASCAL()

global conf;
global tps_interp;

ShowGT = false;

% load matching pair
load(fullfile(conf.datasetDir,'parsePascalVOC.mat'), 'PascalVOC');

for ci = 1:numel(conf.class)
    
    fprintf('Processing %s...\n',conf.class{ci});
    
    % load the annotation file
    load(fullfile(conf.benchmarkDir,sprintf('KP_%s.mat',conf.class{ci})), 'KP');
    
    % load the indices of valid object proposals and corresponding object bounding box
    load(fullfile(conf.benchmarkDir,sprintf('AP_%s.mat',conf.class{ci})), 'AP');
    idx_for_active_op=AP.idx_for_active_op;             % indices for valid object proposals
    idx_for_bbox_of_active_op=AP.idx_for_bbox_of_active_op;     % indices for obj bounding box of valid object proposals
    
    % set matching pair
    classInd = pascalClassIndex(conf.class{ci});
    pair = PascalVOC.pair{classInd};
    
    pair_num = length(pair);
    data.images = cell(pair_num,2);
    data.proposals = cell(pair_num,2);
    data.proposals_GT = cell(pair_num,2);
    data.idx_for_active_opA = cell(pair_num,1);
    data.IoU2GT = cell(pair_num,1);
    data.bbox = cell(pair_num,2);
    data.part_x = cell(pair_num,2);
    data.part_y = cell(pair_num,2);

    for fi = 1:length(pair)
        % =========================================================================
        % configuration for image A
        % =========================================================================
        imgA_name = cell2mat(strcat(pair(fi,1)));
        imgA=imread(fullfile(conf.imageDir,[imgA_name '.jpg']));
        imgA_height=size(imgA,1);imgA_width=size(imgA,2);
        imgA_idx = find(strcmp(KP.image_name,[imgA_name '.jpg']));
        
        % load proposals
        load(fullfile(conf.proposalDir,KP.image_dir{imgA_idx},...
            [ imgA_name '_' func2str(conf.proposal) '.mat' ]), 'op');
        op.coords = op.coords';
        opA = op;
        
        idx_for_active_opA = idx_for_active_op{imgA_idx};
        
        %coordinates for upper left and lower right points of active proposals
        coords_for_active_opA = opA.coords(:,idx_for_active_opA);
        
        annoA = KP.image2anno{imgA_idx};
        part_x_A = KP.part_x(:,annoA);
        part_y_A = KP.part_y(:,annoA);
        % delete invisible parts
        part_x_A = part_x_A(~isnan(part_x_A));
        part_y_A = part_y_A(~isnan(part_y_A));
        
        % indices for coords_for_active_opA (clock-wise)
        xy_idx = zeros(4, size(coords_for_active_opA,2));
        xy_idx(1,:) =  sub2ind([imgA_width,imgA_height],coords_for_active_opA(1,:),coords_for_active_opA(2,:));
        xy_idx(2,:) =  sub2ind([imgA_width,imgA_height],coords_for_active_opA(3,:),coords_for_active_opA(2,:));
        xy_idx(3,:) =  sub2ind([imgA_width,imgA_height],coords_for_active_opA(3,:),coords_for_active_opA(4,:));
        xy_idx(4,:) =  sub2ind([imgA_width,imgA_height],coords_for_active_opA(1,:),coords_for_active_opA(4,:));
        
        % =========================================================================
        % configuration for image B
        % =========================================================================
        imgB_name = cell2mat(strcat(pair(fi,2)));
        imgB=imread(fullfile(conf.imageDir,[imgB_name '.jpg']));
        imgB_height=size(imgB,1);imgB_width=size(imgB,2);
        imgB_idx = find(strcmp(KP.image_name,[imgB_name '.jpg']));
        
        % load proposals
        load(fullfile(conf.proposalDir, KP.image_dir{fi},...
            [ imgB_name '_' func2str(conf.proposal) '.mat' ]), 'op');
        op.coords = op.coords';
        opB = op;
        
        annoB = KP.image2anno{imgB_idx};
        part_x_B = KP.part_x(:,annoB);
        part_y_B = KP.part_y(:,annoB);
        % delete invisible parts
        part_x_B = part_x_B(~isnan(part_x_B));
        part_y_B = part_y_B(~isnan(part_y_B));
        
        % =========================================================================
        % TPS warping using keypoints in image A and B
        % =========================================================================
        [coords_warped_opA_y, coords_warped_opA_x, imgW, imgWr]  = ...
            tpswarp(imgA,[imgB_width imgB_height],[double(part_y_A) double(part_x_A)],[double(part_y_B) double(part_x_B)],tps_interp); % thin plate spline warping
        
        % revising warped coordinates out of the image frame
        coords_warped_opA_x = max(min(round(coords_warped_opA_x),imgB_width),1);
        coords_warped_opA_y = max(min(round(coords_warped_opA_y),imgB_height),1);
        
        % x,y coordinates for (warped) points from image A (clock-wise)
        coords_warped_opA = [coords_warped_opA_x(xy_idx(1,:)), coords_warped_opA_y(xy_idx(1,:)),...
            coords_warped_opA_x(xy_idx(2,:)), coords_warped_opA_y(xy_idx(2,:)),...
            coords_warped_opA_x(xy_idx(3,:)), coords_warped_opA_y(xy_idx(3,:)),...
            coords_warped_opA_x(xy_idx(4,:)), coords_warped_opA_y(xy_idx(4,:))]';
        
        
        % ground truth (tight rectangular boxes, computed by averaging coordinates)
        coords_GT_opA_x = sort([coords_warped_opA_x(xy_idx(1,:)),coords_warped_opA_x(xy_idx(2,:)),...
            coords_warped_opA_x(xy_idx(3,:)),coords_warped_opA_x(xy_idx(4,:))]');
        coords_GT_opA_y = sort([coords_warped_opA_y(xy_idx(1,:)),coords_warped_opA_y(xy_idx(2,:)),...
            coords_warped_opA_y(xy_idx(3,:)),coords_warped_opA_y(xy_idx(4,:))]');
        coords_GT_opA = [(coords_GT_opA_x(1,:)+coords_GT_opA_x(2,:))./2;...
            (coords_GT_opA_y(1,:)+coords_GT_opA_y(2,:))./2;...
            (coords_GT_opA_x(3,:)+coords_GT_opA_x(4,:))./2;...
            (coords_GT_opA_y(3,:)+coords_GT_opA_y(4,:))./2];
        
        % revising coordinates of ground truth out of the image frame
        coords_GT_opA(1,:)=max(min(coords_GT_opA(1,:),imgB_width),1);
        coords_GT_opA(2,:)=max(min(coords_GT_opA(2,:),imgB_height),1);
        coords_GT_opA(3,:)=max(min(coords_GT_opA(3,:),imgB_width),1);
        coords_GT_opA(4,:)=max(min(coords_GT_opA(4,:),imgB_height),1);
        
        %convert bbox [xmin ymin xmax ymax] to [x y width hieght]
        coords_GT_opA_xywh = [coords_GT_opA(1,:);coords_GT_opA(2,:);...
            coords_GT_opA(3,:)-coords_GT_opA(1,:)+1;coords_GT_opA(4,:)-coords_GT_opA(2,:)+1];
        coords_opB_xyhw=[opB.coords(1,:);opB.coords(2,:);...
            opB.coords(3,:)-opB.coords(1,:)+1;opB.coords(4,:)-opB.coords(2,:)+1];
        
        % =========================================================================
        % compute IoU between GT of obj propoasl in image A and all obj proposals in image B
        % =========================================================================
        IoU2GT = 1-bboxOverlapRatio(coords_GT_opA_xywh', coords_opB_xyhw', 'Union');
        
        % =========================================================================
        % visualization ground truth for each object proposal in image A
        % =========================================================================
        if ShowGT
            colorCode = makeColorCode(100);
            
            % show warped image (from image A to B)
            warpout = appendimages(uint8(imgWr),uint8(imgW));
            clf(figure(1),'reset')
            figure(1);
            imshow(warpout);hold on;
            
            numOfPlot = 1;
            [score_IoU, idx_IoU] = sort(IoU2GT,2);
            
            imout = appendimages(imgA,imgB);
            
            for ki=1:10:length(idx_for_active_opA)
                clf(figure(2),'reset');
                figure(2);imshow(rgb2gray(imout)); hold on;
                
                % show keypoints in image A and B
                
                for kp=1:length(part_x_A)
                    plot(part_x_A(kp),part_y_A(kp),'o','MarkerEdgeColor','k',...
                        'MarkerFaceColor',colorCode(:,kp),'MarkerSize', 20);
                end
                
                for kp=1:length(part_x_B)
                    plot(part_x_B(kp)+size(imgA,2),part_y_B(kp),'o','MarkerEdgeColor','k',...
                        'MarkerFaceColor',colorCode(:,kp),'MarkerSize', 20);
                end
                
                fprintf('\npart: %d/%d\n', ki,length(idx_for_active_opA));
                
                % show each valid proposal in image A
                drawboxline(opA.coords(:,idx_for_active_opA(ki)),'LineWidth',4,'color',[255/255,215/255,0]);
                
                % candidate proposals in image B (ranked w.r.t 1-IoU, control parameter = numOfPlot)
                % for numOfPlot=1, upperbound match
                for kj=1:numOfPlot
                    % box
                    drawboxline(opB.coords(:,idx_IoU(ki,kj)),'LineWidth',4,'color',colorCode(:,kj),'offset',[ size(imgA,2) 0 ]);
                    h=text(double(10+size(imgA,2)),20,['1-overlap score: ' num2str(score_IoU(ki,kj))]);
                    h.FontSize = 14;
                    h.BackgroundColor = colorCode(:,kj);
                    
                    fprintf('1-overlap score (%d): %f \n', kj, score_IoU(ki,kj));
                end
                
                % show warped proposals from image A
                drawpolygon(coords_warped_opA(:,ki),'LineWidth',4,'color',[255/255,215/255,0],'offset',[size(imgA,2) 0 ]);
                h=text(double(10),double(20),'warped obj box');
                h.FontSize = 12;
                h.BackgroundColor = [255/255,215/255,0];
                
                % show ground truths
                drawboxline(coords_GT_opA(:,ki),'LineWidth',4,'color',[0,1,0],'offset',[ size(imgA,2) 0 ]);
                h=text(double(101),double(20),'ground truth');
                h.FontSize = 12;
                h.BackgroundColor = [0,1,0];
                
                h=text(double(10),double(40),'upper bound box for overlap score');
                h.FontSize = 12;
                h.BackgroundColor = colorCode(:,kj);
                
                pause;
            end
        end
        
        %save 1-IoU scores for all possible matches between GT and obj proposals in image B
        save(fullfile(conf.matchGTDir,KP.image_dir{fi},[ imgA_name ...
            '-' imgB_name '_' func2str(conf.proposal) '.mat' ]), 'IoU2GT');
        
        fprintf('%03d/%03d processed.\n',fi, length(pair));
        
        data.images{fi,1} = imgA;
        data.images{fi,2} = imgB;
        data.proposals_GT{fi,1} = single(coords_for_active_opA');
        data.proposals_GT{fi,2} = single(coords_GT_opA');
        data.proposals{fi, 1} = single(opA.coords');
        data.proposals{fi, 2} = single(opB.coords');
        data.idx_for_active_opA{fi, 1} = idx_for_active_opA;
        data.IoU2GT{fi, 1} = single(IoU2GT);
        data.bbox{fi, 1} = KP.bbox(:,annoA)';
        data.bbox{fi, 2} = KP.bbox(:,annoB)';
        data.part_x{fi, 1} = KP.part_x(:,annoA)';
        data.part_y{fi, 1} = KP.part_y(:,annoA)';
        data.part_x{fi, 2} = KP.part_x(:,annoB)';
        data.part_y{fi, 2} = KP.part_y(:,annoB)';
    end
    save(fullfile(conf.matchGTDir,KP.image_dir{fi},[ conf.class{ci} '.mat' ]), 'data');
end

end
