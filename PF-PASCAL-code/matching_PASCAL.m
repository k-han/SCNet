% script for matching object proposals
% written by Bumsub Ham, Inria - WILLOW / ENS, Paris, France

function matching_PASCAL()

global conf;

bShowMatch = false;  % show k best matches for each pairwise matching

% load matching pair
load(fullfile(conf.datasetDir,'parsePascalVOC.mat'), 'PascalVOC');

%loop through features
for ft = 1:numel(conf.feature)
    
    % measure time
    time = struct([]);
    for fa = 1:numel(conf.algorithm)
        time(fa).method = func2str(conf.algorithm{fa});
        time(fa).sec    = [];
    end
    
    for ci = 1:numel(conf.class)
        
        fprintf('Processing %s...',conf.class{ci});
        % load the annotation file
        load(fullfile(conf.benchmarkDir,sprintf('KP_%s.mat',conf.class{ci})), 'KP');
        load(fullfile(conf.benchmarkDir,sprintf('AP_%s.mat',conf.class{ci})), 'AP');
        
        % set matching pair
        classInd = pascalClassIndex(conf.class{ci});
        pair = PascalVOC.pair{classInd};
        
        % box matching from image A to image B
        for i=1:length(pair)
            
            % load object proposals and features for image A
            imgA_name = cell2mat(strcat(pair(i,1)));
            imgA=imread(fullfile(conf.imageDir,[imgA_name '.jpg']));
            imgA_idx = find(strcmp(KP.image_name,[imgA_name '.jpg']));
            
            load(fullfile(conf.proposalDir,KP.image_dir{i},[ imgA_name...
                '_' func2str(conf.proposal) '.mat' ]), 'op');
            load(fullfile(conf.featureDir,KP.image_dir{i},conf.feature{ft},[ imgA_name...
                '_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ]), 'feat');
            viewA = load_view(imgA,op,feat,'conf', conf);
            
            if bShowMatch
                % indicies for active object proposals transform current indices to candidate indices
                idx_for_active_opA = zeros(AP.num_op_all(imgA_idx),1,'int32');
                idx_for_active_opA(AP.idx_for_active_op{imgA_idx}) = 1:numel(AP.idx_for_active_op{imgA_idx}); % original index to current index
                idx_for_active_opA = idx_for_active_opA(viewA.idx2ori)';
            end
            
            % load object proposals and features for image B
            imgB_name = cell2mat(strcat(pair(i,2)));
            imgB=imread(fullfile(conf.imageDir,[imgB_name '.jpg']));
            imgB_idx = find(strcmp(KP.image_name,[imgB_name '.jpg']));
            
            load(fullfile(conf.proposalDir,KP.image_dir{i},[ imgB_name...
                '_' func2str(conf.proposal) '.mat' ]), 'op');
            load(fullfile(conf.featureDir,KP.image_dir{i},conf.feature{ft},[ imgB_name...
                '_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ]), 'feat');
            viewB = load_view(imgB, op, feat, 'conf', conf);
            
            fprintf('\n========== %s-(%03d/%03d) ==========\n',conf.class{ci}, i, length(pair));
            fprintf('+ features: %s\n', conf.feature{ft} );
            fprintf('+ object proposal: %s\n', func2str(conf.proposal) );
            fprintf('+ number of proposals: A %d => B %d\n', size(viewA.desc,2), size(viewB.desc,2) );
            
            % run matching algorithms
            for fa = 1:numel(conf.algorithm)
                
                fprintf(' - %s matching... ', func2str(conf.algorithm{fa}));
                
                % options for matching
                opt.bDeleteByAspect = true;
                opt.bDensityAware = false;
                opt.bSimVote = true;
                opt.bVoteExp = true;
                opt.feature = conf.feature{ft};
                %profile on;
                
                tic;
                confidenceMap = feval( conf.algorithm{fa}, viewA, viewB, opt );
                t_match = toc;
                time(fa).sec  = [time(fa).sec; t_match];
                
                fprintf('   took %.2f secs\n',t_match);
                
                % matching confidence
                [ confidenceA, max_id ] = max(confidenceMap,[],2);
                
                pmatch.confidence = confidenceMap;
                pmatch.match = [ viewA.idx2ori viewB.idx2ori(max_id) ]';
                pmatch.match_confidence = confidenceA;
                
                if isempty(dir(fullfile(conf.matchDir,conf.class{ci},conf.feature{ft})))
                    mkdir(fullfile(conf.matchDir,conf.class{ci},conf.feature{ft}));
                end
                
                save(fullfile(conf.matchDir, KP.image_dir{i},conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.mat' ]), 'pmatch');
                
                % =========================================================================
                % visualization top k-matches and their valid matches
                % according to the IoU threshold.
                % paramter:
                % num_of_top_k_matches
                % IoU_threshold
                % =========================================================================
                
                if bShowMatch
                    strVisMode = 'box';
                    
                    load(fullfile(conf.matchGTDir,KP.image_dir{i},...
                        [ imgA_name '-' imgB_name...
                        '_' func2str(conf.proposal) '.mat' ]), 'IoU2GT'); % for visualization
                    
                    idx_for_opB = pmatch.match(2,:);
                    idx_valid = find((idx_for_active_opA > 0) & (idx_for_opB > 0));
                    
                    %                 %% top k-matches
                    %                 [~,idx_sort_conf]=sort(confidenceA,'descend');
                    %                 num_of_top_k_matches = numel(rmatch.match(1,:));
                    %                 temp_idx=zeros(1,num_of_top_k_matches);
                    %
                    %                 for kk=1:num_of_top_k_matches
                    %                     if isempty(find(idx_valid(:)==idx_sort_conf(kk))) == 0
                    %                     temp_idx(kk)=idx_valid(find(idx_valid(:)==idx_sort_conf(kk)));
                    %                     end
                    %                 end
                    %                 idx_valid = temp_idx(temp_idx > 0);
                    
                    
                    % computing top k-matches
                    num_of_top_k_matches = numel(pmatch.match(1,:));
                    [~,idx_sort_conf]=sort(confidenceA,'descend');
                    idx_sort_conf=idx_sort_conf(1:num_of_top_k_matches);
                    
                    [Lia,Locb] = ismember(idx_valid,idx_sort_conf);
                    idx=sort(Locb,'ascend');
                    idx=idx(idx>0);
                    
                    idx_temp=zeros(1,num_of_top_k_matches);
                    if isempty(Lia) == 0
                        idx_temp(idx)=idx_sort_conf(idx);
                    end
                    idx_valid = idx_temp(idx_temp > 0);
                    
                    
                    IoU_threshold = 0.3;
                    match_cand = [ idx_for_active_opA; idx_for_opB ];
                    id_true = false(numel(viewA.idx2ori),1);
                    for l=1:numel(idx_valid)
                        li = idx_valid(l);
                        id_true(li) = IoU2GT(match_cand(1,li),match_cand(2,li)) <= IoU_threshold;
                    end
                    
                    match = [ 1:numel(max_id); max_id'];
                    
                    hFig_match = figure(1); clf;
                    imgInput = appendimages( viewA.img, viewB.img, 'h' );
                    imshow(rgb2gray(imgInput)); hold on; iptsetpref('ImshowBorder','tight');
                    
                    fprintf(' - Visualizing results... \n');
                    showColoredMatches(viewA.frame, viewB.frame, match(:,idx_sort_conf(1:num_of_top_k_matches)),...
                        confidenceA(idx_sort_conf(1:num_of_top_k_matches)), 'offset', [ size(viewA.img,2) 0 ], 'mode', strVisMode);
                    pause;
                    clf;
                    
                    fprintf('   correct match / active proposal (threshold %.2f): %03d/%03d\n\n',IoU_threshold, nnz(id_true), AP.num_op_active(imgA_idx));
                    
                    imshow(rgb2gray(imgInput)); hold on;
                    showColoredMatches(viewA.frame, viewB.frame, match(:,id_true),...
                        confidenceA(id_true), 'offset', [ size(viewA.img,2) 0 ], 'mode', strVisMode);
                    pause(0.1);
                    pause;
                    
                end
            end
        end
    end
    
    for fa = 1:numel(conf.algorithm)
        time(fa).avg = mean(time(fa).sec);
        time(fa).std = std(time(fa).sec);
    end
    save( fullfile(conf.timeDir, [ 'PF_time_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ] ), 'time');
end

