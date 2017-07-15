function y = vl_nnhoughvote(dotF, proposals_A, proposals_B, img_A, img_B, dzdy, varargin)
%by khan
useGPU = isa(dotF, 'gpuArray');
if useGPU
    dotF = gather(dotF);
    proposals_A = gather(proposals_A);
    proposals_B = gather(proposals_B);
    dzdy = gather(dzdy);
end

bbox_left = proposals_A(2:end, :)';
bbox_right = proposals_B(2:end, :)';    
viewA = vl_getView(bbox_left, img_A);
viewB = vl_getView(bbox_right, img_B);

sim_thresh = 0.4;
%voteA = dotF.*(dotF > sim_thresh);
voteA = max(dotF - sim_thresh, 0);
[ wVote, tableBina] = vl_GeoVote(viewA, viewB, voteA);
wVote = wVote + 0.01;
% tableBina = single(tableBina).*(dotF > sim_thresh);
% wVote = wVote.*(dotF > sim_thresh);
%% visualize Kernal matrix
% part_tableBina = tableBina(1:5000);
% num = numel(part_tableBina);
% vis_mat = false(num, num);
%     offSets = unique(part_tableBina);
%     if numel(offSets) > 1
%             offSets = offSets(2:end); 
%             for i = 1:num
%                 if part_tableBina(i) >0 
%                 vis_mat(i, :) = (part_tableBina == part_tableBina(i));
%                 end
%             end
%     end
%     imagesc(vis_mat);

%% estimate avg_k Kernal matrix
%
% part_tableBina = tableBina(:)';
% avg_K = zeros(size(tableBina));
%     offSets = unique(part_tableBina);
%     if numel(offSets) > 1
%             offSets = offSets(2:end); 
%             for i = 1:numel(offSets)
%                 selected = (tableBina == offSets(i));  
%                 s = mean(selected(:));
%                 avg_K(selected) = s;
%             end
%     end
    


if isempty(dzdy)
    y = wVote;
else
    y = zeros(size(dotF));
    offSets = unique( tableBina(dzdy~=0) );
%   tic
    if numel(offSets) > 1
            offSets = offSets(2:end); 
            for i = 1:numel(offSets)
                selected = (tableBina == offSets(i));
                selected = single(selected).*(voteA > 0);
                if numel(find(selected))> 0 
                    s = sum(sum(dzdy.*selected));
                    y(logical(selected)) = s;
                end
            end
    end
%   toc
    if useGPU
        y = gpuArray(single(y));
    end
end
end


