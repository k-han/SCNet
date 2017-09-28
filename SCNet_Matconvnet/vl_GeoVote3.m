function [ voteMap, tableValid, dLdF ] = vl_GeoVote3( viewA, viewB, voteA, varargin)
% Rafael Rezende, Inria - WILLOW
bVerbose = false;
bDeleteByAspect = true;
bFeature = 1;
thresAspect = 2.0;
wVoteFull = voteA;

if isempty(varargin)
    isBackward = false;
else
    isBackward = true;
    dzdy = varargin{1};
end
tic;
% aspect ratio
aTrain = viewA.frame(3,:)./viewA.frame(6,:);
aTest = viewB.frame(3,:)./viewB.frame(6,:);

tableValid = true(size(viewA.frame,2),size(viewB.frame,2));

if bVerbose
    fprintf('vote weight: max %f - min %f\n', max(wVoteFull(:)), min(wVoteFull(:)));
end

nFeatTrain = size(viewA.frame,2);
nFeatTest  = size(viewB.frame,2);

if bDeleteByAspect
    for j=1:nFeatTrain
        bValid = (aTest/aTrain(j) < thresAspect) & (aTest/aTrain(j) > 1/thresAspect);
        tableValid(j, ~bValid) = false;
    end
end

idxValid = find(tableValid>0 & wVoteFull>0);
[subValid1, subValid2] = ind2sub([nFeatTrain, nFeatTest], idxValid);

voteMap = zeros(nFeatTrain,nFeatTest,'single');
confidenceMap = zeros(nFeatTrain,nFeatTest,'single');
confidenceMap(idxValid) = wVoteFull(idxValid);

% offset-reg
IoU_threshold = 0;

offset_x = zeros(size(viewA.frame,2),size(viewB.frame,2));
offset_y = zeros(size(viewA.frame,2),size(viewB.frame,2));
offset_s = zeros(size(viewA.frame,2),size(viewB.frame,2));

offset_x(idxValid) = viewA.frame(1,subValid1) - viewB.frame(1,subValid2);
offset_y(idxValid) = viewA.frame(2,subValid1) - viewB.frame(2,subValid2);
offset_s(idxValid) = viewA.frame(3,subValid1).*viewA.frame(6,subValid1)./ (viewB.frame(3,subValid2).*viewB.frame(6,subValid2));

% derivative
if isBackward dLdF = zeros(nFeatTrain, nFeatTest, 'single'); else dLdF = []; end

%frame2box
boxA=frame2box(viewA.frame);
boxA_xywh = [boxA(1,:);boxA(2,:);boxA(3,:)-boxA(1,:)+1;boxA(4,:)-boxA(2,:)+1];
if (sum(boxA_xywh(3,:)<0) >0)
    boxA_xywh([1,3],:) = [boxA(3,:);boxA(1,:)-boxA(3,:)];
    boxA([1,3],:) = boxA([3,1],:);
end
if (sum(boxA_xywh(4,:)<0) >0)
    boxA_xywh([2,4],:) = [boxA(4,:);boxA(2,:)-boxA(4,:)];
    boxA([2,4],:) = boxA([4,2],:);
end
%boxA_xywh
boxA_xywh = max(boxA_xywh,1);
boxA_IoU=bboxOverlapRatio(boxA_xywh', boxA_xywh', 'Union');

boxA_IoU_idx=zeros(size(boxA,2),size(boxA,2));
boxA_IoU_idx(boxA_IoU>IoU_threshold)=1;



for i=1:nFeatTrain
    idx_local_neighborhood = find(boxA_IoU_idx(i,:)==1)';
    local_wVoteFull=confidenceMap(idx_local_neighborhood,:);
    nNeigh = size(idx_local_neighborhood, 1);
  
    [matching_conf, matching_idx]=max(local_wVoteFull,[],2);
    
    [global_matching_conf, ~]=max(confidenceMap,[],2);
   
    indices = sub2ind(size(offset_x), idx_local_neighborhood, matching_idx);
    local_offset_x_for_matching_point=offset_x(indices);
    local_offset_y_for_matching_point=offset_y(indices);
    local_offset_s_for_matching_point=offset_s(indices);

    local_offset = [local_offset_x_for_matching_point,...
        local_offset_y_for_matching_point,...
        local_offset_s_for_matching_point];
    
    G_xy = sqrt( (repmat(offset_x(i,:), [nNeigh,1])-repmat(local_offset(:,1), [1,nFeatTest])).^2 + (repmat(offset_y(i,:), [nNeigh,1])-repmat(local_offset(:,2), [1,nFeatTest])).^2);
    G_s = repmat(offset_s(i,:), [nNeigh,1]) - repmat(local_offset(:,3), [1,nFeatTest]);
    G = exp(-0.05.*G_xy-0.3.*abs(G_s));
    
    % updating matching score matrix
    determinant = sum(global_matching_conf);
    determinant = determinant*(determinant>0) + (determinant==0);
    voteMap(i,:) = matching_conf'*G/determinant;
    confidenceMap(i,:)=confidenceMap(i,:).*voteMap(i,:);
    

    % updating der
    if isBackward
        dLdF(indices) = G*dzdy(i,:)'/determinant;
    end
end
end

