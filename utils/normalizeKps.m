function [kpNum,kpCoords] = normalizeKps(kps,bbox,dims)
deltaX = (bbox(3)-bbox(1)+1)/dims(1);
deltaY = (bbox(4)-bbox(2)+1)/dims(2);

kpInds = 1:size(kps,1);
kps(:,1) = floor((kps(:,1)-bbox(1))/deltaX) + 1;
kps(:,2) = floor((kps(:,2)-bbox(2))/deltaY) + 1;
%goodInds = (kps(:,1)>0) & (kps(:,2) > 0) & (kps(:,1)<=dims(1)) & (kps(:,2) <=dims(2));
goodInds = true(size(kpInds)); %the gaussian thing below will filter automatically
kpNum = kpInds(goodInds);kpCoords = kps(goodInds,:);

end