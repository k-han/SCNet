function [ viewInfo ] = vl_getView(boxes, img)
% load image and make view info
imgSz = size(img);
conf = [];
cand = 1:size(boxes,1);

if isempty(cand)
    bValid1 = boxes(:,1) > imgSz(2)*0.01 & boxes(:,3) < imgSz(2)*0.99 ...
        & boxes(:,2) > imgSz(1)*0.01 & boxes(:,4) < imgSz(1)*0.99;
    bValid2 = boxes(:,1) < imgSz(2)*0.01 & boxes(:,3) > imgSz(2)*0.99 ...
        & boxes(:,2) < imgSz(1)*0.01 & boxes(:,4) > imgSz(1)*0.99;
    idxValid = find(bValid1 | bValid2);
else
    idxValid = cand;
end

boxes = boxes(idxValid,:);

viewInfo.idx2ori = int32(idxValid); % current index to original index

viewInfo.frame = box2frame(boxes');

viewInfo.imgSz = imgSz;    
viewInfo.bbox = [ 1, 1, imgSz(2), imgSz(1) ]';