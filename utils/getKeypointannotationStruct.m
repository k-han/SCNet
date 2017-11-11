function [annot] = getKeypointannotationStruct(class,fnames)
%GETKEYPOINTANNOTATIONSTRUCT Summary of this function goes here
%   Detailed explanation goes here

globals;
var = load(fullfile(segkpAnnotationDir, class));
var = var.keypoints;
if(nargin > 1)
    goodInds = ismember(var.voc_image_id,fnames);
else
    goodInds = true(size(var.voc_image_id));
end

annot.bounds = var.bbox(goodInds,:);
annot.img_name = var.voc_image_id(goodInds);
annot.coords = permute(var.coords(goodInds,:,:),[2 3 1]);
annot.class = class;
annot.kps_labels = var.labels;

end

