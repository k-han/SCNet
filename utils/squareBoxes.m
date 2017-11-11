function [sqBoxes] = squareBoxes(boxes)
%SQUAREBOXES Summary of this function goes here
%   Detailed explanation goes here

sizes = boxes(:,[3 4]) - boxes(:,[1 2]);
whDiff = sizes(:,1) - sizes(:,2);

xAdd = round(max(-whDiff,0));
yAdd = round(max(whDiff,0));
sqBoxes = [boxes(:,1), boxes(:,2), boxes(:,3) + xAdd, boxes(:,4) + yAdd];

end