function boxes =  overlappingBoxes(box,imgDims)
deltaX = (box(3)-box(1))/6;
deltaY = (box(4)-box(2))/6;
boxes = [];
for x1Shift = -1:1
    for y1Shift = -1:1
        for x2Shift = -1:1
            for y2Shift = -1:1
                boxes(end+1,:) = [box(1)+x1Shift*deltaX box(2)+y1Shift*deltaY box(3)+x2Shift*deltaX box(4)+y2Shift*deltaY];
            end
        end
    end
end
boxes = round(boxes);
%boxes(:,1) = max(boxes(:,1),1);
%boxes(:,2) = max(boxes(:,2),1);
%boxes(:,3) = min(boxes(:,3),imgDims(2));
%boxes(:,4) = min(boxes(:,4),imgDims(1));
boxes = unique(boxes,'rows');
if(sum((boxes(:,1)>boxes(:,3)) | (boxes(:,2)>boxes(:,4))))
    disp('oops')
end
goodInds = boxes(:,1)<imgDims(2) & boxes(:,2)<imgDims(1);
boxes = boxes(goodInds,:);

end