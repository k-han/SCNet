function iou =  bbox_iou(boxes1, boxes2)
% by khan
% estimate iou of proposals in ulbr form
    x11 = boxes1(:,1);
    y11 = boxes1(:,2);
    x12 = boxes1(:,3);
    y12 = boxes1(:,4);

    x21 = boxes2(:,1);
    y21 = boxes2(:,2);
    x22 = boxes2(:,3);
    y22 = boxes2(:,4);

    xx1 = max(x11, x21);
    yy1 = max(y11, y21);
    xx2 = min(x12, x22);
    yy2 = min(y12, y22);

    area1 = (x12-x11+1) .* (y12-y11+1);
    area2 = (x22-x21+1) .* (y22-y21+1);
    
    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1);

    inter = w.*h;
    iou = inter ./ (area1 + area2 - inter);
end