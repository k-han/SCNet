function kps = normKps(kps,bbox,dims)
%kps is (N*Nkps) X 2

deltaX = (bbox(:,3)-bbox(:,1)+1)/dims(1);
deltaY = (bbox(:,4)-bbox(:,2)+1)/dims(2);

kps(:,1) = ((kps(:,1)-bbox(:,1))./deltaX);
kps(:,2) = ((kps(:,2)-bbox(:,2))./deltaY);

badInds = isnan(kps(:,1)) |  isnan(kps(:,2)) | (kps(:,1))<0 | kps(:,2) < 0 | kps(:,1) >= dims(1) | kps(:,2) >= dims(2) ;
kps(badInds,:) = nan;

end