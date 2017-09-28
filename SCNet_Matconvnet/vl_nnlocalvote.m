function y = vl_nnlocalvote(dotF, proposals_A, proposals_B, img_A, img_B, dzdy, varargin)
%by Rafael Rezende
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

if isempty(dzdy)
    [ wVote, ~] = vl_GeoVote3(viewA, viewB, voteA);
    %wVote = wVote + 0.01;
    y = wVote;
else
    [ wVote, ~, dLdF ] = vl_GeoVote3(viewA, viewB, voteA, dzdy);
    %wVote = wVote + 0.01;

    y = dLdF;
    if useGPU
        y = gpuArray(single(y));
    end
end

