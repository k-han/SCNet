function Y = vl_nnpfloss(X,idx_for_active_opA,IoU2GT,dzdy,varargin)

opts.instanceWeights = [] ;
opts.classWeights = [] ;
opts.threshold = 0 ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;
useGPU = isa(idx_for_active_opA, 'gpuArray');

if useGPU
    X = gather(X);
    idx_for_active_opA = gather(idx_for_active_opA);
    IoU2GT = gather(IoU2GT);
end

inputSize = [size(X,1) size(X,2)] ;

% --------------------------------------------------------------------
% Do the work
% --------------------------------------------------------------------

%% positive 
        leftp_num = inputSize(1);
        
%% get product similarity and boxes
        A = X(idx_for_active_opA,:);

%% get the IOU matrix, note IoU2GT is actually 1 - IoU.  
          iou_mat = 1 - IoU2GT;
          
%% positive pairs
          pos_iou_thresh = 0.5;
          POS_FLAG = (iou_mat > pos_iou_thresh);  %mxn matrix
          NUM_POS = sum(POS_FLAG,2); %mx1 vector
%% combine IOU matrix and A (i.e. appearance similarity) to find negative pairs
          neg_iou_thresh = 0.2;
          NEG_FLAG = zeros(size(iou_mat));
          neg_A = A;
          neg_A(iou_mat > neg_iou_thresh) = -inf;
          [neg_A_sorted, neg_idx_cand] = sort(neg_A, 2, 'descend');
          for i = 1:size(iou_mat,1)
              net_num_temp = NUM_POS(i);
              if net_num_temp > 0
              neg_idx_temp = neg_idx_cand(i, 1:net_num_temp);
              NEG_FLAG(i,neg_idx_temp) = -1;
              end
          end

        
%% combine pos and neg
        m = 1;% margin
        N_loss = sum(NUM_POS)*2;
        POS_VAL = m.*POS_FLAG - POS_FLAG.*A;
        NEG_VAL = m.*abs(NEG_FLAG) - NEG_FLAG.*A;

if nargin <= 3 || isempty(dzdy)
%%  estimate loss
    if N_loss > 0
    t = max(0, POS_VAL) + max(0, NEG_VAL);
    Y = sum(t(:))/(N_loss);
    else
        Y = 0;
    end
else
%%  estimate derivative
    if N_loss > 0
       dLdF_active = (POS_VAL > 0).*(-1);
       dLdF_active = dLdF_active + (NEG_VAL > 0);
       dLdF = zeros(inputSize);
       dLdF(idx_for_active_opA, :) = dLdF_active;
       Y = single(dLdF)./N_loss;
       Y = dzdy.*Y;
    else
       Y = single(zeros(inputSize));
    end
    if useGPU
        Y = gpuArray(Y);
    end
end
end