function y = vl_nndot(inputs, dim, dzdy, varargin)
%by khan

% opts.inputSizes = [] ;
% opts = vl_argparse(opts, varargin, 'nonrecursive') ;
if nargin < 2, dim = 3; end;
if nargin < 3, dzdy = []; end;
useGPU = isa(inputs{1}, 'gpuArray');

inputSize_A = size(inputs{1});
inputSize_B = size(inputs{2});

if(numel(inputSize_A) == 3)
    inputSize_A = [inputSize_A, 1];
end

if(numel(inputSize_B) == 3)
    inputSize_B = [inputSize_B, 1];
end

if useGPU
    X1 = reshape(gather(inputs{1}), inputSize_A(3:4));
    X2 = reshape(gather(inputs{2}), inputSize_B(3:4));
    dzdy = gather(dzdy);
else
    X1 = reshape(inputs{1}, inputSize_A(3:4));
    X2 = reshape(inputs{2}, inputSize_B(3:4));
end
if isempty(dzdy)
y = X1'*X2;
if useGPU
    y = gpuArray(y);
end
else
    dzdx1 = zeros(size(X1));
    dzdx2 = zeros(size(X2));
    num = size(X2, 1);
    [n, k] = size(dzdy);
% 	tic
%    parpool(5)
    parfor i = 1:num
        dzdx1(i,:) = sum(dzdy.*repmat(X2(i,:),[n, 1]), 2)';
        dzdx2(i,:) = sum(dzdy.*repmat(X1(i,:)',[1, k]), 1);
    end
% 	toc
%     time_dot = end_t - start_t
%     num = size(X1, 1);
%     parfor i = 1:num
%         dzdx2(i, :) = sum(dzdy.*repmat(X1(i, :)',[1, k]), 1);
%     end

    y = cell(1, 2) ;
    if useGPU
    y{1} = gpuArray(reshape(single(dzdx1), inputSize_A));
    y{2} = gpuArray(reshape(single(dzdx2), inputSize_B));
    else
    y{1} = reshape(single(dzdx1), inputSize_A);
    y{2} = reshape(single(dzdx2), inputSize_B);
    end
end
