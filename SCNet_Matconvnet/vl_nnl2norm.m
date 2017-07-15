function y = vl_nnl2norm(x, param, varargin)
% VL_L2NORM computes l2 normalization at each location
%
% Author: Subhransu Maji, Tsung-Yu Lin


backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

thresh = param(1);

gpuMode = isa(x, 'gpuArray');

[h, w, ch, bs] = size(x);
if gpuMode
    y = gpuArray(zeros([h, w, ch, bs], 'single'));
else
    y = zeros([h, w, ch, bs], 'single');
end

x_norm = sqrt(sum(x.*x, 3)+thresh);
if backMode
    E = bsxfun(@times, dzdy, x_norm.^(-1));
    F = sum(x.*dzdy,3);
    F = F.*x_norm.^(-3);
    F = bsxfun(@times, x, F);
    y = E-F;
else
    y = x./repmat(x_norm, [1, 1, ch, 1]);
end