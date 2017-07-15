function [net, info] = SCNet_A(varargin)
%SCNet_A: train and eva SCNet_A

addpath(genpath('../utils/'));
run(fullfile('../utils', 'vlfeat-0.9.20','toolbox', 'vl_setup.m')) ;
run(fullfile(fileparts(mfilename('fullpath')),...
  '../..', 'matlab', 'vl_setupnn.m')); % Matconvnet path

dataset = 'PF-PASCAL-RP-500'; %dataset name
opts.batchNormalization = true ;
opts.imdbPath = fullfile('../data', [dataset '.mat']); %dataset path
opts.train = struct() ;
opts.train.gpus = [];
opts.train.expDir = fullfile('data',['exp_SCNet_A_' dataset]) ;%folder to save trained nets
opts.numGpus = numel(opts.train.gpus);
opts = vl_argparse(opts, varargin) ;


% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
net = SCNet_A_init();
imdb = load(opts.imdbPath);

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
trainfn = @SCNet_train;

[net, info] = trainfn(net, imdb, getBatch(opts), opts.train) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
image_mean = imdb.data.image_mean;
image_std = imdb.data.image_std;
images_A = imdb.data.images{batch,1} ;
images_A = (single(images_A) - image_mean)./image_std;
images_B = imdb.data.images{batch,2} ;
images_B = (single(images_B) - image_mean)./image_std;
proposals_A = imdb.data.proposals{batch,1};
proposals_B = imdb.data.proposals{batch,2};
proposals_A = [ones(size(proposals_A,1), 1).*1 proposals_A];
proposals_B = [ones(size(proposals_B,1), 1).*2 proposals_B];
idx_for_active_opA = imdb.data.idx_for_active_opA{batch,1};
IoU2GT = imdb.data.IoU2GT{batch,1};
if opts.numGpus > 0
  inputs = {'b1_input', gpuArray(im2single(images_A)), 'b2_input', gpuArray(im2single(images_B)), 'b1_rois', gpuArray(single(proposals_A')), 'b2_rois', gpuArray(single(proposals_B')), 'idx_for_active_opA', gpuArray(single(idx_for_active_opA)), 'IoU2GT', gpuArray(single(IoU2GT))} ;
else
  inputs = {'b1_input', im2single(images_A), 'b2_input', im2single(images_B), 'b1_rois', single(proposals_A'), 'b2_rois', single(proposals_B'), 'idx_for_active_opA', single(idx_for_active_opA), 'IoU2GT', single(IoU2GT)} ;
end