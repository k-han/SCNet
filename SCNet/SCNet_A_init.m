function netFused = SCNet_A_init(varargin)
addpath(genpath('../utils/fuseNet'))
opts.modelPath = fullfile('../data', 'models','imagenet-vgg-verydeep-16.mat');
opts = vl_argparse(opts, varargin);
display(opts);

% Load an imagenet pre-trained cnn model.
net = load(opts.modelPath);
net = vl_simplenn_tidy(net);
pool4 = find(cellfun(@(a) strcmp(a.name, 'pool4'), net.layers)==1);
net.layers = net.layers(1:pool4);
% Convert to DagNN.
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

%add roipooling layer
net.addLayer('roipool', dagnn.ROIPooling('method','max','transform',1/16,...
    'subdivisions',[7,7],'flatten',0), ...
    {net.layers(pool4).outputs{1},'rois'}, 'xRP');

cnv_fc_1 = dagnn.Conv('size',[7 7 512 2048],'pad',0,'stride',1,'hasBias',true);
net.addLayer('cnv_fc_1', cnv_fc_1, {'xRP'}, {'FC1'},{'cnv_fc_1f','cnv_fc_1b'});
mIdx = net.getParamIndex('cnv_fc_1f');
net.params(mIdx).value = 0.01 * randn(7,7,512,2048,'single');
mIdx = net.getParamIndex('cnv_fc_1b');
net.params(mIdx).value = zeros(2048, 1, 'single');

%L2 norm
net.addLayer('l2_norm', dagnn.L2Norm(), {'FC1'}, {'L2NM'});

netStruct=net.saveobj;
netStructB1 = netNamePrefix(netStruct,'b1_','b1_','');
netStructB2 = netNamePrefix(netStruct,'b2_','b2_','');
netStructFused=fuseNetStruct(netStructB1,netStructB2);
netFused=dagnn.DagNN.loadobj(netStructFused);

%A: appearance
netFused.addLayer('dot', dagnn.Dot(), {'b1_L2NM', 'b2_L2NM'}, {'AA_out'});
netFused.addLayer('relu_A', dagnn.ReLU(), {'AA_out'}, {'A_out'});

%loss
netFused.addLayer('loss', dagnn.PFLoss(), {'A_out','idx_for_active_opA', 'IoU2GT'}, 'objective');

% plotNet(netFused);
[netFused.params(1:20).learningRate] = deal(0);
netFused.rebuild();
