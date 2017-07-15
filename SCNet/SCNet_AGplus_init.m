function net = SCNet_AGplus_init(varargin)
addpath(genpath('../utils/fuseNet'))
net = SCNet_A_init();
net = dagnn.DagNN.loadobj(net);
mIdx = net.getParamIndex('cnv_fc_1f');
para_fc_f = net.params(mIdx).value;
mIdx = net.getParamIndex('cnv_fc_1b');
para_fc_b = net.params(mIdx).value;
net.layers = net.layers([1:25 28:52]);

%b1: FC1 +  L2 norm
cnv_fc_1 = dagnn.Conv('size',[7 7 512 2048],'pad',0,'stride',1,'hasBias',true);
net.addLayer('b1_cnv_fc_1', cnv_fc_1, {'b1_xRP'}, {'b1_FC1'},{'cnv_fc_1f','cnv_fc_1b'});
mIdx_1f = net.getParamIndex('cnv_fc_1f');
net.params(mIdx_1f).value = para_fc_f;
mIdx_1b = net.getParamIndex('cnv_fc_1b');
net.params(mIdx_1b).value = para_fc_b;
net.addLayer('b1_l2_norm1', dagnn.L2Norm(), {'b1_FC1'}, {'b1_L2NM_A'});

%b2: FC1 +  L2 norm
cnv_fc_1 = dagnn.Conv('size',[7 7 512 2048],'pad',0,'stride',1,'hasBias',true);
net.addLayer('b2_cnv_fc_1', cnv_fc_1, {'b2_xRP'}, {'b2_FC1'},{'cnv_fc_1f','cnv_fc_1b'});
mIdx_1f = net.getParamIndex('cnv_fc_1f');
net.params(mIdx_1f).value = para_fc_f;
mIdx_1b = net.getParamIndex('cnv_fc_1b');
net.params(mIdx_1b).value = para_fc_b;
net.addLayer('b2_l2_norm1', dagnn.L2Norm(), {'b2_FC1'}, {'b2_L2NM_A'});

%b1: FC2 +  L2 norm
cnv_fc_2 = dagnn.Conv('size',[7 7 512 2048],'pad',0,'stride',1,'hasBias',true);
net.addLayer('b1_cnv_fc_2', cnv_fc_2, {'b1_xRP'}, {'b1_FC2'},{'cnv_fc_2f','cnv_fc_2b'});
mIdx_2f = net.getParamIndex('cnv_fc_2f');
net.params(mIdx_2f).value = para_fc_f;
mIdx_2b = net.getParamIndex('cnv_fc_2b');
net.params(mIdx_2b).value = para_fc_b;
net.addLayer('b1_l2_norm2', dagnn.L2Norm(), {'b1_FC2'}, {'b1_L2NM_G'});

%b2: FC2 +  L2 norm
cnv_fc_2 = dagnn.Conv('size',[7 7 512 2048],'pad',0,'stride',1,'hasBias',true);
net.addLayer('b2_cnv_fc_2', cnv_fc_2, {'b2_xRP'}, {'b2_FC2'},{'cnv_fc_2f','cnv_fc_2b'});
mIdx_2f = net.getParamIndex('cnv_fc_2f');
net.params(mIdx_2f).value = para_fc_f;
mIdx_2b = net.getParamIndex('cnv_fc_2b');
net.params(mIdx_2b).value = para_fc_b;
net.addLayer('b2_l2_norm2', dagnn.L2Norm(), {'b2_FC2'}, {'b2_L2NM_G'});

%A: dot of A for appearance
net.addLayer('dot_A', dagnn.Dot(), {'b1_L2NM_A', 'b2_L2NM_A'}, {'AA_out'});
net.addLayer('relu_A', dagnn.ReLU(), {'AA_out'}, {'A_out'});
%A: dot of A for geo
net.addLayer('dot_G', dagnn.Dot(), {'b1_L2NM_G', 'b2_L2NM_G'}, {'g_AA_out'});
net.addLayer('relu_G', dagnn.ReLU(), {'g_AA_out'}, {'g_A_out'});
%G: geometric hough vote
vote = dagnn.ProposalHoughVote();
net.addLayer('vote', vote, {'g_A_out', 'b1_rois', 'b2_rois', 'b1_input', 'b2_input'}, {'G_out'});
%combine A and G
net.addLayer('dot2', dagnn.Dot2DWISE(), {'A_out', 'G_out'}, {'AG_out'});

%loss
net.addLayer('loss', dagnn.PFLoss(), {'AG_out','idx_for_active_opA', 'IoU2GT'}, 'objective');

plotNet(net);
[net.params([mIdx_1f mIdx_1b]).learningRate] = deal(0);
net.rebuild();
