%by khan

function combine_proposal_PASCAL()
global conf;

bShowOP = false;
proposals = [];
for ci = 1:numel(conf.class)
    
    % load the annotation file
    load(fullfile(conf.benchmarkDir,sprintf('KP_%s.mat',conf.class{ci})), 'KP');
    nImage = length(KP.image_name);
    
    classInd = pascalClassIndex(conf.class{ci});
    pair = PascalVOC.pair{classInd};
    
    pair_num = length(pair);
    data.images = cell(pair_num,2);
    data.proposals = cell(pair_num,2);
    % loop through images
    for i = 1:length(pair)
        if exist('modk') && exist('modv') && mod(i, modv) ~= modk
            continue;
        end
        image_name = fullfile(conf.imageDir,KP.image_name{i});
        img = imread(image_name);
               
        % load extracted object proposals
        load(fullfile(conf.proposalDir,KP.image_dir{i},[ KP.image_name{i}(1:end-4) '_' func2str(conf.proposal) '.mat' ]), 'op');
        
        proposals = [proposals; op.coords];
%         op.coords=proposal; % (x,y) coordinates ([col,row]) for left-top and right-bottom points
%         op.scores=scores;   % objectness score
 
        % =========================================================================
        % show extracted object proposasls
        % =========================================================================
        if bShowOP
            img=imread(fullfile(conf.imageDir,KP.image_name{i}));
            clf; imshow(rgb2gray(img)); hold on;
            op.coords=op.coords';
            colorCode = makeColorCode(length(op.coords));
            for k=1:length(op.coords)
                drawboxline(op.coords(:,k),'LineWidth',4,'color',colorCode(:,k));
            end
            pause;
        end
        
    end
end

save(fullfile(conf.proposalDir,'OP-PF-PASCAL.mat'), 'proposals')

