% script for evaluating dense flow field
% written by Bumsub Ham, Inria - WILLOW / ENS, Paris, France

function dense_flow_eva_PASCAL()

global conf;

%temp
conf.feature = conf.feature(1);

colorCode = makeColorCode(100);

% bins for plots
tbinv = linspace(0,1,101);

% load matching pair
load(fullfile(conf.datasetDir,'parsePascalVOC.mat'), 'PascalVOC');

%loop through features
for ft = 1:numel(conf.feature)
    
    avg_PCK = struct([]);
    for fa = 1:numel(conf.algorithm)
        avg_PCK(fa).method = func2str(conf.algorithm{fa});
        avg_PCK(fa).histo = zeros(numel(tbinv),1);
        avg_PCK(fa).color = colorCode(:,fa);
    end
    
    for ci = 1:length(conf.class)
        
        fprintf('Processing %s feature for %s...\n',conf.feature{ft},conf.class{ci});
        
        % load the annotation file
        load(fullfile(conf.benchmarkDir,sprintf('KP_%s.mat',conf.class{ci})), 'KP');
        
        % set matching pair
        classInd = pascalClassIndex(conf.class{ci});
        pair = PascalVOC.pair{classInd};
        
        % histogram for each class
        PCK = struct([]);
        for fa = 1:numel(conf.algorithm)
            PCK(fa).method = func2str(conf.algorithm{fa});
            PCK(fa).histo = zeros(numel(tbinv),1);
            PCK(fa).color = colorCode(:,fa);
        end
        
        for fi = 1:length(pair)
            
            fprintf('%03d/%03d\n', fi,length(pair));
            
            imgA_name = cell2mat(strcat(pair(fi,1)));
            imgA_idx = find(strcmp(KP.image_name,[imgA_name '.jpg']));
            
            annoA = KP.image2anno{imgA_idx};
            part_x_A = KP.part_x(:,annoA);
            part_y_A = KP.part_y(:,annoA);
            % delete invisible parts
            part_x_A = part_x_A(~isnan(part_x_A));
            part_y_A = part_y_A(~isnan(part_y_A));
            
            % compare it to other images
            imgB_name = cell2mat(strcat(pair(fi,2)));
            imgB_idx = find(strcmp(KP.image_name,[imgB_name '.jpg']));
            
            annoB = KP.image2anno{imgB_idx};
            part_x_B = KP.part_x(:,annoB);
            part_y_B = KP.part_y(:,annoB);
            % delete invisible parts
            part_x_B = part_x_B(~isnan(part_x_B));
            part_y_B = part_y_B(~isnan(part_y_B));
            
            for fa = 1:numel(conf.algorithm) % for each algorithm
                % load matching results
                load(fullfile(conf.flowDir,KP.image_dir{fi},conf.feature{ft},...
                    [ imgA_name '-' imgB_name...
                    '_' func2str(conf.proposal) '_' conf.feature{ft} '_' func2str(conf.algorithm{fa}) '.mat' ]), 'dmatch');
                
                PCK2GT = zeros(min(numel(part_x_A),numel(part_x_B)),1);
                
                vx=dmatch.vx;
                vy=dmatch.vy;
                
                for k=1:numel(PCK2GT)
                    px=round(part_x_A(k));
                    py=round(part_y_A(k));
                    
                    PCK2GT(k) = (part_x_A(k)+vx(py,px)-part_x_B(k))^2+(part_y_A(k)+vy(py,px)-part_y_B(k))^2;
                end
                PCK2GT=sqrt(PCK2GT);
                PCK2GT=PCK2GT./max(KP.bbox(3:4,imgB_idx) - KP.bbox(1:2,imgB_idx));
                
                bin_PCK = vl_binsearch(tbinv, double(PCK2GT));
                for p=1:numel(bin_PCK)
                    PCK(fa).histo(bin_PCK(p)) = PCK(fa).histo(bin_PCK(p)) + 1.0/numel(PCK2GT);
                end
            end
        end
        
        for fa = 1:numel(PCK)
            PCK(fa).histo = PCK(fa).histo ./ length(pair);
        end
        
        fprintf('%03d images processed.\n',length(pair));
        
        fileID = fopen(fullfile(conf.evaDFDir,[ conf.class{ci} '-' func2str(conf.proposal) '-' conf.feature{ft} '.txt']),'w');
        fprintf(fileID,'%s\n', conf.class{ci});
        
        fprintf(fileID,'\n%s\n', 'PCK (alpha=0.05)');
        for fa=1:numel(PCK)
            fprintf(fileID, '%.2f\n',max(cumsum(PCK(fa).histo(1:5))) );
        end
        fprintf(fileID,'\n%s\n', 'PCK (alpha=0.1)');
        for fa=1:numel(PCK)
            fprintf(fileID, '%.2f\n',max(cumsum(PCK(fa).histo(1:10))) );
        end
        fprintf(fileID,'\n%s\n', 'PCK (alpha=0.2)');
        for fa=1:numel(PCK)
            fprintf(fileID, '%.2f\n',max(cumsum(PCK(fa).histo(1:20))) );
        end
        fclose(fileID);
        
        save(fullfile(conf.evaDFDir, conf.class{ci},...
            [ conf.class{ci} '_' func2str(conf.proposal) '_' conf.feature{ft} '.mat' ]), 'PCK');
        
        %average
        for fa = 1:numel(conf.algorithm)
            avg_PCK(fa).histo = avg_PCK(fa).histo+PCK(fa).histo;
        end
    end
    
    for fa = 1:numel(conf.algorithm)
        avg_PCK(fa).histo = avg_PCK(fa).histo ./ length(conf.class);
    end
    
    fileID = fopen(fullfile(conf.evaDFavgDir,[ func2str(conf.proposal) '-' conf.feature{ft} '.txt']),'w');
    fprintf(fileID,'%s: %s\n', func2str(conf.proposal), conf.feature{ft});
    fprintf(fileID,'\n%s\n', 'PCK (alpha=0.05)');
    for fa=1:numel(avg_PCK)
        fprintf(fileID, '%.2f\n',max(cumsum(avg_PCK(fa).histo(1:5))) );
    end
    fprintf(fileID,'\n%s\n', 'PCK (alpha=0.1)');
    for fa=1:numel(avg_PCK)
        fprintf(fileID, '%.2f\n',max(cumsum(avg_PCK(fa).histo(1:10))) );
    end
    fprintf(fileID,'\n%s\n', 'PCK (alpha=0.2)');
    for fa=1:numel(avg_PCK)
        fprintf(fileID, '%.2f\n',max(cumsum(avg_PCK(fa).histo(1:20))) );
    end
    fclose(fileID);
    
    save(fullfile(conf.evaDFavgDir, [ func2str(conf.proposal) '_' conf.feature{ft} '.mat' ]), 'avg_PCK');
end

