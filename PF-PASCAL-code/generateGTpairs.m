run set_path;
run set_conf_PASCAL; %set number of proposals to 500 for traing and 1000 for testing.
run prepKP_PASCAL;
run ext_proposal_PASCAL;
run ext_active_proposal_PASCAL;
run makeGT_PASCAL;
run makeGT_combine_PASCAL;
run makeGT_normalize_PASCAL; %only for training data, comment it when generating testing data
run clean_PF_PASCAL;
run mirror_PF_PASCAL; %only for training data, comment it when generating testing data;
