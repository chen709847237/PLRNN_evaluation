clear all;
clc;

startpath='/home/jiarui.chen/Project/PLRNN_new';        % define current folder path     
addpath(genpath(startpath))

patPLRNN=[startpath '\run_ssm\PLRNNreg_BOLD_SSM'];      % define PLRNN-BOLD-SSM model directory
addpath(genpath(patPLRNN))
patLDS=[startpath '\run_ssm\LDS_BOLD_SSM'];             % define LDS-BOLD-SSM model directory
addpath(genpath(patLDS))

model_storage_root_path = '/home/jiarui.chen/zi-flstorage/jiarui.chen/PLRNN/';

% input model root directory  
%input_path = [model_storage_root_path 'dataset_resample_para_search_HC'];
input_path = [model_storage_root_path 'dataset_resample_para_search_SCZ'];

% output model root directory 
%output_path = [model_storage_root_path 'dataset_resample_para_search_HC_4python'];
output_path = [model_storage_root_path 'dataset_resample_para_search_SCZ_4python'];

% % list all model files
folder_list = dir([input_path]);
for i = 1:length(folder_list)
    if(isequal(folder_list(i).name, '.')||isequal(folder_list(i).name, '..')||~folder_list(i).isdir)
        continue;
    end
    
    folder_name = folder_list(i).name;

    input_folder_path = [input_path '/' folder_name];
    disp(['FOLDER >>>> ' folder_name]);
    
    output_folder_path = [output_path '/' folder_name];
    if ~exist("output_folder_path",'dir')
       mkdir(output_folder_path);                
    end 

    input_model_file_list = dir(input_folder_path);
    input_model_file_num = length(input_model_file_list);
    input_model_file_count = 1;
    for j = 1:input_model_file_num
        if(isequal(input_model_file_list(j).name, '.')||isequal(input_model_file_list(j).name, '..'))
            continue;
        end

        input_model_file_name = input_model_file_list(j).name;
        disp(['#' num2str(input_model_file_count) ' FILE >>>> ' input_model_file_name]);
        input_model_file_count = input_model_file_count + 1;
        input_model_file_path = [input_folder_path '/' input_model_file_name];

        file_name_short = input_model_file_name(1:end-4);
        % >>> read general model info from file name
        file_name_split = strsplit(file_name_short, '_');
        % smaple id
        sample_id = file_name_split{2};
        norm_method = file_name_split{3};
        % undersampleing resample TR value
        resample_tr = strsplit(file_name_split{4}, 're'); 
        resample_tr = str2num(resample_tr{2});
        % ROI number   
        roi_num = strsplit(file_name_split{5}, 'X');
        roi_num = str2num(roi_num{2});
        % latent z dim
        latent_z_dim = strsplit(file_name_split{6}, 'Z');
        latent_z_dim = str2num(latent_z_dim{2});
        % lamda vlaue  
        lamda = strsplit(file_name_split{7}, 'lam');
        lamda = str2num(lamda{2});
        % repeate versions of model using the same hyperparameter
        rep_ver = file_name_split{8};
    
        %load trained PLRNN model
        dat = load(input_model_file_path);
        
        Ezi = dat.Ezi;                        % model parameter Ezi 8x137
        M = size(Ezi,1);                      % M--> latent state space dimensioin  
        X = dat.X;                            % fMRI-BOLD data 12x137
        [q,T] = size(X);                      
        R = dat.data_PLRNNv.rp;               % R nuiscance effects data 137x6
        cov = dat.data_PLRNNv.cov;            % covariate as nuisance regressor 137x2
        R(:,end+1:end+size(cov,2)) = cov;     % merge 137x(6+2)=137x8 
        regions = dat.data_PLRNNv.ROI_label_s;
        
        % model object：net
        net = dat.net;          
        
        mu0 = net.mu0;            %initialization parameter of latent state 8x1
        W = net.W;                % model parameter W 8x8
        A = net.A;                % model parameter A 8x8
        h = net.h;                % model parameter h 8x1
        B_est = net.B;            % model parameter B 12x8
        M_est = net.J;            % model parameter J 12x8 
        G_est = net.Gamma;        % model parameter Gamma 12x12
        H = net.getConvolutionMtx(M); % define hrf convolution function H (137*8)x(137*8) 1096*1096
    
       
        % Free run result in MATLAB as calculate ref in python
        Z_fr = getNStepPred(Ezi, T, A, W, h,mu0);    
        %Z_fr = getNStepPred(Ezi, 5, A, W, h,mu0);
        
        Z_fr_1d = reshape(Z_fr,1,numel(Z_fr));
        hZ_fr = H * Z_fr_1d';   
        hZ_fr = reshape(hZ_fr,M,T);             
        X_fr_pre = B_est * hZ_fr + M_est * R';      % predicted signal !!!
      
        
        save([output_folder_path '/' input_model_file_name], 'sample_id', 'norm_method', 'resample_tr', 'roi_num', 'latent_z_dim', 'lamda', 'rep_ver',...
                                 'Ezi', 'M', 'X', 'q', 'T', 'R', 'cov', 'regions',...
                                 'mu0', 'W', 'A', 'h', 'B_est', 'M_est', 'G_est', 'H', 'Z_fr', 'X_fr_pre');
                                       
                             
        clear sample_id norm_method resample_tr roi_num latent_z_dim lamda rep_ver
        clear Ezi M X q T R cov regions mu0 W A h B_est M_est G_est H Z_fr X_fr_pre
        clear net Z_fr_1d dat hZ_fr

        % FOR TEST 
        %break

    end

    % FOR TEST 
    %break

end


