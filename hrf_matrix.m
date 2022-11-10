function H = hrf_matrix(T, dx, TR)
    
    clc;
    clearvars -except T dx TR
    
    startpath='D:\Projects_local\PLRNN_new';
    patPLRNN=[startpath '\run_ssm\PLRNNreg_BOLD_SSM'];
    addpath(genpath(patPLRNN))

    hrf = spm_hrf(TR);

    l=1;
    for i=1:length(hrf)
        zhrf(l) = hrf(i); 
        l=l+1;       
        for j=2:dx
            zhrf(l)=0; 
            l=l+1;
        end
    end
    Hconv=convmtx(zhrf',dx*T);
    H=Hconv(1:T*dx,:);
    
    
    
end

