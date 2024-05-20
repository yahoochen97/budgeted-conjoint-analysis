if ~exist('data_name','var')
    data_name =  "hainmueller_immigrant"; %  "hainmueller_candidate"; %
end

maxNumCompThreads(1);

% add gpml tool box; clear env;
close all;
% add gpml path
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=16;
rng(12345);

file_name = "./data/" + data_name + ".csv";
opts = detectImportOptions(file_name);
data = readtable(file_name, opts);

raw_x = data{:,1:(end-1)};
d = size(raw_x,2)/2;
train_x = transformdummy(raw_x);
train_y = 2*data{:,end}-1;
test_x = train_x;

diff_in_mean;

% build a gp preference learning model for grad
learn_HYP = 1;
n_gauss_hermite = 10;
gp_pref_grad;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gp preference learning GMM effect
[gp_GMM_mu,gp_GMM_std]=gp_AMCE(mu_GMM_avg,sigma_GMM_avg, data_name, train_x);
gp_GMM_mu = gp_GMM_mu * std(dim_mu)/ std(gp_GMM_mu); 
gp_GMM_std = gp_GMM_std * std(dim_mu)/ std(gp_GMM_mu); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = size(train_x,2)/2;
results = array2table(zeros(2*D,2),'VariableNames',...
    {'mean','std'});
results.model = cell(2*D,1);

results(1:D,1) = num2cell(dim_mu)';
results(1:D,2) = num2cell(dim_std)';
results(1:D,3) = {'diffinmean'};

results((1*D+1):(2*D),1) = num2cell(gp_GMM_mu)';
results((1*D+1):(2*D),2) = num2cell(gp_GMM_std)';
results((1*D+1):(2*D),3) = {'gpGMM'};

writetable(results,"./results_application/"+data_name+".csv");

dim_flag = sign(dim_mu);
dim_flag(sign((dim_mu+1.96*dim_std))~=sign((dim_mu-1.96*dim_std))) = 0;
gp_GMM_flag = sign(gp_GMM_mu);
gp_GMM_flag(sign((gp_GMM_mu+1.96*gp_GMM_std))~=sign((gp_GMM_mu-1.96*gp_GMM_std))) = 0;

if data_name ==  "hainmueller_candidate"
   attributes = ["Served", "Jewish", "Catholic", "Mainline protestant", "Evangelical protestant", ...
       "Mormon", "Baptist college", "Community college", "State university", "Small college", "Ivy League university",...
       "Lawyer", "Doctor", "High school teacher", "Farmer", "Car dealer", "Female", ...
       "54K", "65K", "92K", "210K", "5.1M", "Native American", "Black", "Hispanic", ...
       "Caucasian", "Asian American", "45", "52", "60", "68", "75"];
else
    attributes = ["male", "4th grade", "8th grade", "high school", "college degree",...
        "graduate degree", "broken English", "tried English but unable", "used interpreter",...
        "Germany", "France", "Mexico", "Philippines", "Poland", "India", "China", "Sudan", "Somalia",...
        "Iraq", "waiter", "child care provider", "gardener", "financial analyst", "construction worker",...
        "teacher", "computer programmer", "nurse", "research scientist", "doctor",...
        "1?2 years", "3?5 years", "5+ years", "interviews with employer", "will look for work",...
        "no plans to look for work", "seek better job", "escape persecution", "once as tourist",...
        "many times as tourist", "six months with family", "once w/o authorization"];
end

% for i=find(dim_flag~=gp_GMM_flag)
%     if dim_flag(i), dim_effect = ""; else, dim_effect = "null"; end
%     if gp_GMM_flag(i), gp_effect = ""; else, gp_effect = "null"; end
%     fprintf("%s %s effect dim: %.3f, %s gp: %.3f\n", attributes(i), dim_effect,...
%         dim_mu(i), gp_effect, gp_GMM_mu(i));
% end

for i=[-1,0,1]
   for j=[-1,0,1]
      if i==j
         fprintf("dim: %d, gp: %d\n", i, j);
         disp(attributes(find((dim_flag==i) & (gp_GMM_flag==j))));
      end
   end
end

% interactive effect
% if data_name ==  "hainmueller_candidate"
%     % High school teacher, Farmer, Native American, Black, Caucasian,
%     % 45, 75
%    for i=[14,15, 23, 24, 26, 28,32]
%        
%    end
% end