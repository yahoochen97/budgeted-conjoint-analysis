% https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html

if strcmp(data_name,"twoDplane")
   % simulate profile data from 2D plane (categorical attributes)
   d = 5;
   raw_x = [2*randi([0 1],N,1)-1,randi([-1 1],N,d-1),2*randi([0 1],N,1)-1,randi([-1 1],N,d-1)];
   [transformed_x,dummy_flag] = transformdummy(raw_x);
   [pair_y, dgp_p, dgp_f, dgp_df, dgp_dy] = twoDplane(transformed_x);
elseif strcmp(data_name,"Friedman")
   % simulate profile data from Friedman (continuous attributes)
   d = 3;
   raw_x = [rand(N,d),rand(N,d)];
   [transformed_x,dummy_flag] = transformdummy(raw_x);
   [pair_y, dgp_p, dgp_f, dgp_df, dgp_dy] = Friedman(transformed_x);
end

% train/test data
train_x = transformed_x;
train_y = pair_y;
test_x = train_x;
if test_anchor
    for j=(size(test_x,2)/2+1):size(test_x,2)
        test_x(:,j) = min(test_x(:,j)); % anchoring point as base comparison
    end
    if strcmp(data_name,"twoDplane")
        [test_y, dgp_p, dgp_f, dgp_df, dgp_dy] = twoDplane(test_x);
    elseif strcmp(data_name,"Friedman")
        [test_y, dgp_p, dgp_f, dgp_df, dgp_dy] = Friedman(test_x);
    end
end