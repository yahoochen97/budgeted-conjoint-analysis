function [data, dummy_flag]=transformdummy(raw_x)
% iterate over every attribute in x
% and transform categorical attributes to binary indicators
    d = size(raw_x,2);
    data = []; dummy_flag = [];
    for j=1:d
       tmp = unique(raw_x(:,j));
       if floor(tmp(1))==tmp(1)
           % is categorical
           vs = sort(tmp);
           for k=2:numel(vs)
               data = [data, raw_x(:,j)==vs(k)];
               dummy_flag = [dummy_flag, 1];
           end
       else
           data = [data, raw_x(:,j)];
           dummy_flag = [dummy_flag, 0];
       end
    end
end