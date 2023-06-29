function data=transformdummy(raw_x)
% iterate over every attribute in x
% and transform categorical attributes to binary indicators
    d = size(raw_x,2);
    data = []; 
    for j=1:d
       tmp = unique(raw_x(:,j));
       if floor(tmp(1))==tmp(1)
           % is categorical
           vs = sort(tmp);
           for k=2:numel(vs)
               data = [data, raw_x(:,j)==vs(k)];
           end
       else
           data = [data, raw_x(:,j)];
       end
    end
end