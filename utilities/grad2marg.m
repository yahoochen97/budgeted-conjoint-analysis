function marg_effect=grad2marg(grad_mean)
    d = (size(grad_mean(:,1),1)+1)/3;
    marg_effect = zeros(2*d-1,1);
    marg_effect(1) = mean(grad_mean(1:2))*2;
    for j=2:d
        marg_effect(2*j-2) = grad_mean(3*j-3)/2+grad_mean(3*j-2)/2;
        marg_effect(2*j-1) = grad_mean(3*j-2)/2+grad_mean(3*j-1)/2;
    end
end