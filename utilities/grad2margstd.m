function marg_effect_std=grad2margstd(grad_std)
    d = (size(grad_std(:,1),1)+1)/3;
    marg_effect_std = zeros(2*d-1,1);
    marg_effect_std(1) = 2*sqrt(sum(grad_std(1:2).^2));
    for j=2:d
        marg_effect_std(2*j-2) = sqrt(grad_std(3*j-3)^2+grad_std(3*j-2)^2)/2;
        marg_effect_std(2*j-1) = sqrt(grad_std(3*j-2)^2+grad_std(3*j-1)^2)/2;
    end
end