function idx_cur = policy_uniform(idx_other, BATCH_SIZE)
    idx_cur = randsample(idx_other,BATCH_SIZE);
end