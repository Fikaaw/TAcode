function weights = midasWeights(nlag, w)
    x = (0:nlag-1)' / (nlag-1);
    weights = (1-x.^w).^w;
    weights = weights / sum(weights);
end
