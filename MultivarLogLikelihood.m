function [logL, Variance] = MultivarLogLikelihood(params, Y, X_mat, period, nlag)
    % Extract parameters
    mu = params(1);
    alpha = params(2);
    beta = params(3);
    theta = params(4:end-2);
    w = params(end-1);
    m = params(end);

    % Get dimensions
    [ndays, nmonths, nvars] = size(X_mat);
    
    % Initialize outputs
    logL = zeros(nmonths * period, 1);
    Variance = zeros(nmonths * period, 1);
    
    % Calculate MIDAS weights
    weights = midasWeights(w, nlag);
    
    % Calculate monthly component (tau)
    tau = zeros(nmonths, 1);
    
    % Loop through months
    for t = 1:nmonths
        tau_temp = 0;  % Initialize temporary tau value
        
        % Sum over macro variables and lags
        for k = 1:nvars
            for j = 1:min(nlag, t)
                if t-j+1 > 0
                    tau_temp = tau_temp + theta(k) * weights(j) * X_mat(1,t-j+1,k);
                end
            end
        end
        
        tau(t) = m * exp(tau_temp);  % Apply exponential and multiply by m
    end
    
    % Calculate daily variances and log-likelihood
    h = zeros(nmonths * period, 1);
    returns = Y(:);
    valid_idx = ~isnan(returns);
    
    % Initialize first variance
    h(1) = tau(1);
    
    % GARCH recursion
    for t = 2:length(returns)
        if valid_idx(t)
            curr_month = ceil(t/period);
            if curr_month > nmonths
                break;
            end
            
            h(t) = tau(curr_month) * (1 - alpha - beta) + ...
                   alpha * (returns(t-1) - mu)^2 + ...
                   beta * h(t-1);
            
            % Calculate log-likelihood
            if h(t) > 0
                logL(t) = -0.5 * log(2*pi) - 0.5 * log(h(t)) - ...
                          0.5 * (returns(t) - mu)^2 / h(t);
            else
                logL(t) = -1e10;  % Penalty for negative variance
            end
        end
    end
    
    Variance = h;
end

function weights = midasWeights(w, nlag)
    % Calculate beta weights for MIDAS
    weights = zeros(nlag, 1);
    for k = 1:nlag
        weights(k) = betaWeight(k/nlag, w);
    end
    % Normalize weights
    weights = weights / sum(weights);
end

function w = betaWeight(x, param)
    % Beta weighting function
    w = x^(param-1) * (1-x)^(param-1);
end