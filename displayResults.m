function displayResults(params, se, pvalues, logL, Variance, Y, X_mat)
    % Calculate variance ratios for each macro variable
    nvar = size(X_mat, 3);
    var_ratios = zeros(nvar, 1);
    total_variance = var(Y(:));
    
    % Calculate individual contributions using theta coefficients
    for i = 1:nvar
        theta_i = params(3+i);
        X_i = squeeze(X_mat(:,:,i));
        component_var = var(theta_i * X_i(:));
        var_ratios(i) = component_var / total_variance;
    end
    
    disp('MGARCH-MIDAS Model Results:')
    disp('===========================')
    disp('Parameter Estimates and Variable Contributions:')
    disp('-----------------------------------------')
    
    % Display first three parameters (mu, alpha, beta)
    paramNames = {'mu', 'alpha', 'beta'};
    for i = 1:3
        fprintf('%s: %.4f (SE: %.4f, p-value: %.4f)', paramNames{i}, params(i), se(i), pvalues(i));
        if pvalues(i) < 0.01
            fprintf(' ***\n');
        elseif pvalues(i) < 0.05
            fprintf(' **\n');
        elseif pvalues(i) < 0.1
            fprintf(' *\n');
        else
            fprintf('\n');
        end
    end
    
    % Display theta parameters with their variance ratios
    fprintf('\nMacro Variable Contributions:\n');
    fprintf('---------------------------\n');
    for i = 1:nvar
        fprintf('theta%d: %.4f (SE: %.4f, p-value: %.4f)', i, params(3+i), se(3+i), pvalues(3+i));
        if pvalues(3+i) < 0.01
            fprintf(' ***');
        elseif pvalues(3+i) < 0.05
            fprintf(' **');
        elseif pvalues(3+i) < 0.1
            fprintf(' *');
        end
        fprintf('\nVariance Contribution: %.2f%%\n\n', var_ratios(i)*100);
    end
    
    % Display w and m parameters without using paramNames
    for i = (4+nvar):length(params)
        paramType = (i == 4+nvar) * 'w' + (i == 5+nvar) * 'm';
        fprintf('%s: %.4f (SE: %.4f, p-value: %.4f)', paramType, params(i), se(i), pvalues(i));
        if pvalues(i) < 0.01
            fprintf(' ***\n');
        elseif pvalues(i) < 0.05
            fprintf(' **\n');
        elseif pvalues(i) < 0.1
            fprintf(' *\n');
        else
            fprintf('\n');
        end
    end
    
    % Calculate and display model fit statistics
    nParams = length(params);
    nObs = length(Y(:));
    aic = -2*sum(logL) + 2*nParams;
    bic = -2*sum(logL) + log(nObs)*nParams;
    explained_variance = var(sqrt(Variance));
    total_var_ratio = explained_variance / total_variance;
    
    fprintf('\nModel Fit Statistics:\n');
    fprintf('-------------------\n');
    fprintf('Log-Likelihood: %.4f\n', sum(logL));
    fprintf('AIC: %.4f\n', aic);
    fprintf('BIC: %.4f\n', bic);
    fprintf('Total Model Variance Ratio: %.2f%%\n\n', total_var_ratio*100);
    
    % Display interpretation
    fprintf('Interpretation:\n');
    fprintf('-------------\n');
    fprintf('Most influential macro variables:\n');
    [sorted_ratios, idx] = sort(var_ratios, 'descend');
    for i = 1:nvar
        fprintf('Variable %d: %.2f%% of total variance\n', idx(i), sorted_ratios(i)*100);
    end
    
    fprintf('\nSignificance levels:\n');
    fprintf('*** p<0.01, ** p<0.05, * p<0.1\n');
end