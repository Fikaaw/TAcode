function [params, logL, Variance, se, pvalues, condVol] = MGarchMidas(Y, X_mat, period, nlag)
    % MGARCH-MIDAS estimation with corrected Hessian computation
    % Inputs:
    %   Y: Target variable (returns)
    %   X_mat: Matrix of explanatory variables
    %   period: Sampling period
    %   nlag: Number of lags
    
    % Get dimensions of the problem
    nvar = size(X_mat, 3);
    nobs = numel(Y);
    nparams = 3 + nvar + 2;  % mu, alpha, beta, thetas, w, m
    
    % Initialize parameters with robust starting values
    mu0 = median(Y(:));      % More robust than mean
    alpha0 = 0.05;           % Conservative ARCH effect
    beta0 = 0.85;            % Slightly lower GARCH persistence
    theta0 = zeros(nvar,1);  % Start from neutral impact
    w0 = 2;                  % Conservative weighting
    m0 = var(Y(:));         % Sample variance as initial guess
    params0 = [mu0; alpha0; beta0; theta0; w0; m0];
    
    % Set parameter bounds with careful consideration
    lb = [-10*std(Y(:)); 1e-6; 1e-6; -0.99*ones(nvar,1); 1.001; 1e-6];
    ub = [10*std(Y(:)); 0.3; 0.999; 0.99*ones(nvar,1); 20; 10*var(Y(:))];
    
    % Configure optimization with more robust settings
    options = optimoptions('fmincon', ...
        'Algorithm', 'interior-point', ...
        'Display', 'iter', ...
        'MaxFunctionEvaluations', 2000, ...
        'MaxIterations', 2000, ...
        'OptimalityTolerance', 1e-6, ...
        'StepTolerance', 1e-8, ...
        'ConstraintTolerance', 1e-8, ...
        'HessianApproximation', 'bfgs', ...
        'SpecifyObjectiveGradient', false);
    
    % Objective function with scaling
    scale_factor = 1/nobs;
    objFun = @(p) -scale_factor * sum(MultivarLogLikelihood(p, Y, X_mat, period, nlag));
    
    try
        % Single optimization run with improved error handling
        [params, fval, exitflag, output, lambda, hessian] = ...
            fmincon(objFun, params0, [], [], [], [], lb, ub, [], options);
        
        % Verify optimization success
        if exitflag > 0
            % Compute numerical Hessian with corrected indexing
            hessian = computeNumericalHessian(objFun, params, nparams);
            
            % Ensure Hessian is well-conditioned
            if rcond(hessian) < eps^0.5
                % Add small regularization term if poorly conditioned
                hessian = hessian + eye(nparams) * sqrt(eps);
            end
            
            % Compute standard errors and statistics
            se = sqrt(diag(inv(hessian)));
            tStats = params ./ se;
            pvalues = 2 * (1 - tcdf(abs(tStats), nobs - length(params)));
            
            % Calculate final values
            [logL, Variance] = MultivarLogLikelihood(params, Y, X_mat, period, nlag);
            condVol = sqrt(Variance);
        else
            error('Optimization did not converge to a valid solution');
        end
        
    catch ME
        fprintf('Error details: %s\n', getReport(ME));
        rethrow(ME);
    end
end

function H = computeNumericalHessian(f, x, n)
    % Compute numerical Hessian with corrected indexing
    % f: objective function
    % x: parameter vector at which to evaluate Hessian
    % n: number of parameters
    
    H = zeros(n, n);
    h = sqrt(eps);  % Step size for finite differences
    fx = f(x);      % Function value at current point
    
    % Initialize temporary vectors for parameter perturbation
    xplus = x;
    xminus = x;
    
    % Compute diagonal elements first
    for i = 1:n
        % Compute diagonal elements using central differences
        xplus(i) = x(i) + h;
        xminus(i) = x(i) - h;
        
        fplus = f(xplus);
        fminus = f(xminus);
        
        % Second derivative approximation
        H(i,i) = (fplus - 2*fx + fminus) / (h^2);
        
        % Reset the vectors
        xplus(i) = x(i);
        xminus(i) = x(i);
    end
    
    % Compute off-diagonal elements
    for i = 1:n
        for j = (i+1):n
            % Compute mixed partial derivatives
            xpp = x;
            xpm = x;
            xmp = x;
            xmm = x;
            
            % Set perturbations for both parameters
            xpp(i) = x(i) + h;
            xpp(j) = x(j) + h;
            
            xpm(i) = x(i) + h;
            xpm(j) = x(j) - h;
            
            xmp(i) = x(i) - h;
            xmp(j) = x(j) + h;
            
            xmm(i) = x(i) - h;
            xmm(j) = x(j) - h;
            
            % Compute mixed partial derivative
            H(i,j) = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*h^2);
            H(j,i) = H(i,j);  % Symmetric matrix
        end
    end
    
    % Ensure symmetry (handle numerical errors)
    H = (H + H')/2;
end