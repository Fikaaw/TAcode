function plot_volatility(y, Variance, LongRunVar, period, numLags, var_name)
    figure('Name', ['GARCH-MIDAS Volatility - ' var_name]);
    nobs = size(y,1);
    seq = (period*numLags+1:nobs)';
    
    % Annualized volatility
    plot(seq, sqrt(252*Variance(seq)), 'g--', 'LineWidth', 1);
    hold on;
    plot(seq, sqrt(252*LongRunVar(seq)), 'b-', 'LineWidth', 2);
    title(['Volatility Components - ' var_name]);
    legend('Total Volatility', 'Secular Volatility', 'Location', 'SouthEast');
    xlabel('Observation');
    ylabel('Annualized Volatility');
    hold off;
end
