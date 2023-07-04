function mean = GeometricMean(S0, v0, theta, sigma, kappa, rho, r, T)
% r: Interest rate
% S0: Stock price at time 0
% v0: variance at time 0
% kappa: speed of mean reversion of v_t to its long-run average theta
% sigma: volatility of the variance process
% rho: Correlation between the two brownian motions
% T: Time to maturity

dt = 0.001; % Time step
time_grid = 0:dt:T; % We start with time 0

stock_values = ones(1, length(time_grid));
variance_values = ones(1, length(time_grid));

% Initialising the first values of the underlying stock and its variance.
stock_values(1) = S0;
variance_values(1) = v0;

% Generating random samples from multivariate normal distribution

mean_vector = [0 0]; % Since the mean of dW1 and dW2 is 0
covariance_matrix = [1, rho; rho, 1];

% `mvrnd` generates random samples from multivariate normal distribution

dZ = mvnrnd(mean_vector, covariance_matrix, length(time_grid)); 



% Simulating the heston model

tic;

for itr = 2:length(time_grid)
    dW1 = dZ(itr, 1);
    dW2 = dZ(itr, 2);

    [stock_values(itr), variance_values(itr)] = stock_value(r, stock_values(itr - 1), ...
                                                           variance_values(itr - 1), ...
                                                           kappa, theta, sigma, dW1, ...
                                                           dW2, dt);

end


toc;

mean = geomean(stock_values);



end




function [stock_price, variance_value] = stock_value(r, S_t, v_t, kappa, theta, sigma, dW1, dW2, dt)
%% Parameters:
% r: Interest rate
% S_t: Stock price at time t
% v_t: variance at time t
% kappa: speed of mean reversion of v_t to its long-run average theta
% sigma: volatility of the variance process
% dW1: Brownian motion value for stock movement
% dW2: Brownian motion value for variance movement
% dt: Time stamp
stock_price = S_t .* exp((r - 0.5 * v_t).*dt + sqrt(v_t .* dt) .* dW1);
variance_value = max(v_t + kappa .* (theta - v_t) .* dt + sigma .* sqrt(v_t .*dt) .* dW2, 0);
end


