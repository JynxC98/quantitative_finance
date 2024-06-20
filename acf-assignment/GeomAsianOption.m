
%% Implementation of theorem 4.1
function result = GeomAsianOption(S0, v0, theta, sigma, kappa, rho, r, n, T, K)

%% Parameters
%S0: initial stock price;
%v0: initial stock volatility;
%theta: long run average of volatility;
%sigma: the volatility of volatility;
%kappa: rate of mean reversion;
%rho: correlation between two brownian motions;
%r: risk-free interest rate;
%n: number of terms in series expansions
%T: time to maturity;
%K: strike price.
%geometric_mean: Geometric mean of the option upto time T

geometric_mean = (1/T) .* GeometricMean(S0, v0, theta, sigma, kappa, rho, r, T);

exponent = exp .^ ((-r .* (T)) + geometric_mean); % Here t = 0
complex_function = (function_psi(1, 0, S0, v0, theta, sigma, kappa, rho, r, n, T, K) ...
                     - K) .* 0.5 + (1/pi) .* complexIntegral(S0, v0, theta, ...
                                                             sigma, kappa, ...
                                                             rho, r, n, T, K); 
result = exponent .* complex_function;
end


%% Taken from theorem 3.3 
function result = function_psi(s, w, S0, v0, theta, sigma, kappa, rho, r, n, T, K)


%% Extra parameters from above
%s, w : Complex numbers


% Parameters from a1 to a5. In this problem, we start from the timestamp of 
% t= 0
a1 = 2 .* v0/(sigma .^ 2);
a2 = 2 .* kappa * theta / (sigma .^ 2);
a3 = (log(S0) + (r .* sigma - (kappa .* theta .* rho)) .* power(T, 2))/(2 .* sigma .* T);
a4 = log(S0) - (rho/sigma) .* v0 + (r - (rho .* kappa .* theta)/sigma) .* T;
a5 = (kappa .* v0 + power(kappa, 2) .* theta .* T)/ power(sigma, 2);


% Here we add two extra variables to incorporate h_{-2} and h_{-1}

h_matrix=zeros(n+3,size(s,2)); %matrix for saving h_n's value, h-2=h-1=0
h_matrix(3,:)=1;               %h0=1,save in the 3rd entry
h_matrix(4,:)=T*(kappa-w*rho*sigma)/2; %h1, save in the 4th entry
% from h2 to hn:
nmat=linspace(1,n,n)';     % nx1 vector, number from 1 to n
A1=1./(4*nmat(2:end,1).*(nmat(2:end,1)-1)); %first part of h_n formula
A2=-s.^2*sigma^2*(1-rho^2)*T^2; %second part of h_n formula
A3=(s*sigma*T*(sigma-2*rho*kappa)-2*s*w*sigma^2*T*(1-rho^2)); %third part
A4=T*(kappa^2*T-2*s*rho*sigma-w*(2*rho*kappa-sigma)*sigma*T...≠
        -w^2*(1-rho^2)*sigma^2*T); %last part 
for j=5:(n+3) % compute h2 to hn, save from the 5th entry
    h_matrix(j,:)=A1(j-4 ,1) .* (A2.*h_matrix(j-4,:) + A3.*...
        (T*h_matrix(j-3,:)) +A4.* h_matrix(j-2,:));
end
% compute the H and H_tilde
H=sum(h_matrix(3:end,:),1); %value of H, sum from h0 to hn
h_tilde=(nmat/T).*h_matrix(4:end,:); %matrix for computing H_tilde
H_tilde=sum(h_tilde,1); %value of H_tilde, sum from h1 to hn
result =exp(-a1 .* (H_tilde./H)-a2 .* log(H)+a3 .*s + a4 .*w + a5); %return psi(s,w)
end



