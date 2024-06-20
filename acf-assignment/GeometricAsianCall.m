function call_option = GeometricAsianCall(S0, v0, theta, sigma, kappa, rho, r, n, T, K)

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

%% Pricing formula of theorem 4.1
% geometric_mean = (1/T) .* GeometricMean(S0, v0, theta, sigma, kappa, rho, r, T);
call_option = (exp(-r .* T) * psi(1, 0, S0, v0, theta, sigma, kappa, rho, r, n, T) ...
    - K) .* 0.5 + (1/pi) .* GeometricIntegral(S0, v0, theta, sigma, kappa, rho, r, n, T, K);

end

function value = GeometricIntegral(S0, v0, theta, sigma, kappa, rho, r, n, T, K)
% Calculates the integral value of theorem 4.1
value = integral (@(x) Integrand(x, S0, v0, theta, sigma, kappa, rho, r, n, T, K), ...
                  0, 10e5);
end

function value = Integrand(x, S0, v0, theta, sigma, kappa, rho, r, n, T, K)
% Calculates the value of integrand of theorem 4.1
A=psi(1+1i*x,0,S0,v0,theta,sigma,kappa,rho,r,n,T);
B=psi(x*1i,0,S0,v0,theta,sigma,kappa,rho,r,n,T);
C=exp(-1i*x*log(K))./(1i*x);
value=real((A-K.*B).*C); %return the real part only
end

%=====================subfunction: Psi(s,w)=============================
function ret=psi(s,w,S0,v0,theta,sigma,kappa,rho,r,n,T)
% a1 to a5
a1=2*v0/sigma^2;  
a2=2*kappa*theta/sigma^2;
a3=log(S0)+((r*sigma-kappa*theta*rho)*T)/(2*sigma)-(rho*v0)/sigma;
a4=log(S0)-(rho*v0/sigma)+(r-rho*kappa*theta/sigma)*T;
a5=(kappa*v0+kappa^2*theta*T)/(sigma^2);

% recursively computing h_n
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
    h_matrix(j,:)=A1(j-4,1).*(A2.*h_matrix(j-4,:)+A3.*...
        (T*h_matrix(j-3,:))+A4.*h_matrix(j-2,:));
end
% compute the H and H_tilde
H=sum(h_matrix(3:end,:),1); %value of H, sum from h0 to hn
h_tilde=(nmat/T).*h_matrix(4:end,:); %matrix for computing H_tilde
H_tilde=sum(h_tilde,1); %value of H_tilde, sum from h1 to hn
ret=exp(-a1*(H_tilde./H)-a2*log(H)+a3*s+a4*w+a5); %return psi(s,w)
end














