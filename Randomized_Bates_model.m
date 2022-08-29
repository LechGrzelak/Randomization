function Randomized_Bates_model
close all;clc;
% Randomization of the Bates model.
% 
% The code provided is based on the article 'On Randomization of Affine Diffusion Processes
% with Application to Pricing of Options on VIX and S&P 500' by Lech A. Grzelak, 
% L.A. Grzelak@uu.nl
% 
% @article{grzelakRAnD,
%   title={On Randomization of Affine Diffusion Processes with Application to Pricing of Options on {VIX} and {S&P} 500},
%   author={Grzelak, Lech A.},
%   journal={arXiv:2208.12518},  
%   year = {2022}
% }

K = linspace(60,140,25)/100;

% The number of Quadrature points
N = 9;

% The Bates model parameters
r         = 0.0;
muJ       = -0.1; 
sigmaJ    = 0.06; 
xiP       = 0.08; 
kappa     = 0.5; 
gamma     = 0.5;
vbar      = 0.13;
rho       = -0.7;
T         = 1/12;
v0        = 0.13; 
S0        = 1;
CP        = 'c';

% REFERENCE: Implied Volatilities for the Bates model with constant
% parameters
cf_Bates       = @(u)ChFBates(u, T, kappa,vbar,gamma,rho, v0, r, muJ, sigmaJ, xiP);
Call_COS_Bates = CallPutOptionPriceCOSMthd(cf_Bates,CP,S0,r,T,K',1000,8);
IV_COS_Bates   = ImpliedVols(K,Call_COS_Bates,S0,T,r,0.3);
figure(1)
set(gca, 'ColorOrder', [0 0 0; 0.7 0.1 0.2;0.7 0.3 0.9;0.1 0.2 0.7]); 
hold on;grid on;
plot(log(K),IV_COS_Bates*100,'r','LineWidth',1.5)

%%%%%%%%%%%%%%%% Randomization of vol-vol (gamma) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p1Vec = [0.01,0.2,0.1,0.75];
p2Vec = [0.25,5,2,1.75];

% Randomization with Uniform distribution
leg={strcat('$\gamma = ',num2str(gamma),'$')};
idx =1;
for i =1 :length(p1Vec)
    p1 = p1Vec(i);
    p2 = p2Vec(i);
    [x_i,w_i]    = UniformCollocation(p1,p2,N); 
    EX = sum(x_i.*w_i);
    cf_generic    = @(u)ChFBates_RAnD(u, T, kappa, vbar, rho, v0, r, muJ, sigmaJ, xiP, x_i, w_i);
    Call_COS_RAnD = CallPutOptionPriceCOSMthd(cf_generic,CP,S0,r,T,K',1000,8);
    IV_COS_RAnD   = ImpliedVols(K,Call_COS_RAnD,S0,T,r,0.3);
    plot(log(K),IV_COS_RAnD*100)
    idx = idx +1;
    leg{idx}={strcat('$\gamma \sim ',' \mathcal{U}([',num2str(p1),',',num2str(p2),']),','\;\;\;  {\bf{E}}[\gamma]= ',num2str(EX),'$')};
    
end  

legend(string(leg),'interpreter','latex')
title('Implied Volatilities for RAnD Bates Model')
xlabel('Strike, K (log-moneyness)')
ylabel('Implied Volatility, [%]')

% Exact pricing of European Call/Put option with the Black-Scholes model
function value=BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
% Black-Scholes Call option price
d1    = (log(S_0 ./ K) + (r + 0.5 * sigma^2) * tau) / (sigma * sqrt(tau));
d2    = d1 - sigma * sqrt(tau);
if lower(CP) == 'c' || lower(CP) == 1
    value =normcdf(d1) * S_0 - normcdf(d2) .* K * exp(-r * tau);
elseif lower(CP) == 'p' || lower(CP) == -1
    value =normcdf(-d2) .* K*exp(-r*tau) - normcdf(-d1)*S_0;
end

function [IV]=ImpliedVols(K,C,S0,T,r,initial)
IV=zeros([length(K),1]);
for i = 1:length(K)
    IV(i)=ImpliedVolatility('c',C(i),K(i),T,S0,r,initial);
end

function impliedVol = ImpliedVolatility(CP,marketPrice,K,T,S_0,r,initialVol)
    func = @(sigma) (BS_Call_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice);
    impliedVol = fzero(func,initialVol);

function cf=ChFBates(u, tau, kappa,vBar,gamma,rho, v0, r, muJ, sigmaJ, xiP)
i     = complex(0,1);

% functions D_1 and g
D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

% complex valued functions A and C
C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

% Adjustment for the Bates model
A = A - xiP*i*u*tau*(exp(muJ+1/2*sigmaJ^2)-1) + xiP*tau*(exp(i*u*muJ-1/2*sigmaJ^2*u.^2)-1);

% ChF for the Bates model
cf = exp(A + C * v0);

function CfTotal=ChFBates_RAnD(u, tau, kappa,vBar,rho, v0, r, muJ, sigmaJ, xiP, x_i, w_i)
i     = complex(0,1);

CfTotal = zeros([1,length(u)]);
for k = 1: length(x_i)
    % Get realization for gamma
    gamma = x_i(k);
                
    % functions D_1 and g
    D_1  = sqrt(((kappa -i*rho*gamma.*u).^2+(u.^2+i*u)*gamma^2));
    g    = (kappa- i*rho*gamma*u-D_1)./(kappa-i*rho*gamma*u+D_1);    

    % complex valued functions A and C
    C = (1/gamma^2)*(1-exp(-D_1*tau))./(1-g.*exp(-D_1*tau)).*(kappa-gamma*rho*i*u-D_1);
    A = i*u*r*tau + kappa*vBar*tau/gamma^2 * (kappa-gamma*rho*i*u-D_1)-2*kappa*vBar/gamma^2*log((1-g.*exp(-D_1*tau))./(1-g));

    % Adjustment for the Bates model
    A = A - xiP*i*u*tau*(exp(muJ+1/2*sigmaJ^2)-1) + xiP*tau*(exp(i*u*muJ-1/2*sigmaJ^2*u.^2)-1);

    % ChF for the Bates model
    cf = exp(A + C * v0);
    CfTotal = CfTotal + cf.*w_i(k);
end

function [x_i,w_i]=UniformCollocation(a,b,N)

a_idx =@(k)a.^(0:1:k);
b_idx =@(k)b.^(k:-1:0);

EX = @(k)1./(k+1)*sum(a_idx(k).*b_idx(k));
% Moments for standard nodmal i.e. N(0,1)
for i=1:N+1
    for j=1:N+1
        if (i==1&&j==1)
        M(i,j)=1;
        else
        M(i,j)=EX(i+j-2);
        end
    end
end

%%% Zeros as the diagonal of Matrix B
[x_i,w_i]     = FindCollocationPoints(M);

function [x_i,w_i] = FindCollocationPoints(M)

[N,~] =size(M);
N =N-1;

R=chol(M);

alpha(1)  =  R(1,2);
beta(1)   =  (R(2,2)/R(1,1))^2;
for i=2:N-1
    alpha(i) =R(i,i+1)/R(i,i)-R(i-1,i)/R(i-1,i-1);
    beta(i)  =(R(i+1,i+1)/R(i,i))^2;
end
alpha(N)=R(N,N+1)/R(N,N)-R(N-1,N)/R(N-1,N-1);

%%% Construction of the array and zeros calculation
J=diag(sqrt(beta),-1)+diag(alpha,0)+diag(sqrt(beta),1);
[w_i,B]   = eig(J);

w_i=(w_i(1,:)).^2;
w_i= w_i';

%%% Zeros as the diagonal of Matrix B
x_i     = diag(B);

function value = CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L)
i = complex(0,1);
% cf   - characteristic function as a functon, in the book denoted as \varphi
% CP   - C for call and P for put
% S0   - Initial stock price
% r    - interest rate (constant)
% tau  - time to maturity
% K    - vector of strikes
% N    - Number of expansion terms
% L    - size of truncation domain (typ.:L=8 or L=10)  

x0 = log(S0 ./ K);   

% Truncation domain
a = 0 - L * sqrt(tau); 
b = 0 + L * sqrt(tau);

k = 0:N-1;              % row vector, index for expansion terms
u = k * pi / (b - a);   % ChF arguments

H_k = CallPutCoefficients(CP,a,b,k);
temp    = (cf(u) .* H_k).';
temp(1) = 0.5 * temp(1);     % adjust the first element by 1/2

mat = exp(i * (x0 - a) * u);  % matrix-vector manipulations

% Final output
value = exp(-r * tau) * K .* real(mat * temp);

% Coefficients H_k for the COS method
function H_k = CallPutCoefficients(CP,a,b,k)
    if lower(CP) == 'c' || CP == 1
        c = 0;
        d = b;
        [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
         if a < b && b < 0.0
            H_k = zeros([length(k),1]);
         else
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k);
         end
    elseif lower(CP) == 'p' || CP == -1
        c = a;
        d = 0.0;
        [Chi_k,Psi_k]  = Chi_Psi(a,b,c,d,k);
         H_k = 2.0 / (b - a) * (- Chi_k + Psi_k);       
    end

function [chi_k,psi_k] = Chi_Psi(a,b,c,d,k)
    psi_k        = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
    psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
    psi_k(1)     = d - c;
    
    chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2); 
    expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d)  - cos(k * pi... 
                  * (c - a) / (b - a)) * exp(c);
    expr2 = k * pi / (b - a) .* sin(k * pi * ...
                        (d - a) / (b - a))   - k * pi / (b - a) .* sin(k... 
                        * pi * (c - a) / (b - a)) * exp(c);
    chi_k = chi_k .* (expr1 + expr2);