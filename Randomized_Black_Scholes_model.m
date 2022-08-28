function Randomized_Black_Scholes_model
close all;clc;
% Randomization of the Black-Scholes model.
% 
% The code provided is based on the article 'On Randomization of Affine Diffusion Processes
% with Application to Pricing of Options on VIX and S&P 500' by Lech A. Grzelak, 
% L.A. Grzelak@uu.nl
% 
% @article{grzelakRAnD,
%   title={On Randomization of Affine Diffusion Processes with Application to Pricing of Options on {VIX} and {S&P} 500},
%   author={Grzelak, Lech A.},
%   journal={arxiv},  
%   year = {2022}
% }

r         = 0.0;
S0        = 1;
K         = linspace(0.72*S0,1.4*S0,30)';

% Settings for the COS method
L = 5;
Ncos = 5000;

% Number of Quadrature points
N_ref   = 6; 
  
% Define a range for expiries, for the surface
Tvec = linspace(0.05,1,25);

%%%%%%%%%%%%%%%%%%% UNIFORM  %%%%%%%%%%%%%%%%%%%%%%%%%
a = 0.1;
b = 0.45;
[x_i,w_i] = UniformCollocation(a,b,N_ref); 

% Constant sigma is taken as the average over randomized sigma
m1 = sum(x_i.*w_i);
sigma_const = m1;

IV = zeros([length(Tvec),length(K)]);
IV_const = zeros([length(Tvec),length(K)]);

idx = 1;
for T=Tvec
    % Randomized ChF & the COS method
    cf_sigma        = @(u) ChF_RAnD_BS(u, T, r, x_i, w_i );
    Call_COS        = CallPutOptionPriceCOSMthd(cf_sigma,'c',S0,r,T,K,Ncos,L);
    
    % For comparison value BS model with constant sigma
    Call_const      = BS_Call_Option_Price('c',S0,K,sigma_const,T,r);
    IV(idx,:)       = ImpliedVols(K,Call_COS,S0,T,r,0.3);
    IV_const(idx,:) = ImpliedVols(K,Call_const,S0,T,r,0.3);
    idx = idx + 1;
end

hold on; grid on;
surf(log(K),Tvec,IV_const*100,'EdgeColor','r')
surf(log(K),Tvec,IV*100,'EdgeColor','b')
ylabel('T')
xlabel('strike K, log-moneyness')
zlabel('Implied Volatilities [%]')
legend('Black-Scholes','RAnD Black-Scholes')
title('Implied Volatilities for the RAnD Black-Scholes model','interpreter','latex')
view([44.2574697659734 35.7127874966164])

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

function CfTotal = ChF_RAnD_BS(u, tau, r, x_i, w_i)
i     = complex(0,1);

CfTotal = zeros([1,length(u)]);

for k = 1: length(x_i)  
    sigma   = x_i(k);
    cf      = exp((r - 1 / 2 * sigma.^2) .* i .* u * tau - 1/2 * sigma.^2 * u.^2 * tau);
    CfTotal = CfTotal + cf.*w_i(k);
end

function [IV]=ImpliedVols(K,C,S0,T,r,initial)
IV=zeros([length(K),1]);
for i = 1:length(K)
    IV(i)=ImpliedVolatility('c',C(i),K(i),T,S0,r,initial);
end

function impliedVol = ImpliedVolatility(CP,marketPrice,K,T,S_0,r,initialVol)
    func = @(sigma) (BS_Call_Option_Price(CP,S_0,K,sigma,T,r) - marketPrice);
    impliedVol = fzero(func,initialVol);
   
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
    
function [x_i,w_i] = FindCollocationPoints(M)

[N,~] =size(M);
N = N-1;

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