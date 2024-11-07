%%  Description
%   重点是保证学习数据的质量，也就是必须是系统状态收敛过程中产生的数据，才能逼近到准确值。
%   否则，会使权重往错误方向更新。再次强调了策略是admissible的重要性。
%   此过程不用重复完整的收敛过程，只需对状态加噪声后用更新的权重（计算过程
%   不使用加噪的控制输入）计算输入。
%
%   需要弄清楚，PE条件物理意义到底是什么，应当如何满足？？
%   noise的选取为尽量多的三角函数高阶项组合。且需满足一定强度（如线性例子），
%   才可以产生足够多的数据以供学习。
%   应当注意，加噪声后的输入只作用于系统状态更新，而不应该作用于权重更新过程中！！！

%%  Simulation
rng('default'); % Reset random number generator for reproducibility
close all;
clear;
clc;

global Q;
global R;

global W1;
global W2;

global alpha1;
global alpha2;
global F1;
global F2;

global num_states;
global num_centers;

global is_learning;


% Initialize variables
num_states = 6;
num_centers = 6;
F1 = ones(num_centers,1);
F2 = eye(num_centers);
R = 1; % 大小可以决定值函数对于u和e的trade-off。过大可能导致不收敛
alpha1 = 20;
alpha2 = 1;
is_learning = 1;
Q = eye(num_states) * 1;

e0=rand(num_states,1);
W1=ones(num_centers,1)*1;
W2=ones(num_centers,1);


tvec = [];
xmat = [];
umat = [];

% Critic, actor weights
W1_mat = [];
W2_mat = [];

t_learn = 480;
t_sim = 20;


x0_sim = [e0; W1; W2];
tspan = [0, t_learn];
[t, x] = ode45(@odefunct, tspan, x0_sim);

% Store time data
tvec = [    tvec
    t           ];

% Store system state data
xmat = [    xmat
    x(:,1:num_states)    ];

% Store critic weights c
W1_mat = [   W1_mat
    x(:,num_states+1:num_states+num_centers)    ];

% Store actor weights w
W2_mat = [   W2_mat
    x(:,end-num_centers+1:end)    ];


% DEBUGGING: Final critic NN params
W1

% DEBUGGING: Final actor NN params
W2

is_learning = 0;

e0=x(end,1:num_states)';

x0_sim = [e0; W1; W2];
tspan = [t_learn, t_learn+t_sim];
[t, x] = ode45(@odefunct, tspan, x0_sim);

% Store time data
tvec = [    tvec
    t ];

% Store system state data
xmat = [    xmat
    x(:,1:num_states)    ];

% Store critic weights c
W1_mat = [   W1_mat
    x(:,num_states+1:num_states+num_centers)    ];

% Store actor weights w
W2_mat = [   W2_mat
    x(:,end-num_centers+1:end)    ];

for k=1:size(W1_mat,1)
    g = G(xmat(k,:));
    dPhi = Derivative_Phi(xmat(k,:));    
    u = -0.5 * inv(R) * g' * dPhi' * W1_mat(k,:)';
    umat = [umat
        u'];
end

% Plotting
figure;
plot(tvec, umat);
title('u');

figure;
plot(tvec, W1_mat);
title('W1');
legend('W11', 'W12', 'W13', 'W14', 'W15', 'W16');
figure;
plot(tvec, W2_mat);
title('W2');
legend('W21', 'W22', 'W23', 'W24', 'W25', 'W26');

figure;
plot(tvec, xmat);
title('System States');
legend('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6');

function dPhi = Derivative_Phi(xs)
dPhi=[2*xs(1),0,0,0,0,0;...
%     xs(2),xs(1),0,0,0,0;...
%     xs(3),0,xs(1),0,0,0;...
%     xs(4),0,0,xs(1),0,0;...
%     xs(5),0,0,0,xs(1),0;...
%     xs(6),0,0,0,0,xs(1);...
    0,2*xs(2),0,0,0,0;...
%     0,xs(3),xs(2),0,0,0;...
%     0,xs(4),0,xs(2),0,0;...
%     0,xs(5),0,0,xs(2),0;...
%     0,xs(6),0,0,0,xs(2);...
    0,0,2*xs(3),0,0,0;...
%     0,0,xs(4),xs(3),0,0;...
%     0,0,xs(5),0,xs(3),0;...
%     0,0,xs(6),0,0,xs(3);...
    0,0,0,2*xs(4),0,0;...
%     0,0,0,xs(5),xs(4),0;...
%     0,0,0,xs(6),0,xs(4);...
    0,0,0,0,2*xs(5),0;...
%     0,0,0,0,xs(6),xs(5);...
    0,0,0,0,0,2*xs(6)];
end

function f = F(xs)
q=[xs(4);xs(5);xs(6)];
m=5;
m_I=0.1125;
L=0.14;
C_t=0.3;
R1=0.0635;
cM=[m,0,0;...
    0,m,0;...
    0,0,m_I];
cC=[0,m*xs(6),0;...
    -m*xs(6),0,0;...
    0,0,0];
F_theta=[cos(xs(3)), -sin(xs(3)), 0;...
    sin(xs(3)), cos(xs(3)), 0;...
    0,         0,         1];
f1 = F_theta * q;
f2=-inv(cM)*cC*q;
f = [f1
        f2];
end

function g = G(xs)
m=5;
m_I=0.1125;
L=0.14;
C_t=0.3;
R1=0.0635;
cM=[m,0,0;...
    0,m,0;...
    0,0,m_I];
cB=[1/2,1/2,-1;...
    sqrt(3)/2,-sqrt(3)/2,0;...
    L,L,L];
g1=ones(3,3);
g2=inv(cM)*cB*C_t/R1;
g=[g1
   g2];
end

function xdot = odefunct(t,x)
global Q;
global R;

global W1;
global W2;

global alpha1;
global alpha2;
global F1;
global F2;
global num_states;
global num_centers;
global is_learning;

es = x(1:num_states);

% Critic NN state variables
W1 = x(num_states+1:end-num_centers);

% Actor NN state variables
W2 = x(end-num_centers+1:end);

x_r = sin(2*t);
xs = es + x_r;

dPhi = Derivative_Phi(es);
g = G(xs);
actor = -0.5 * inv(R) * g' * dPhi' * W2;
%             actor = actor + (rand(size(actor)) - 0.5) * noise_coef;

sigma2 = dPhi * (F(xs) + g *actor);
normsigma2 = sigma2' * sigma2 + 1;
mx = sigma2 / (sigma2' * sigma2 + 1)^2;
W1Change = -alpha1 * (sigma2 / normsigma2^2) * (sigma2' * W1 + xs' * Q * xs + actor' * R * actor);

D1x = dPhi * g * inv(R) * g' * dPhi';
W2Change = -alpha2 * ((F2 * W2 - F2 * W1) - 0.25 * D1x * W2 * mx' * W1);

noise1 = ones(1,1) * exp(-0.001*t)*1*(sin(t)^2*cos(t) +...
    sin(2*t)^2*cos(0.1*t) +...
    sin(-1.2*t)^2*cos(0.5*t) + sin(t)^5 + sin(1.12*t)^2 +...
    cos(2.4*t)*sin(2.4*t)^3);
noise2 = ones(1,1) * exp(-0.001*t)*5*(sin(t)^2*cos(t) +...
    sin(2*t)^2*cos(0.1*t) +...
    sin(-1.2*t)^2*cos(0.5*t) + sin(t)^5 + sin(1.12*t)^2 +...
    cos(2.4*t)*sin(2.4*t)^3);
noise3 = ones(1,1) * exp(-0.001*t)*10*(sin(t)^2*cos(t) +...
    sin(2*t)^2*cos(0.1*t) +...
    sin(-1.2*t)^2*cos(0.5*t) + sin(t)^5 + sin(1.12*t)^2 +...
    cos(2.4*t)*sin(2.4*t)^3);
noise = [noise1;noise2;noise3];
% noise = sin(1.12*t)^2;
if is_learning
    u = -0.5 * inv(R) * g' * dPhi' * W2 + noise;
else
    u = -0.5 * inv(R) * g' * dPhi' * W2;
end
dz = F(xs) + g * u;
if is_learning
    xdot = [dz
        W1Change
        W2Change];
else
    xdot = [dz
        zeros(num_centers,1)
        zeros(num_centers,1)];
end
end