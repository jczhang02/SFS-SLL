function [P, V, A] = SFL(X, cluster_num, YYl, nl, param, NITER)

% X : d*n samples
% P d*c feature selection matrix
% A n*c soft lable matrix
% V c*c projection matrix
% c

% SFL Function Details
%   Input
%   -----
%   X: [Original Data Matrix], {numpy array}, shape {d, n};
%   YYL: [True Label Matrix of Labelled Samples], {numpy array}, shape {nl, cluster_num};
%   cluster_num: [cluster number], {int};
%   param: [parameters], {stuct}, shape {1, 4};
%   nl: [Number of Labelled Samples], {int};
%   NITER: [Iteration Number], {int};
%   -----
%   Output
%   -----
%   P: [Projection Matrix for Feature Selection], {numpy array}, shape {d, cluster_num};
%   V: [Projection Matrix], {numpy array}, shape {cluster_num, cluster_num};
%   A: [Soft Label Matrix], {numpy array}, shape {n, cluster_num};
%   ------

[d, n] = size(X);

%%Init
P = zeros(d, cluster_num);
V = rand(cluster_num, cluster_num);
A = rand(n, cluster_num);
U = rand(n, cluster_num);
L = YYl;
options = [2;5;1e-5;0];
[centers, U] = fcm(X', cluster_num, options);
U = U';

%%optimization
for count = 1:NITER

    %% update P
    Pi = sqrt(sum(P.*P, 2) + eps);
    hh = 0.5 ./ Pi;
    H = diag(hh);
    P = inv(X * X' + param.lambda * H) * X * A;
    %% update A
    for i = 1:nl
        vl = L * V';
        sk = sum(V' * V);
        ad = 0.5 * (param.beta * (X(:, i)' * P) +  param.alpha * vl(i, :) + U(i, :)) / (param.beta + 1 + param.alpha * sk) ;
        A(i, :) = EProjSimplex_new(ad);
    end
    for k = nl+1:n
       ad = 0.5 * (param.beta * (X(:, k)' * P) + U(k, :)) / (param.beta + 1);
       A(k, :) = EProjSimplex_new(ad);
    end
    %% update V
    I = eye(cluster_num);
    V = inv(A(1:nl, :)' * A(1:nl, :) + param.gamma * I) * A(1:nl, :)' * L;
end
