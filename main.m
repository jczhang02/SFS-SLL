clc;
clear;
warning('off');

% load('ecoli_10p.mat'); data = 'ecoli 10p'
% load('Yale_20p.mat'); data = 'yale 20p'
% load('ecoli_30p.mat'); data = 'ecoli 30p'
% load('ecoli_40p.mat'); data = 'ecoli 40p'
% load('ecoli_50p.mat'); data = 'ecoli 50p'
load('ecoli_10p.mat'); data = 'ba 20p';

cluster_number = length(unique(Y));

[N, D] = size(X);
Nl = size(Yl,1);

alpha = [1e-4, 1e-2, 1, 10, 1e3, 1e5];
lambda = [1e-4, 1e-2, 1, 10, 1e3, 1e5];
gamma = [1e-4, 1e-2, 1, 10, 1e3, 1e5];
beta = [1e-4, 1e-2, 1, 10, 1e3, 1e5];
ep = [1e-4, 1e-2, 1, 10, 1e3, 1e5];

NITER = 25;
runtime = 5;
r = 100;

cnt = 1;
pp = 1;
for aa = 1:1
    for bb = 1:1
        for cc = 1:1
             for dd = 1:1
%                 for ee = 1:6
                    param.lambda = lambda(aa);
                    param.gamma = gamma(bb);
                    param.beta = beta(cc);
                    param.alpha = alpha(dd);
%                     param.ep = ep(ee);
                    % Xw is the feature selected by algorithm
                    [ P, V, A ] = SFL(X', cluster_number, YYl, Nl, param, NITER);

                    if isequal(V(1,1), NaN)
                        disp('A and B are not the same size.')
                        pause
                    end
                    dc = size(P, 1);
                    [P_des, P_idx] = sort(P, 'descend');
                    P_idx = P_idx(1:r, :);
                    P_r = P_des(1:r, :);
                    P_identify = zeros(r, dc);
                    for i = 1:r
                        P_identify(i, P_idx(i)) = 1;
                    end
                    Xw = P_identify * X';

                    % Randomly make train data and test data
                    ndata = randperm(size(Xw',1));
                    matrix = Xw';
                    label = Y;

                    % train data
                    train_matrix = matrix(ndata(1:Nl),:);
                    train_label = label(ndata(1:Nl),:);

                    % test data
                    test_matrix = matrix(ndata(Nl+1:end),:);
                    test_label = label(ndata(Nl+1:end),:);

                    % Normalization
                    [Train_matrix,PS] = mapminmax(train_matrix');
                    Train_matrix = Train_matrix';
                    Test_matrix = mapminmax('apply',test_matrix',PS);
                    Test_matrix = Test_matrix';

                    %% SVM Create/Train(RBF Kernel)

                    % 1.find best c/g
                    [c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
                    [m,n] = size(c);
                    cg = zeros(m,n);
                    eps = 10^(-4);
                    v = 5;
                    bestc = 1;
                    bestg = 0.1;
                    bestacc = 0;
                    for i = 1:m
                        for j = 1:n
                            cmd = ['-q ', '-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
                            cg(i,j) = svmtrain(train_label,Train_matrix,cmd);
                            if cg(i,j) > bestacc
                                bestacc = cg(i,j);
                                bestc = 2^c(i,j);
                                bestg = 2^g(i,j);
                            end
                            if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j)
                                bestacc = cg(i,j);
                                bestc = 2^c(i,j);
                                bestg = 2^g(i,j);
                            end
                        end
                    end
                    cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];

                    % 2.create/train svm model
                    model = svmtrain(train_label,Train_matrix,cmd);
                    [predict_label_1,accuracy_1,prob_estimates] = svmpredict(train_label,Train_matrix,model);
                    [predict_label_2,accuracy_2,prob_estimates2] = svmpredict(test_label,Test_matrix,model);
                    result_1 = [train_label predict_label_1];
                    result_2 = [test_label predict_label_2];

                    Final_result(cnt,(1:3)) = accuracy_1;
                    Final_result(cnt,(4:6)) = accuracy_2;
                    Final_result(cnt,7) = param.lambda;
                    Final_result(cnt,8) = param.gamma;
                    Final_result(cnt,9) = param.beta;
                    Final_result(cnt,10) = param.alpha;
%                     Final_result(cnt,11) = param.ep;
                    cnt = cnt + 1;
%                 end
             end
        end
    end
end

save('ba_20p_1.mat', 'Final_result');
