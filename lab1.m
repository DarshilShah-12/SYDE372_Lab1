clear
% clf(figure(1))
% clf(figure(2))
clf(figure(3))

mean_arg = [5; 10];
variance_arg = [8 0; 0, 4];

global Class_A
global Class_B
global Class_C
global Class_D
global Class_E

Class_A = data(200, [5; 10], [8, 0; 0, 4]);
Class_A_test = data(200, [5; 10], [8, 0; 0, 4]);
Class_B = data(200, [10; 15], [8, 0; 0, 4]);
Class_C = data(100, [5; 10], [8, 4; 4, 40]);
Class_C_test = data(200, [5; 10], [8, 4; 4, 40]);
Class_D = data(200, [15; 10], [8, 0; 0, 8]);
Class_E = data(150, [10; 5], [10, -5; -5, 20]);

Cont_A = std_cont([5; 10], [8, 0; 0, 4]);
Cont_B = std_cont([10; 15], [8, 0; 0, 4]);
Cont_C = std_cont([5; 10], [8, 4; 4, 40;]);
Cont_D = std_cont([15; 10], [8, 0; 0, 8]);
Cont_E = std_cont([10; 5], [10, -5; -5, 20]);

% figure(1)
% % axis equal
% title("Classes A and B");
% hold on
% scatter(Class_A(1, :), Class_A(2, :),'x', 'red');
% plot(Cont_A(1, :), Cont_A(2, :), 'red', 'LineWidth', 2);
% scatter(Class_B(1, :), Class_B(2, :), 'x', 'blue');
% plot(Cont_B(1, :), Cont_B(2, :), 'blue', 'LineWidth', 2);
% boundary(@MED1, [-15, 0], [35, 30], '-');
% boundary(@gemClassAB, [-15, 0], [35, 30], ':');
% boundary(@map1, [-15, 0], [35, 30], '--');
% legend('A', 'A', 'B', 'B', 'MED', 'GEM', 'MAP');
% hold off

% figure(2)
% % axis equal
% title("Classes C, D, and E");
% hold on
% scatter(Class_C(1, :), Class_C(2, :), 'x', 'red');
% plot(Cont_C(1, :), Cont_C(2, :), 'red', 'LineWidth', 2);
% scatter(Class_D(1, :), Class_D(2, :), 'x', 'blue');
% plot(Cont_D(1, :), Cont_D(2, :), 'blue', 'LineWidth', 2);
% scatter(Class_E(1, :), Class_E(2, :), 'x', 'green');
% plot(Cont_E(1, :), Cont_E(2, :), 'green', 'LineWidth', 2);
% boundary(@MED2, [-25, -80], [45, 120], '-');
% boundary(@gemClassCDE, [-25, -80], [45, 120], ':');
% boundary(@map2, [-25, -80], [45, 120], '--');
% legend('A', 'A', 'B', 'B', 'C', 'C', 'MED', 'GEM', 'MAP');
% hold off

figure(3)
% axis equal
title("Classes A and B NN");
hold on
scatter(Class_A(1, :), Class_A(2, :), 'x', 'red');
scatter(Class_B(1, :), Class_B(2, :), 'x', 'blue');
boundary(@K5NN_AB, [-5, 5], [20, 20], '-');
hold off


% dcm(Class_A, @map1, 1);
% dcm(Class_B, @map1, 2);
% dcm(Class_C, @map2, 3);
% dcm(Class_D, @map2, 4);
% dcm(Class_E, @map2, 5);

dcm(Class_A_test, Class_B, @MED1);
dcm2(Class_C, Class_D, Class_E, @map2);

dcm(Class_A_test, Class_B, @gemClassAB);
dcm2(Class_C, Class_D, Class_E, @gemClassCDE);

dcm(Class_A_test, Class_B, @MED1);
dcm2(Class_C, Class_D, Class_E, @map2);

dcm(Class_A_test, Class_B, @gemClassAB);
dcm2(Class_C, Class_D, Class_E, @gemClassCDE);

dcm(Class_A, Class_B, @map1);
dcm2(Class_C, Class_D, Class_E, @map2);

med1 = classify(@MED1, Class_A);
med1Err = errorRate(med1, length(Class_A(1,:)));
med2 = classify(@MED2, Class_C);
med2Err = errorRate(med2, length(Class_C(1,:)));
gem1 = classify(@gemClassAB, Class_A);
gem1Err = errorRate(gem1, length(Class_A(1,:)));
gem2 = classify(@gemClassCDE, Class_C);
gem2Err = errorRate(gem2, length(Class_C(1,:)));
MAP1 = classify(@map1, Class_A);
MAP1Err = errorRate(MAP1, length(Class_A(1,:)));
MAP2 = classify(@map2, Class_C);
MAP2Err = errorRate(MAP2, length(Class_C(1,:)));
N1 = NNclassify(@NN_AB, Class_A_test);
N1Err = errorRate(N1, length(Class_A_test(1,:)));
N2 = NNclassify(@NN_CDE, Class_C_test);
N2Err = errorRate(N2, length(Class_C_test(1,:)));
K1 = NNclassify(@K5NN_AB, Class_A_test);
K1Err = errorRate(k1, length(Class_A_test(1,:)));

disp('NN for multiclass error rate:')
disp(N2Err)

% disp(K2Err)


function Class = data(n, u, cov)
   [V, D] = eig(cov);
   x = randn(2, n);
   x = V * sqrt(D) * x;
   Class = bsxfun(@plus, x, u);
end

function contour = std_cont(u, cov)
    [V, D] = eig(cov);
    x = linspace(0, 2 * pi);
    unit_centered_contour = [cos(x); sin(x)];
    centered_contour = V * sqrt(D) * unit_centered_contour;
    contour = bsxfun(@plus, centered_contour, u);
end


%%MED
function x = MED1(x1, x2)
    % x = MED1(8, 13)
    uA = [5; 10];
	uB = [10; 15];
    
    d1 = norm([x1;x2]-uA);
    d2 = norm([x1;x2]-uB);
    
    if d1 < d2
        x = 1;
    else
        x = 2;
    end
end

function x = MED2(x1, x2)
    % x = MED2(8, 13)
    uC = [5; 10];
	uD = [15; 10];
    uE = [10; 5];
    
    d1 = norm([x1;x2]-uC);
    d2 = norm([x1;x2]-uD);
    d3 = norm([x1;x2]-uE);

    if d1 < d2
        if d1 < d3 %132
            x = 1;
        else %312
            x = 3;
        end
    else
        if d2 < d3 %213
            x = 2;
        else %321
            x = 3;
        end
    end
end


%%GED
function gemClassification = gemClassAB(x1, x2)
    sample = [x1; x2];
    
    %find distance A.    
    a_cov = [8, 0; 0, 4];
    a_u = [5; 10];
    distance_A = transpose(sample - a_u) * inv(a_cov) * (sample - a_u);
    
    %find distance B.
    b_cov = [8, 0; 0, 4];
    b_u = [10; 15];
    distance_B = transpose(sample - b_u) * inv(b_cov) * (sample - b_u);
   
    %Compare distances & classify.    
    if(distance_A < distance_B)
        determinedClass = 1;
    end
    if (distance_A > distance_B)
        determinedClass = 2;
    end
    if (distance_A == distance_B)
        determinedClass = 0;
    end
    gemClassification = determinedClass;
end

%%GED
function gemClassification = gemClassCDE(x1, x2)
    sample = [x1; x2];
    
    %find distance C.
    c_cov = [8, 4; 4, 40];
    c_u  = [5; 10];
    distance_C = transpose(sample - c_u) * inv(c_cov) * (sample - c_u);

    %find distance D.
    d_cov = [8, 0; 0, 8];
    d_u  = [15; 10];
    distance_D = transpose(sample - d_u) * inv(d_cov) * (sample - d_u);
    
    %find distance E.
    e_cov = [10, -5; -5, 20];
    e_u  = [10; 5];
    distance_E = transpose(sample - e_u) * inv(e_cov) * (sample - e_u);
   
    %Compare distances & classify.
    distances = [distance_C, distance_D, distance_E];
    if distance_C == min(distances)
        gemClassification = 1;
        return;
    end
    if distance_D == min(distances)
        gemClassification = 2;
        return;
    end
    if distance_E == min(distances)
        gemClassification = 3;
        return;
    end
    
end


%%MAP

function MAP_classifier_case_1 = map1(x1, x2)
    mu_a = [5; 10];
    sigma_a = [8, 0; 0, 4];
    
    mu_b = [10; 15];
    sigma_b = [8, 0; 0, 4];

    x = [x1; x2];
    
    p_x_given_a_and_p_a = ((1/(sqrt(2*pi)*sqrt(det(sigma_a))))*exp((-1/2)*(transpose(x-mu_a)*inv(sigma_a))*(x-mu_a)))*200;
    p_x_given_b_and_p_b = ((1/(sqrt(2*pi)*sqrt(det(sigma_b))))*exp((-1/2)*(transpose(x-mu_b)*inv(sigma_b))*(x-mu_b)))*200;
    
    if(p_x_given_a_and_p_a >= p_x_given_b_and_p_b)
        MAP_classifier_case_1 = 1;
    else
        MAP_classifier_case_1 = 2;
    end
end

function MAP_classifier_case_2 = map2(x1, x2)
    mu_c = [5; 10];
    sigma_c = [8, 4; 4, 40];
    
    mu_d = [15; 10];
    sigma_d = [8, 0; 0, 8];
    
    mu_e = [10; 5];
    sigma_e = [10, -5; -5, 20];
    
    x = [x1; x2];
    
    p_x_given_c = ((1/(sqrt(2*pi)*sqrt(det(sigma_c))))*exp((-1/2)*(transpose(x-mu_c)*inv(sigma_c))*(x-mu_c)))*100;
    p_x_given_d = ((1/(sqrt(2*pi)*sqrt(det(sigma_d))))*exp((-1/2)*(transpose(x-mu_d)*inv(sigma_d))*(x-mu_d)))*200;
    p_x_given_e = ((1/(sqrt(2*pi)*sqrt(det(sigma_e))))*exp((-1/2)*(transpose(x-mu_e)*inv(sigma_e))*(x-mu_e)))*150;
    
    if((p_x_given_c >= p_x_given_d) && (p_x_given_c >= p_x_given_e))
        MAP_classifier_case_2 = 1;
    elseif((p_x_given_d >= p_x_given_c) && (p_x_given_d >= p_x_given_e))
        MAP_classifier_case_2 = 2;
    else
        MAP_classifier_case_2 = 3;
    end
end


%%KNN
function ab = K5NN_AB(x1, x2)
    p1 = [x1; x2];
    [ab,~] = KNN(5,p1);
end

function cde = K5NN_CDE(x1, x2)
    p1 = [x1; x2];
    [~,cde] = KNN(5,p1);
end

function minPoints = Points(k,p1,class)
    [~,minPoints] = kmin(k,p1,class);
end

function [ab,cde] = KNN(k,p1)
    global Class_A
    global Class_B
    global Class_C
    global Class_D
    global Class_E
    
    [ADist, ~] = kmin(k,p1,Class_A);
    [BDist, ~] = kmin(k,p1,Class_B);
    [CDist, ~] = kmin(k,p1,Class_C);
    [DDist, ~] = kmin(k,p1,Class_D);
    [EDist, ~] = kmin(k,p1,Class_E);
    arr1=[sum(ADist), sum(BDist)];
    arr2=[sum(CDist),sum(DDist),sum(EDist)];

    [~,I1] = min(arr1);
    ab = I1;
    [~,I2] = min(arr2);
    cde = I2;
end

function [minDist, minPoints] = kmin(k, p1, Dataset)
    minDist = realmax*ones(k,1);
    minPoints = zeros(length(k),1);

    for i = 1:size(Dataset,2)
        m = Dataset(:,i);
        newDist = euclidDist(p1,m);
        if newDist<max(minDist)
            [~,H] = max(minDist);
            minDist(H) = newDist;
        end
    end
end

function eucDi = euclidDist(p1,p2)
	eucDi = (p1(1)-p2(1))^2+(p1(2)-p2(2))^2;
end

%%NN

function ab = NN_AB(x1, x2)
    p1 = [x1; x2];
    [ab,~] = KNN(1,p1);
end

function cde = NN_CDE(p1)
    [~,cde] = KNN(1,p1);
end

function developConfusionMatrix = dcm(class_a, class_b, classifier)
    
    expected_values_for_a = ones(1, 200);
    expected_values_for_b = ones(1, 200)*2;
    
    all_expected_values = [expected_values_for_a, expected_values_for_b];
    
    
    
%     disp(all_expected_values);
    all_predicted_values = zeros(1, 400);
    for i=1:size(class_a, 2)
        all_predicted_values(i) = classifier(class_a(1, i), class_a(2, i));
    end
    
    for i=1:size(class_b, 2)
        all_predicted_values(200 + i) = classifier(class_b(1, i), class_b(2, i));
    end
    
    C = confusionmat(all_expected_values, all_predicted_values);
    
    disp(C);
    
%     correct_count = 0;
%     for i=1:size(class, 2)
%         if(classifier(class(1, i), class(2, i)) == expected_value)
%             correct_count = correct_count + 1;
%         end
%     end
%     disp(correct_count);
%     developConfusionMatrix = correct_count;
end

function confusionMatrix2 = dcm2(class_c, class_d, class_e, classifier)
    
    expected_values_for_c = ones(1, 100)*3;
    expected_values_for_d = ones(1, 200)*4;
    expected_values_for_e = ones(1, 150)*5;
    
    all_expected_values = [expected_values_for_c, expected_values_for_d, expected_values_for_e];
    
%     disp(all_expected_values);
    all_predicted_values = zeros(1, 450);
    for i=1:size(class_c, 2)
        all_predicted_values(i) = classifier(class_c(1, i), class_c(2, i));
    end
    
    for i=1:size(class_d, 2)
        all_predicted_values(100 + i) = classifier(class_d(1, i), class_d(2, i));
    end
    
    for i=1:size(class_e, 2)
        all_predicted_values(300 + i) = classifier(class_e(1, i), class_e(2, i));
    end
    
    C = confusionmat(all_expected_values, all_predicted_values);
    
    C( all(~C,2), : ) = [];
    C( :, all(~C, 1)) = [];
    disp(C);
    
%     correct_count = 0;
%     for i=1:size(class, 2)
%         if(classifier(class(1, i), class(2, i)) == expected_value)
%             correct_count = correct_count + 1;
%         end
%     end
%     disp(correct_count);
%     developConfusionMatrix = correct_count;
end

function boundary(classifier, start, finish, style)
    x = linspace(start(1), finish(1), 100);
    y = linspace(start(2), finish(2), 100);
    [X, Y] = meshgrid(x,y);
    A = arrayfun(classifier, X, Y);
    contour(X, Y, A, [1, 2, 3, 4, 5], ['k', style]);
end


function hits = classify(classifier, data)
    hits = 0;
    for i = 1:length(data(1, :))
        result = classifier(data(1,i), data(2,i));
        if result == 1
            hits = hits + 1;
        end
    end
end

function hits = NNclassify(classifier, data)
    hits = 0;
    for i = 1:length(data(1,:))
        p = [data(1,i); data(2,i)];
        result = classifier(p);
        if result == 1
            hits = hits + 1;
        end
    end
end

function err = errorRate(result, expected)
    err = (expected-result)/expected;
end
