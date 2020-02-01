clf(figure(1))
clf(figure(2))

mean_arg = [5; 10];
variance_arg = [8 0; 0, 4];

Class_A = data(200, [5; 10], [8, 0; 0, 4]);
Class_B = data(200, [10; 15], [8, 0; 0, 4]);
Class_C = data(100, [5; 10], [8, 4; 4, 40;]);
Class_D = data(200, [15; 10], [8, 0; 0, 8]);
Class_E = data(150, [10; 5], [10, -5; -5, 20]);

Cont_A = std_cont([5; 10], [8, 0; 0, 4]);
Cont_B = std_cont([10; 15], [8, 0; 0, 4]);
Cont_C = std_cont([5; 10], [8, 4; 4, 40;]);
Cont_D = std_cont([15; 10], [8, 0; 0, 8]);
Cont_E = std_cont([10; 5], [10, -5; -5, 20]);

% figure(1)
% title("Classes A and B");
% hold on
% scatter(Class_A(1, :), Class_A(2, :),'x', 'red');
% plot(Cont_A(1, :), Cont_A(2, :), 'red', 'LineWidth', 2);
% scatter(Class_B(1, :), Class_B(2, :), 'x', 'blue');
% plot(Cont_B(1, :), Cont_B(2, :), 'blue', 'LineWidth', 2);
% hold off
% 
figure(2)
title("Classes C, D, and E")
hold on
scatter(Class_C(1, :), Class_C(2, :), 'x', 'red');
plot(Cont_C(1, :), Cont_C(2, :), 'red', 'LineWidth', 2);
scatter(Class_D(1, :), Class_D(2, :), 'x', 'blue');
plot(Cont_D(1, :), Cont_D(2, :), 'blue', 'LineWidth', 2);
scatter(Class_E(1, :), Class_E(2, :), 'x', 'green');
plot(Cont_E(1, :), Cont_E(2, :), 'green', 'LineWidth', 2);
hold off

figure(3)
hold on
scatter(Class_A(1, :), Class_A(2, :),'x', 'red');
scatter(Class_B(1, :), Class_B(2, :), 'x', 'blue');
boundary(@map1, [-15, 0], [35, 30]);
boundary(@MED1, [-15, 0], [35, 30]);
hold off

figure(4)
hold on
scatter(Class_C(1, :), Class_C(2, :), 'x', 'red');
scatter(Class_D(1, :), Class_D(2, :), 'x', 'blue');
scatter(Class_E(1, :), Class_E(2, :), 'x', 'green');
boundary(@map2, [-25, -80], [45, 120]);
hold off

function Class = data(n, u, cov)
   x = randn(2, n);
   x = cov * x;
   Class = bsxfun(@plus, x, u);
end

function contour = std_cont(u, cov)
    x = linspace(0, 2 * pi);
    unit_centered_contour = [cos(x); sin(x)];
    centered_contour = cov * unit_centered_contour;
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
        x = 1;
        if d1 < d3
        x = d1;
        end
    elseif d2 < d3
        x = 2;
    else
        x = 3;
    end
end


%%GED


%%GED



%%MAP


%%MAP

function MAP_classifier_case_1 = map1(x1, x2)
    mu_a = [5; 10];
    sigma_a = [8, 0; 0, 4];
    
    mu_b = [10; 15];
    sigma_b = [8, 0; 0, 4];

    x = [x1; x2];
    
    p_x_given_a = (1/(sqrt(2*pi)*sqrt(det(sigma_a))))*exp((-1/2)*(transpose(x-mu_a)*inv(sigma_a))*(x-mu_a));
    p_x_given_b = (1/(sqrt(2*pi)*sqrt(det(sigma_b))))*exp((-1/2)*(transpose(x-mu_b)*inv(sigma_b))*(x-mu_b));
    
    if(p_x_given_a >= p_x_given_b)
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
    
    p_x_given_c = (1/(sqrt(2*pi)*sqrt(det(sigma_c))))*exp((-1/2)*(transpose(x-mu_c)*inv(sigma_c))*(x-mu_c));
    p_x_given_d = (1/(sqrt(2*pi)*sqrt(det(sigma_d))))*exp((-1/2)*(transpose(x-mu_d)*inv(sigma_d))*(x-mu_d));
    p_x_given_e = (1/(sqrt(2*pi)*sqrt(det(sigma_e))))*exp((-1/2)*(transpose(x-mu_e)*inv(sigma_e))*(x-mu_e));
    
    if((p_x_given_c >= p_x_given_d) & (p_x_given_c >= p_x_given_e))
        MAP_classifier_case_2 = 3;
    elseif((p_x_given_d >= p_x_given_c) & (p_x_given_d >= p_x_given_e))
        MAP_classifier_case_2 = 4;
    else
        MAP_classifier_case_2 = 5;
    end
end


%%KNN


%KNN


function output = boundary(classifier, start, finish)
    x = linspace(start(1), finish(1), 500);
    y = linspace(start(2), finish(2), 500);
    [X, Y] = meshgrid(x,y);
    A = arrayfun(classifier, X, Y);
    contour(X, Y, A, 1.5);
end