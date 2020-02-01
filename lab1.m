clear


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

figure(1)
hold on
scatter(Class_A(1, :), Class_A(2, :),'x', 'red');
plot(Cont_A(1, :), Cont_A(2, :), 'red', 'LineWidth', 2);
scatter(Class_B(1, :), Class_B(2, :), 'x', 'blue');
plot(Cont_B(1, :), Cont_B(2, :), 'blue', 'LineWidth', 2);
hold off

figure(2)
hold on
scatter(Class_C(1, :), Class_C(2, :), 'x', 'red');
plot(Cont_C(1, :), Cont_C(2, :), 'red', 'LineWidth', 2);
scatter(Class_D(1, :), Class_D(2, :), 'x', 'blue');
plot(Cont_D(1, :), Cont_D(2, :), 'blue', 'LineWidth', 2);
scatter(Class_E(1, :), Class_E(2, :), 'x', 'green');
plot(Cont_E(1, :), Cont_E(2, :), 'green', 'LineWidth', 2);
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


%%MED



%%GED


%%GED



%%MAP


%%MAP



%%KNN

function ab = K5NN_AB(p1)
    [ab,~] = KNN(5,p1)
end

function cde = K5NN_CDE(p1)
    [~,cde] = KNN(5,p1)
end

function minPoints = Points(k,p1,class)
    [~,minPoints] = kmin(k,p1,class);
end

function [ab,cde] = KNN(k,p1)
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
    minDist = intmax*ones(length(k),1);
    minPoints = zeros(length(k),1);

    for i = 1:size(Dataset,2)
        m = Dataset(:,i);
        newDist = euclidDist(p1,m);
        if newDist<minDist(k)
            n=k-1;
            while newDist<minDist(n) && n>0
                minDist(n+1)=minDist(n);
                minPoints(n+1)=minPoints(n);
                n=n-1;
            end
            minDist(n+1) = newDist;
            minPoints(n+1) = i;

           end
    end
end

function eucDi = euclidDist(p1,p2)
	eucDi = sqrt((p1(1)-p2(1))^2+(p1(2)-p2(2))^2);
end

%%NN
function ab = NN_AB(p1)
    [ab,~] = KNN(1,p1)
end

function cde = NN_CDE(p1)
    [~,cde] = KNN(5,p1)
end
