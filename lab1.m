clf(figure(1))
clf(figure(2))


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
title("Classes A and B");
hold on
scatter(Class_A(1, :), Class_A(2, :),'x', 'red');
plot(Cont_A(1, :), Cont_A(2, :), 'red', 'LineWidth', 2);
scatter(Class_B(1, :), Class_B(2, :), 'x', 'blue');
plot(Cont_B(1, :), Cont_B(2, :), 'blue', 'LineWidth', 2);
hold off

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


%%GED



%%MAP


%%MAP



%%KNN

%%Use KNN(5,p1)

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


function output = boundary(classifier, start, finish)
    x = linspace(start(1), finish(1), 100);
    y = linspace(start(2), finish(2), 100);
    [X, Y] = meshgrid(x,y);
    A = arrayfun(classifier, X, Y);
    contour(X, Y, A, 1.5);
end

function output = classify(classifier, data)
    for i = data(1)
        for j = data(2)
            output = arrayfun(classifier, i, j);
        end
    end
end

function err = errorRate(output)

end
