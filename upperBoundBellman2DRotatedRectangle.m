function v = upperBoundBellman2DRotatedRectangle(x,y,phi)
% function v = upperBoundBellman2DRotatedRectangle(x,y,phi)
%
% upper bound on optimal value (lower bound on Bellman function)
% for 2D problem with control in the disc
% computes value of Bellman function for control in the centered rectangle
% rotated by phi and with best side ratio
% x,y are the 2D arguments
% the best side ratio is sought by the golden ratio method
% the value has to be maximized over the side ratio
tol = 10^(-14);
c = cos(phi);
s = sin(phi);
O = [c, -s; s, c];
x = O*x;
y = O*y;
if (abs(x(1)) < tol) && (abs(y(1)) < tol)
    v = pureBellman1D(x(2),y(2));
    return;
end
if (abs(x(2)) < tol) && (abs(y(2)) < tol)
    v = pureBellman1D(x(1),y(1));
    return;
end
h = 2^(-6);
while true
    X = (0:h:1)*pi/2;
    X(1) = [];
    X(end) = [];
    F = zeros(1,length(X));
    for k = 1:length(X)
        xi = X(k);
        F(k) = upperBoundBellman2DAngleRectangle(x,y,0,xi);
    end
    if (F(1) < F(2)) && (F(end-1) > F(end))
        % function decreases to -infty at the ends
        break;
    end
    h = h/2;
end
[~,maxind] = max(F);
h = X(2)-X(1);
ximin = X(maxind) - h;
ximax = X(maxind) + h;
gr = (sqrt(5)-1)/2;
grp = 1 - gr;
xil = ximin*gr + ximax*grp;
xir = ximax*gr + ximin*grp;
f = [upperBoundBellman2DAngleRectangle(x,y,0,ximin), upperBoundBellman2DAngleRectangle(x,y,0,xil), upperBoundBellman2DAngleRectangle(x,y,0,xir), upperBoundBellman2DAngleRectangle(x,y,0,ximax)];
while ximax - ximin > 10^(-10)
    if (f(1) >= f(2)) || (f(3) <= f(4))
        break;
    end
    if f(2) < f(3)
        f = f([2 3 3 4]);
        ximin = xil;
        xil = xir;
        xir = ximax*gr + ximin*grp;
        f(3) = upperBoundBellman2DAngleRectangle(x,y,0,xir);
    else
        f = f([1 2 2 3]);
        ximax = xir;
        xir = xil;
        xil = ximin*gr + ximax*grp;
        f(2) = upperBoundBellman2DAngleRectangle(x,y,0,xil);
    end
end
v = max(f);
