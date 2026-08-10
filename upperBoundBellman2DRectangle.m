function v = upperBoundBellman2DRectangle(x,y)
% function v = upperBoundBellman2DRectangle(x,y)
%
% upper bound on optimal value (lower bound on Bellman function)
% for 2D problem with control in the disc
% computes value of Bellman function for control in the best rotated centered rectangle
% x,y are the 2D arguments
% the best rectangle is sought by a two-level algorithm
% on the lower level the best side ratio is sought by the golden ratio method
% on the upper level the best angle is sought by the golden ratio method
% the value has to be maximized over the side ratio
tol = 10^(-13);
% test for collinearity
nxy = norm([x; y]);
if (nxy < tol)
    % x = y = 0
    v = 0;
    return;
end
nx = norm(x);
if nx < tol
    % x = 0
    v = pureBellman1D(0,norm(y));
    return;
end
if (det([x,y])/nxy^2 < tol)
    % x,y collinear
    v = pureBellman1D(nx,(x'*y)/nx);
    return;
end
h = 2^(-6);
P = (0:h:1)*pi/2;
F = zeros(1,length(P));
for k = 1:length(P)
    phi = P(k);
    F(k) = upperBoundBellman2DRotatedRectangle(x,y,phi);
end
[~,maxind] = max(F);
h = P(2)-P(1);
phimin = P(maxind) - h;
phimax = P(maxind) + h;
gr = (sqrt(5)-1)/2;
grp = 1 - gr;
phil = phimin*gr + phimax*grp;
phir = phimax*gr + phimin*grp;
f = [upperBoundBellman2DRotatedRectangle(x,y,phimin), upperBoundBellman2DRotatedRectangle(x,y,phil), upperBoundBellman2DRotatedRectangle(x,y,phir), upperBoundBellman2DRotatedRectangle(x,y,phimax)];
while phimax - phimin > 10^(-10)
    if (f(1) >= f(2)) || (f(3) <= f(4))
        break;
    end
    if f(2) < f(3)
        f = f([2 3 3 4]);
        phimin = phil;
        phil = phir;
        phir = phimax*gr + phimin*grp;
        f(3) = upperBoundBellman2DRotatedRectangle(x,y,phir);
    else
        f = f([1 2 2 3]);
        phimax = phir;
        phir = phil;
        phil = phimin*gr + phimax*grp;
        f(2) = upperBoundBellman2DRotatedRectangle(x,y,phil);
    end
end
v = max(f);
