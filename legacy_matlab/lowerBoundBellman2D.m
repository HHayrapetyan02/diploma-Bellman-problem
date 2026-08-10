function v = lowerBoundBellman2D(x,y)
% function v = lowerBoundBellman2D(x,y)
%
% lower bound on optimal value (upper bound on Bellman function)
% for 2D problem with control in the disc
% computes value of Bellman function for control in a unit square [-1,1]
% the best square rotated by an angle phi is taken into account
% the minimum is sought with the golden ratio method
% the value has to be minimized over the angle
% x,y are the 2D arguments
P = (0:0.01:0.5)*pi;
F = zeros(1,length(P));
for k = 1:length(P)
    phi = P(k);
    F(k) = lowerBoundBellman2DAngle(x,y,phi);
end
[~,minind] = min(F);
h = P(2)-P(1);
phimin = P(minind) - h;
phimax = P(minind) + h;
gr = (sqrt(5)-1)/2;
grp = 1 - gr;
phil = phimin*gr + phimax*grp;
phir = phimax*gr + phimin*grp;
f = [lowerBoundBellman2DAngle(x,y,phimin), lowerBoundBellman2DAngle(x,y,phil), lowerBoundBellman2DAngle(x,y,phir), lowerBoundBellman2DAngle(x,y,phimax)];
while phimax - phimin > 10^(-10)
    if (f(1) <= f(2)) || (f(3) >= f(4))
        break;
    end
    if f(2) > f(3)
        f = f([2 3 3 4]);
        phimin = phil;
        phil = phir;
        phir = phimax*gr + phimin*grp;
        f(3) = lowerBoundBellman2DAngle(x,y,phir);
    else
        f = f([1 2 2 3]);
        phimax = phir;
        phir = phil;
        phil = phimin*gr + phimax*grp;
        f(2) = lowerBoundBellman2DAngle(x,y,phil);
    end
end
v = min(f);
