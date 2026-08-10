function v = upperBoundBellman2DAngleRectangle(x,y,phi,xi)
% function v = upperBoundBellman2DAngleRectangle(x,y,phi,xi)
%
% upper bound on optimal value (lower bound on Bellman function)
% for 2D problem with control in the disc
% computes value of Bellman function for control in a rotated by angle phi
% centered rectangle with corner (cos(xi),sin(xi))
% x,y are the 2D arguments
assert((xi >= 0) && (xi <= pi/2))
c = cos(phi);
s = sin(phi);
O = [c, -s; s, c];
x = O*x;
y = O*y;
if xi == 0
    if (x(2) == 0) && (y(2) == 0)
        v = scaledBellman1D(x(1),y(1),cos(xi));
    else
        v = +Inf;
    end
elseif xi == pi/2
    if (x(1) == 0) && (y(1) == 0)
        v = scaledBellman1D(x(2),y(2),sin(xi));
    else
        v = +Inf;
    end
else
    v = scaledBellman1D(x(1),y(1),cos(xi)) + scaledBellman1D(x(2),y(2),sin(xi));
end
