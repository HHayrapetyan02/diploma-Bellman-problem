function v = lowerBoundBellman2DAngle(x,y,phi)
% function v = lowerBoundBellman2DAngle(x,y,phi)
%
% lower bound on optimal value (upper bound on Bellman function)
% for 2D problem with control in the disc
% computes value of Bellman function for control in a unit square [-1,1]
% rotated by an angle phi
% x,y are the 2D arguments
c = cos(phi);
s = sin(phi);
O = [c, -s; s, c];
x = O*x;
y = O*y;
v = pureBellman1D(x(1),y(1)) + pureBellman1D(x(2),y(2));
