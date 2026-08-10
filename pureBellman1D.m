function v = pureBellman1D(x,y)
% function v = pureBellman1D(x,y)
%
% computes 1D Bellman function for Fuller problem with interval [-1,+1]
% x,y are the arguments of the Bellman function
% the function is the negative of the cost, hence nonpositive
bet = 0.444623560185937; % 36\beta^4 + 3\beta^2 - 2 = 0
gam = 6.753024861778741e-02; % gamma = \frac{-\beta^{2}+2\beta-\frac{2}{3}}{10(1-2\beta)^{\frac{3}{2}}}
if x >= -bet*y*abs(y)
    v = -x^2*y/2-x*y^3/3-y^5/15-gam*(y^2+2*x)^(5/2);
else
    v = x^2*y/2-x*y^3/3+y^5/15-gam*(y^2-2*x)^(5/2);
end
