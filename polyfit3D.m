% poly3D.m
% routine to fit 3D polynomial. A weighted least squares fitting is used.
% Polynomial fit order from zeroth to second order.
% Mingming Wu 04.11.2016

function fit_phase = polyfit3D(phase, W_map, order)

[r, c, t] = size(phase);

% Defining the coordinate grid
[xcrd, ycrd, zcrd] = meshgrid(-(r-1)/2:1:(r-1)/2, -(c-1)/2:1:(c-1)/2, -(t-1)/2:1:(t-1)/2);

% Defining basis vectors
% zeroth order
v0 = ones(c,r,t);

% first order
v1 = xcrd;
v2 = ycrd;
v3 = zcrd;

% second order
v4 = xcrd.^2;
v5 = ycrd.^2;
v6 = zcrd.^2;
v7 = xcrd.*ycrd;
v8 = xcrd.*zcrd;
v9 = ycrd.*zcrd;


if order == 0
    X = horzcat(v0(:));
elseif order == 1
    X = horzcat(v0(:), v1(:), v2(:), v3(:));
elseif order == 2
    X = horzcat(v0(:), v1(:), v2(:), v3(:), v4(:), v5(:), v6(:), v7(:), v8(:), v9(:));
end

[r, c, t] = size(X);
W_map = repmat(W_map(:),1,c).^2;

b = inv(X.'*(W_map.*X))*X.'*(W_map(:,1).*phase(:));

if order == 0
    fit_phase = b(1).*v0;
elseif order == 1
    fit_phase = b(1).*v0 + b(2).*v1 + b(3).*v2 + b(4).*v3;
elseif order == 2
    fit_phase = b(1).*v0 + b(2).*v1 + b(3).*v2 + b(4).*v3 + b(5).*v4 + b(6).*v5 + b(7).*v6 + b(8).*v7 + b(9).*v8 + b(10).*v9;
end

end