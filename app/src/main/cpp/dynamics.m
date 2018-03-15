clc;close all;

syms u_0__ u_1__ u_2__ u_3__ 
syms x_0__ x_1__ x_2__ x_3__ x_4__ x_5__ 
syms mass_gravity 
U = [ u_0__ u_1__ u_2__ u_3__ ].';
X = [ x_0__ x_1__ x_2__ x_3__ x_4__ x_5__ ].';
XU = [ X ; U ];

F = sym(zeros(6,1));

u = @(i) U(i+1);
x = @(i) X(i+1);
f = @(i) F(i+1);

thrust = u(0);
phi = u(1);
theta = u(2);   
psi = u(3);

F = [ x(3) 
      x(4) 
      x(5)   
     -thrust * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
      thrust * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta))
     -thrust * cos(phi) * cos(theta) + mass_gravity ];
 
 J = jacobian( F, XU );
 jj = J(:);
 jac = jj(jj ~= 0);