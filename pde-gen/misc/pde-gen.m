// %% Burgers_test.m -- an executable m-file for solving a partial differential equation
// % Automatically created in CHEBGUI by user Prateek Bhustali.
// % Created on April 22, 2020 at 18:43.

// %% Problem description.
// % Solving
// %   u_t = -u*(u) + (0.01/pi)*u_xx,
// % for x in [-1,1] and t in [0,1], subject to
// %   u = 0 at x = -1
// % and
// %   u = 0 at x = 1

// %% Problem set-up
// % Create an interval of the space domain...
// %...and specify a sampling of the time domain:
// % Make the right-hand side of the PDE.
// % Assign boundary conditions.
// % Construct a chebfun of the space variable on the domain,
// %% Setup preferences for solving the problem.
// %% Call pde23t to solve the problem.
// %% Plot the solution.
// % and of the initial condition.
// % surf(u)

tic
dom = [-1,1];
t = 0:.01:0.99;

pdefun = @(t,x,u) -u.*diff(u)+0.01./pi.*diff(u,2);

bc.left = 0;
bc.right = 0;

x = chebfun(@(x) x, dom);
u0 = -sin(8*pi*x);

opts = pdeset('Eps', 1e-7, 'HoldPlot', 'on', 'Ylim', [0,0.8]);

[t, u] = pde23t(pdefun, t, u0, bc, opts);

figure
xlabel('x'), ylabel('t')
toc

x = (linspace(-1,1,256));
time = 1:100;

usol(:,time) = u(x,time);
surf(usol)


filename = 'burgers_shock_IC_sin8pi.mat';

save(filename,'t','x','usol')