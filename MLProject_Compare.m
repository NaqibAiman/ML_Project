% Load and set up the agent
load('trainedAgent.mat');  
agentObj = agent;          

% Simulate PID Model 
open_system('MLProject_PID');
simOut_PID = sim('MLProject_PID');

theta_pid = simOut_PID.simout.signals.values;
time_pid  = simOut_PID.simout.time;

info_pid = stepinfo(theta_pid, time_pid);
mse_pid = mean((theta_pid).^2);  % target theta = 0

% Simulate RL Model
open_system('MLProject_RL');
simOut_RL = sim('MLProject_RL');

% Access 'simout1' from the simOut_RL object
theta_rl = simOut_RL.simout1.signals.values(:,1);  
time_rl  = simOut_RL.simout1.time;

info_rl = stepinfo(theta_rl, time_rl);
mse_rl  = mean((theta_rl).^2);

% Display Results
disp('--- PID Controller ---');
disp(info_pid);
disp(['MSE (PID): ', num2str(mse_pid)]);

disp('--- RL Controller ---');
disp(info_rl);
disp(['MSE (RL): ', num2str(mse_rl)]);