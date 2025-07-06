clear variables;
close all;
clc;

%% Defining the Environment in MATLAB

% Model and block path
model = 'MLProject_RL';  
agentBlk = 'MLProject_RL/RL Agent';  % Path to the RL Agent block

% Define observation and action spaces
obsInfo = rlNumericSpec([2 1], 'LowerLimit', [-inf; -inf], 'UpperLimit', [inf; inf]);
obsInfo.Name = 'observations';

actInfo = rlNumericSpec([1 1], 'LowerLimit', -20, 'UpperLimit', 20); 
actInfo.Name = 'force';

% Create environment
env = rlSimulinkEnv(model, agentBlk, obsInfo, actInfo);

%% Creating the RL Agent

% Critic Network

statePath = [
    featureInputLayer(2,'Normalization','none','Name','state')  % Observation input
    fullyConnectedLayer(24,'Name','fc1_state')
    reluLayer('Name','relu1_state')
    fullyConnectedLayer(24,'Name','fc2_state')
    reluLayer('Name','relu2_state')];

actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action_input')  % Action input
    fullyConnectedLayer(24,'Name','fc1_action')];

criticPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','relu3_critic')
    fullyConnectedLayer(1,'Name','output_critic')];

% Combine into critic representation
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, criticPath);

% Connect state path to addition layer
criticNetwork = connectLayers(criticNetwork,'relu2_state','add/in1');

% Connect action path to addition layer
criticNetwork = connectLayers(criticNetwork,'fc1_action','add/in2');

% Representation
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, ...
    'Observation', {'state'}, 'Action', {'action_input'}, criticOpts);

% Actor Network 

actorNetwork = [
    featureInputLayer(2,'Normalization','none','Name','state')  % Observation input
    fullyConnectedLayer(24,'Name','fc1_actor')
    reluLayer('Name','relu1_actor')
    fullyConnectedLayer(24,'Name','fc2_actor')
    reluLayer('Name','relu2_actor')
    fullyConnectedLayer(1,'Name','output_actor')
    tanhLayer('Name','tanh')
    scalingLayer('Name','scale','Scale',20)];  % Match Saturation [-20, 20]

actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
    'Observation', {'state'}, 'Action', {'scale'}, actorOpts);

% DDPG Agent

agentOpts = rlDDPGAgentOptions(...
    'SampleTime', 0.1,...
    'ExperienceBufferLength', 1e5,...
    'DiscountFactor', 0.99,...
    'MiniBatchSize', 64);

agent = rlDDPGAgent(actor, critic, agentOpts);

%% Training

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',500,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',500);  % You can adjust this

% Start training
trainingStats = train(agent, env, trainOpts);