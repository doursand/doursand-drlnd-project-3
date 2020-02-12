# Description of the implementation

## Algorithm
Multi-Agent Deep Deterministic Policy Gradients (MADDPG) was implemented to solve this environment, as it is particularly suitable for continous action spaces like in the example of the Tennis environment

## Neural networks
Like in the case of the Reacher where I also used the DDPG algorithm, it basically consists on two separated neural networks (and hence in the case of Tennis, we have 2 x 2 neural networks) :

- One network called the "Actor", predicts the next action for the agent (in a deterministic way) from a given state
- One network called the "Critic", evaluate the Q value associated to the state and the predicted action

Each of these networks maintain a current version (that we are training using experience buffer) and a target version which are updated on a regular basis using a soft update mechanism (meaning that only a small portion of the weights from the current networks are transfer to the target network)

The Actor and the Critic neural net use the same architecture in my implementation, ie using two hidden layer of 256 and 128 neurons , batch normalization on the first hidden layer, and Relu for the activation function

## Agent setup in the context of multi agent collaboration/competition
The agents are setup to learn from a common experience buffer shared between the two agents, meaning that we are collecting a given amount of experiences tuples ("state", "action", "reward", "next_state", "done") before using mini-batches randomly selected from this experience buffer in order to train the current version of the neural networks for each agent.
To avoid stationary learning, we are using as well random noise (Ornstein-Uhlenbeck process), so that we keep on exploring new possibilities

## hyperparameters

The following hyperparameters led to the solving of the environment in XXX episodes

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256        # minibatch size
- GAMMA = 0.99             # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-3        # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic
- WEIGHT_DECAY = 0       # L2 weight decay
- LEARNING_STEPS = 10    # steps after which we can trigger learning
- LEARNING_ITER = 10     # amount of time we learn once LEARNING_STEPS is reached


two specific hyperparameters were instrumental to the quick convergence of the model :

- in OUNoise class, decrease the parameter sigma to 0.1 (instead of 0.2)
- in the step() method of the Agent , the agent was modified in such a way that it is learning 10 times in a row (from random subset of the replay buffer) , and this each 10 steps (this led to drastically boosting the learning of the neural networks)

## Graph
![graph](graph.jpg)

## Potential improvements

Apparently applying prioritized experience replay could further boost the learning of the MADDPG algorithm, it could be interesting to confirm this in a next release of this work
[Prioritized Replay paper](https://cardwing.github.io/files/RL_course_report.pdf)
