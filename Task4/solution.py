import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(NeuralNetwork, self).__init__()

        # Define input layer
        self.input_layer = nn.Linear(input_dim, hidden_size)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)
        ])

        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_dim)

        # Activation function
        if activation == 'relu':
            self.activation = torch.nn.functional.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.nn.functional.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")



    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        # Forward pass through input layer
        x = self.activation(self.input_layer(s))

        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Forward pass through output layer
        output = self.output_layer(x)

        return output
    

class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network. 
        # Take a look at the NeuralNetwork class in utils.py. 

        #this network shall output the mean mu and std of the distrib, so we output 2*action_dim (first half is mu, second half is std))
        self.actor_network = NeuralNetwork(self.state_dim, 2*self.action_dim, self.hidden_size, self.hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor, 
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        action_mu, action_log_std = torch.split(self.actor_network(state), self.action_dim, dim=1)
        action_log_std = self.clamp_log_std(torch.log(abs(action_log_std)))
        action_std = torch.exp(action_log_std)
        action_dist = Normal(action_mu, action_std)
        if deterministic:
            action = action_dist.sample()
        else:
            action = action_dist.rsample()   ## using rsample over sample to add noise is possible? (add exploration factor - optional)
        log_prob = action_dist.log_prob(action)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6) #reparam from SAC paper appendix
        log_prob = log_prob.sum(1, keepdim=True) #in case of batchprocess need scalar for the loss
        action = torch.tanh(action) #squeezing in [-1,1]

        assert action.shape == (state.shape[0], self.action_dim) and \
            log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.

        # note: will need to give as in put the torch.cat of [state, action] along dim 1
        # critic network outputs action value.
        self.critic_network = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

class ValueStateEstimate(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int, value_lr: int, state_dim: int = 3, 
                    device: torch.device = torch.device('cpu')):
        super(ValueStateEstimate, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.value_lr = value_lr
        self.state_dim = state_dim
        self.device = device
        self.setup_value_estimate()

    def setup_value_estimate(self):

        self.value_network = NeuralNetwork(self.state_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=self.value_lr)



class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need.   

        #using SAC paper's values
        self.gamma = 0.999999
        self.alpha = 3*1e-4
        self.beta = 3*1e-4
        self.reward_entropy_scale = 2
        self.tau = 0.017 #soft update parameter

        self.actor = Actor(256, 2, self.alpha, self.state_dim, self.action_dim, self.device)
        self.actor_network = self.actor.actor_network

        self.critic_1 = Critic(256, 2, self.beta, self.state_dim, self.action_dim, self.device)
        self.critic_network_1 = self.critic_1.critic_network
        self.critic_2 = Critic(256, 2, self.beta, self.state_dim, self.action_dim, self.device)
        self.critic_network_2 = self.critic_2.critic_network

        self.value_state_estimate = ValueStateEstimate(256, 2, self.beta, self.state_dim, self.device)
        self.value_network = self.value_state_estimate.value_network

        self.target_value_state_estimate = ValueStateEstimate(256, 2, self.beta, self.state_dim, self.device)
        self.target_value_network = self.target_value_state_estimate.value_network

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        # action, _ = self.actor.get_action_and_log_prob(torch.tensor(s, dtype=torch.float32, device=self.device), train).cpu().detach().numpy()
        action, _ = self.actor.get_action_and_log_prob(torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0), train)
        action = action.cpu().detach().numpy()
        action = action[0]
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic, ValueStateEstimate], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward(retain_graph=True)
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork, 
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)
        # if self.memory.mem_cntr < self.batch_size:
        #     print("oom")
        #     return
        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        r_batch = torch.tensor(r_batch, dtype=torch.float32).to(self.actor.device).view(-1)
        state = torch.tensor(s_batch, dtype=torch.float32).to(self.actor.device)
        action = torch.tensor(a_batch, dtype=torch.float32).to(self.actor.device)
        next_state = torch.tensor(s_prime_batch, dtype=torch.float32).to(self.actor.device)

        value = self.value_network(state).view(-1)
        target_value = self.target_value_network(next_state).view(-1)

        actions, log_probs = self.actor.get_action_and_log_prob(state, deterministic=True)
        log_probs = log_probs.view(-1)

        # TODO: Implement Critic(s) update here.
        critic_value_1 = self.critic_network_1(torch.cat([state, actions], dim=1)).view(-1) ## actions from actor here
        critic_value_2 = self.critic_network_2(torch.cat([state, actions], dim=1)).view(-1)
        critic_value_overall = torch.min(critic_value_1, critic_value_2).view(-1)

        value_target_estimate = critic_value_overall - log_probs
        value_loss = 0.5*nn.MSELoss()(value, value_target_estimate.detach())    
        self.run_gradient_update_step(self.value_state_estimate, value_loss)

        # TODO: Implement Policy update here
        actions, log_probs = self.actor.get_action_and_log_prob(state, deterministic=False)
        log_probs = log_probs.view(-1)
        critic_value_1 = self.critic_network_1(torch.cat([state, actions], dim=1)).view(-1) ## actions from actor here
        critic_value_2 = self.critic_network_2(torch.cat([state, actions], dim=1)).view(-1)
        critic_value_overall = torch.min(critic_value_1, critic_value_2).view(-1)
        actor_loss = (log_probs - critic_value_overall).mean()
        self.run_gradient_update_step(self.actor, actor_loss)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = torch.add(self.reward_entropy_scale * r_batch, self.gamma * target_value)

        q1_old = self.critic_network_1(torch.cat([state, action], dim=1)).view(-1)  ## action from buffer here
        q2_old = self.critic_network_2(torch.cat([state, action], dim=1)).view(-1)

        critic_1_loss = 0.5*nn.MSELoss()(q1_old, q_hat.detach())
        critic_2_loss = 0.5*nn.MSELoss()(q2_old, q_hat.detach())  ## not using the function because we reuse common loss
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.critic_target_update(self.value_network, self.target_value_network, self.tau, True)

# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")
    
    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()