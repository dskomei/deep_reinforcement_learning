import torch
import torch.nn.functional as F
import torch.optim as optim
from continuous.models import ActorNet, CriticNet, ClipedCriticNet
from continuous.utils import hard_update, soft_update, convert_network_grad_to_false


class ActorCriticModel(object):

    def __init__(self, state_dim, action_dim, action_scale, args, device):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.target_update_interval = args['target_update_interval']

        self.device = device

        self.actor_net = ActorNet(input_dim=state_dim, output_dim=action_dim, hidden_dim=args['hiden_dim'], action_scale=action_scale).to(self.device)
        self.critic_net = CriticNet(input_dim=state_dim + action_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)
        self.critic_net_target = CriticNet(input_dim=state_dim + action_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _ = self.actor_net.sample(state)
        else:
            _, action = self.actor_net.sample(state)
        return action.cpu().detach().numpy().reshape(-1)

    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, _ = self.actor_net.sample(next_state_batch)
            next_q_values_target = self.critic_net_target(next_state_batch, next_action)
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q_values = self.critic_net(state_batch, action_batch)
        critic_loss = F.mse_loss(q_values, next_q_values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, _ = self.actor_net.sample(state_batch)
        q_values = self.critic_net(state_batch, action)
        actor_loss = - q_values.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)

        return critic_loss.item(), actor_loss.item()


class ActorClipedCriticModel(object):

    def __init__(self, state_dim, action_dim, action_scale, args, device):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.target_update_interval = args['target_update_interval']

        self.device = device

        self.actor_net = ActorNet(input_dim=state_dim, output_dim=action_dim, hidden_dim=args['hiden_dim'], action_scale=action_scale).to(self.device)
        self.critic_net = ClipedCriticNet(input_dim=state_dim + action_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)
        self.critic_net_target = ClipedCriticNet(input_dim=state_dim + action_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _ = self.actor_net.sample(state)
        else:
            _, action = self.actor_net.sample(state)
        return action.cpu().detach().numpy().reshape(-1)

    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, _ = self.actor_net.sample(next_state_batch)
            next_q_values_target1, next_q_values_target2 = self.critic_net_target(next_state_batch, next_action)
            next_q_values_target_min = torch.min(next_q_values_target1, next_q_values_target2)
            next_q_value = reward_batch + mask_batch * self.gamma * next_q_values_target_min

        q_values1, q_values2 = self.critic_net(state_batch, action_batch)
        critic_loss1 = F.mse_loss(q_values1, next_q_value)
        critic_loss2 = F.mse_loss(q_values2, next_q_value)
        critic_loss = critic_loss1 + critic_loss2

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, _ = self.actor_net.sample(state_batch)

        q_values1, q_values2 = self.critic_net(state_batch, action)
        q_values_min = torch.min(q_values1, q_values2)

        actor_loss = - q_values_min.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)

        return critic_loss.item(), actor_loss.item()
