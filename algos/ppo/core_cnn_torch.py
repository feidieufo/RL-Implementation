import torch
from torch.distributions import Normal, MultivariateNormal
import numpy as np

def initialize_weights(mod, initialization_type, scale=1):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                torch.nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


class ActorCnnEmb(torch.nn.Module):
    def __init__(self, s_dim, emb_dim=512):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(s_dim[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )

        h = self.conv2d_size_out(self.conv2d_size_out(
            self.conv2d_size_out(s_dim[1], kernel_size=8, stride=4), 4, 2), 3, 1)
        w = self.conv2d_size_out(self.conv2d_size_out(
            self.conv2d_size_out(s_dim[2], kernel_size=8, stride=4), 4, 2), 3, 1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(h*w*64, emb_dim),
            torch.nn.ReLU(),)

    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, s):
        cnn = self.cnn(s)
        x = cnn.view(s.shape[0], -1)
        x = self.fc(x)
        return x

class Critic(torch.nn.Module):
    def __init__(self, s_dim, emb_dim=512):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 1),
        )

        initialize_weights(self.fc, "orthogonal", scale=1)

    def forward(self, s_emb):
        # s_emb = self.emb(s_emb)
        v = self.fc(s_emb)
        return v


class Actor(torch.nn.Module):
    def __init__(self, s_dim, a_dim, a_max=1, emb_dim=512):
        super().__init__()
        self.act_dim = a_dim
        self.a_max = a_max
        self.emb = ActorCnnEmb(s_dim)

        self.mu = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, a_dim),
            torch.nn.Tanh()
        )
        self.var = torch.nn.Parameter(torch.zeros(1, a_dim))

        initialize_weights(self.emb, "orthogonal", scale=np.sqrt(2))
        initialize_weights(self.mu, "orthogonal", scale=0.01)

    def forward(self, s):
        emb = self.emb(s)
        mu = self.mu(emb)*self.a_max

        return mu, torch.exp(self.var)

    def select_action(self, s):
        with torch.no_grad():
            mean, std = self(s)

            normal = Normal(mean, std)
            action = normal.sample()                  # [1, a_dim]
            action = torch.squeeze(action, dim=0)     # [a_dim]

        return action

    def log_pi(self, s, a):
        mean, std = self(s)

        normal = Normal(mean, std)
        logpi = normal.log_prob(a)
        logpi = torch.sum(logpi, dim=1)
        entropy = normal.entropy().sum(dim=1)

        return logpi, entropy                       # [None,]

class ActorDisc(torch.nn.Module):
    def __init__(self, s_dim, a_num, emb_dim=512):
        super().__init__()
        self.act_num = a_num
        self.emb = ActorCnnEmb(s_dim)

        self.final = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, a_num),
            # torch.nn.Softmax()
        )

        initialize_weights(self.emb, "orthogonal", scale=np.sqrt(2))
        initialize_weights(self.final, "orthogonal", scale=0.01)

    def forward(self, s):
        emb = self.emb(s)
        x = self.final(emb)

        return x

    def select_action(self, s):
        with torch.no_grad():
            x = self(s)
            normal = torch.distributions.Categorical(logits=x)
            action = normal.sample()
            action = torch.squeeze(action, dim=0)

        return action

    def log_pi(self, s, a):
        x = self(s)
        import torch.nn.functional as F
        softx = F.softmax(x, dim=1)

        normal = torch.distributions.Categorical(logits=x)
        logpi = normal.log_prob(a)

        return logpi                      # [None,]

class PPO(torch.nn.Module):
    def __init__(self, state_dim, act_dim, act_max, epsilon, device, lr_a=0.001,
                 c_en=0.01, c_vf=0.5, max_grad_norm=-1, anneal_lr=False, train_steps=1000,):
        super().__init__()
        if type(act_dim) == np.int64 or type(act_dim) == np.int:
            self.actor = ActorDisc(state_dim, act_dim).to(device)
            self.old_actor = ActorDisc(state_dim, act_dim).to(device)
        else:
            self.actor = Actor(state_dim, act_dim[0], act_max).to(device)
            self.old_actor = Actor(state_dim, act_dim[0], act_max).to(device)
        self.critic = Critic(state_dim).to(device)
        self.epsilon = epsilon
        self.c_en = c_en
        self.c_vf = c_vf

        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr

        self.opti = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr_a, eps=1e-5)

        if anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.opti_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opti, lr_lambda=lam)

    def train_ac(self, s, a, adv, vs, oldv, is_clip_v=True):
        self.opti.zero_grad()
        logpi, entropy = self.actor.log_pi(s, a)
        old_logpi, _ = self.old_actor.log_pi(s, a)

        ratio = torch.exp(logpi - old_logpi)
        surr = ratio * adv
        clip_adv = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        aloss = -torch.mean(torch.min(surr, clip_adv))
        # loss_entropy = torch.mean(torch.exp(logpi) * logpi)
        loss_entropy = entropy.mean()
        kl = torch.mean(old_logpi - logpi)

        emb = self.actor.emb(s)
        v = self.critic(emb)
        v = torch.squeeze(v, 1)

        if not is_clip_v:
            v_loss = ((v - vs) ** 2).mean()
        else:
            clip_v = oldv + torch.clamp(v - oldv, -self.epsilon, self.epsilon)
            v_max = torch.max(((v - vs) ** 2), ((clip_v - vs) ** 2))
            v_loss = v_max.mean()

        loss = aloss - loss_entropy*self.c_en + v_loss*self.c_vf
        loss.backward()
        if self.max_grad_norm != -1:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opti.step()

        info = dict(vloss=v_loss.item(), aloss=aloss.item(), entropy=loss_entropy.item(), kl=kl.item(), loss=loss.item())
        return info

    def lr_scheduler(self):
        if self.anneal_lr:
            self.opti_scheduler.step()

    def getV(self, s):
        with torch.no_grad():
            emb = self.actor.emb(s)
            v = self.critic(emb)    # [1,1]

        return torch.squeeze(v)     #

    def update_a(self):
        self.old_actor.load_state_dict(self.actor.state_dict())