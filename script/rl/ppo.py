'''
PPO
GAE广义优势函数+policy loss function+ value loss function+entropy bonus
'''
import torch
import torch.nn as nn

class PPO(nn.Module):
    def __init__(self, obs_dim, act_dim):
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64,act_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim,64), nn.Tanh(),
            nn.Linear(64,1)
        )
    def get_logprob(self, obs, act):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits = logits)
        logp = dist.log_prob(act)
        v = self.value(obs).squeeze(-1)
        return logp, v, dist.entropy()
    
def compute_gae(rewards,values,dones,gamma,lamda): #done is 1 if episode else 0
    T = len(rewards)
    adv = [0]*T
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * nonterminal -values[t]
        gae = delta + gamma * lamda * nonterminal * gae
        adv[t] = gae
    v_targets = adv+values[:-1]
    return adv[:-1],v_targets if len(values) == T else (adv, adv+v_targets)

def ppo_loss(model, obs, act, logp_old, adv, v_target,clip_eps, vf_coef, ent_coef):
    logp, v, entropy = model.get_logprob(obs, act)
    ratio  = torch.exp(logp - logp_old) #新策略/旧策略

    #clip
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0- clip_eps, 1+clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    #value loss
    v_clipped = v_target + (v - v_target).clamp(-clip_eps, clip_eps)
    vf_loss = torch.max((v - v_target)**2, (v_clipped - v_target)**2).mean()

    #entropy bonus
    ent_loss = -entropy.mean()

    #total loss
    loss =  policy_loss + vf_coef * vf_loss + ent_coef * ent_loss
    return loss


