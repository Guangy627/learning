import torch 
import torch.nn as nn
from torch.nn import functional as F

class GRPO(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(GRPO, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64,1)
        )
    def get_logprob(self, obs, act):
        logits = self.policy(obs)
        dist  = torch.distributions.Categorical(logits = logits)
        logp  = dist.log_prob(act)
        v = self.value(obs).squeeze(-1)
        return logp, v, dist.entropy().mean()
    
    @staticmethod
    def _group_mean(x, group_ids):
        unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
        G = unique_ids.numel()
        device = x.device

        sum_per_group = torch.zeros(G, device=device).scatter_add_(0, inverse, x)
        count_per_group = torch.zeros(G, device=device).scatter_add_(0, inverse, torch.ones_like(x))
        mean_per_group = sum_per_group / (count_per_group + 1e-8)
        return mean_per_group[inverse]
    
    def grpo_loss(self, obs, act, rewards, group_ids, logp_old, ref_model, clip_eps, ent_coef, beta_kl):
        logp_new, _, entropy = self.get_logprob(obs, act)

        with torch.no_grad():
            b = self._group_mean(rewards, group_ids)
            adv = rewards - b 
            #这里可以标准化
            gstd = torch.sqrt(self._group_mean(adv**2, group_ids) + 1e-8)
            adv = adv / gstd

            #or 跨batch标准化
            adv = adv - adv.mean() / (adv.std() + 1e-8)

        # ppo clip
        ratio =  torch.exp(logp_new - logp_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0-clip_eps,1+clip_eps) *adv
        pg_loss = -torch.min(surr1, surr2).mean()

        #kl散度
        kl_loss = torch.tensor(0.0,device = obs.device)
        if (ref_model is not None) and beta_kl > 0.0:
            with torch.no_grad():
                ref_logits = ref_model.policy(obs) #ref policy logits
            logp_ref = F.log_softmax(ref_logits,dim = -1).gather(1, act.view(-1,1)).squeeze(1)
            kl = (logp_ref - logp_new).mean()
            kl_loss = beta_kl * kl

        #entropy canonical
        ent_loss = -ent_coef * entropy

        #total loss

