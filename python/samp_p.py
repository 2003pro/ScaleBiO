import torch
import torch.nn.functional as F

class SampProb(torch.nn.Module):
    
    def __init__(self, num_partitions):
        super(SampProb, self).__init__()
        self.num_partitions=num_partitions
        self.p=torch.nn.Parameter(torch.zeros(num_partitions))

    def forward(self, per_sample_loss, per_sample_catogary):
        # calculate weighted average loss
        samp_prob=F.softmax(self.p, dim=0)
        w=samp_prob[per_sample_catogary]
        weight=torch.mul(1.0/torch.bincount(per_sample_catogary)[per_sample_catogary], w)
        loss=torch.dot(weight, per_sample_loss)
        return loss

    @property
    def prob(self):
        return F.softmax(self.p, dim=0).detach().cpu()