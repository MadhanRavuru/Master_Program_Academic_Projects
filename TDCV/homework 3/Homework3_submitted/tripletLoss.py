import torch

class loss_maker(torch.nn.Module):
    
    def __init__(self, margin):
        super(loss_maker,self).__init__()
        self.margin = torch.tensor(margin)
        
    def forward(self, anchor, puller, pusher):
        temp = torch.ones([anchor.shape[0]], dtype=torch.float32)
        diff_pos = torch.sum((anchor - puller)**2,1)
        diff_neg = torch.sum((anchor - pusher)**2,1)
       
        diff = temp - (diff_neg/(diff_pos + self.margin*temp))
        loss_t = torch.mean(torch.clamp(diff, min=0.0))
        loss_p = torch.mean(diff_pos)
        loss = loss_t + loss_p
        
        return loss