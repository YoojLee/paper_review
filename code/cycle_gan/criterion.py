from torch import nn

class AdversarialLoss(nn.Module):
    def __init__(self, mode:str):
        """
        두 개의 bce loss가 필요함.
        one for D(G(x)) vs real label (Generator)

        one for D(y) vs real label and D(G(x)) vs fake label (Discriminator)

        - Args
            mode: 'g' or 'd'
        """
        super(AdversarialLoss, self).__init__()
        assert mode=='g' or mode=='d'
        self.mode = mode
        self.loss = nn.BCEWithLogitsLoss()

    # generator
    def forward_G(self, d_g_x, real):
        return self.loss(d_g_x, real)

    # discriminator
    def forward_D(self, d_y ,real, d_g_x, fake): # d_g_x는 detach되어야 함. 주의할 것.
        d_real_loss = self.loss(d_y, real)
        d_fake_loss = self.loss(d_g_x.detach(), fake)

        d_loss = (d_real_loss + d_fake_loss)/2

        return d_loss
        

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, x, y, f_g_x, g_f_y):
        loss_cyc = self.loss(x,g_f_y) + self.loss(y, f_g_x)

        return loss_cyc