from torch import nn

class AdversarialLoss(nn.Module):
    def __init__(self):
        """
        두 개의 bce loss가 필요함.
        one for D(G(x)) vs real label (Generator)

        one for D(y) vs real label and D(G(x)) vs fake label (Discriminator)
        """
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    # generator
    def forward_G(self, d_g_x, real):
        """
        - Args
            d_g_x: D(G(x)), a predicted probability of G(x) to be real
            real: true value (real label)
        """
        return self.loss(d_g_x, real)

    # discriminator
    def forward_D(self, d_y ,real, d_g_x, fake):
        """
        - Args
            d_y: D(y) a predicted probability that y is real
            real: true value (real label)
            d_g_x: D(G(x)), a predicted probability that y is true
            fake: true value (fake label)
        """
        d_real_loss = self.loss(d_y, real)
        d_fake_loss = self.loss(d_g_x.detach(), fake) # D(G(x)) to be detached in order to prevent a gradient.

        d_loss = (d_real_loss + d_fake_loss)/2

        return d_loss
        

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss, self).__init__()
        self.loss_forward = nn.L1Loss()
        self.loss_backward = nn.L1Loss()
    
    def forward(self, x, y, f_g_x, g_f_y):
        """
        x, y -> true
        f_g_x, g_f_y -> pred
        """
        loss_cyc = self.loss_forward(f_g_x, x) + self.loss_backward(g_f_y, y) # pred, real의 순서 잘 지켜줄 것

        return loss_cyc

class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.loss_x = nn.L1Loss()
        self.loss_y = nn.L1Loss()

    def forward(self, x, y, f_y, g_x):
        loss_idt = self.loss_x(f_y, x) + self.loss_y(g_x, y)

        return loss_idt