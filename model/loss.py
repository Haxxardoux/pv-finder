import torch

# how to define our cost function
class Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-5):
        '''
        Epsilon is a parameter that can be adjusted.
        '''

        self.epsilon = epsilon

    def forward(self, x, y):
        # Make a boolean mask of non-nan values of the target histogram
        valid = ~torch.isnan(y)

        # Compute r, only including non-nan values. r will probably be shorter than x and y.
        r = torch.abs((x[valid] + self.epsilon) / (y[valid] + self.epsilon))

        # Compute -log(2r/(r² + 1))
        alpha = -torch.log(2*r / (r**2 + 1))

        # Sum up the alpha values, and divide by the length of x and y. Note this is not quite
        # a .mean(), since alpha can be a bit shorter than x and y due to masking.
        beta = alpha.sum() / 4000

        return beta
