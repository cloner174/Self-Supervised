# in the name of God
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeighter(nn.Module):
    """
    A module to compute dynamic weights for Joint and Motion streams.
    It takes features from both streams and outputs weights indicating
    the relative importance of each stream for a given sample.
    """
    
    def __init__(self, dim, hidden_dim=64):
        super(DynamicWeighter, self).__init__()
        # Input dimension is dim*2 because we concatenate features from two streams
        self.fc1 = nn.Linear(dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2) # Output two scores, one for Joint, one for Motion
    
    def forward(self, f_joint, f_motion):
        # f_joint and f_motion are features from query encoders of each stream
        # Detach them to prevent this module from affecting the encoder's gradients
        combined_features = torch.cat([f_joint.detach(), f_motion.detach()], dim=1)
        
        x = F.relu(self.fc1(combined_features))
        scores = self.fc2(x)
        
        # Use softmax to get weights that sum to 1
        weights = F.softmax(scores, dim=1)
        
        # Return weights for joint and motion streams separately
        w_joint = weights[:, 0].unsqueeze(1)
        w_motion = weights[:, 1].unsqueeze(1)
        return w_joint, w_motion
    
