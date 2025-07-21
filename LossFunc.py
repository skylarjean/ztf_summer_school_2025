import torch
import math


def GreatCircleLoss(pred, target):
    # Convert normalized predictions/targets to radians
    # Consistent with your accuracy calculation:
    # [-1, 1] -> [0, 2π] for longitude and [-π/2, π/2] for latitude
    l1 = (pred[:, 0] + 1) * math.pi       # [0, 2π]
    b1 = pred[:, 1] * (math.pi/2)         # [-π/2, π/2]
    l2 = (target[:, 0] + 1) * math.pi     # [0, 2π]
    b2 = target[:, 1] * (math.pi/2)       # [-π/2, π/2]

    # Haversine formula with numerical stability
    dlon = l2 - l1
    dlat = b2 - b1
    
    # More numerically stable version
    a = torch.sin(dlat/2)**2 + torch.cos(b1) * torch.cos(b2) * torch.sin(dlon/2)**2
    
    # Safe sqrt with clamping
    a = torch.clamp(a, 0.0, 1.0)  # Ensures a is always in valid range
    c = 2 * torch.asin(torch.sqrt(a))
    
    # Convert to degrees
    distance = c * 180 / math.pi
    
    return torch.mean(distance)
    



def GreatCircleLoss_no_average(pred, target):
    # Convert normalized predictions/targets to radians
    # pred and target are assumed to be in range [-1, 1] for l and b
    l1 = pred[:, 0] * math.pi      # longitude (λ₁) in radians [-π, π]
    b1 = pred[:, 1] * (math.pi/2)  # latitude (φ₁) in radians [-π/2, π/2]
    l2 = target[:, 0] * math.pi    # longitude (λ₂)
    b2 = target[:, 1] * (math.pi/2) # latitude (φ₂)

    # Haversine formula for great-circle distance
    dlon = l2 - l1
    dlat = b2 - b1
    a = torch.sin(dlat/2)**2 + torch.cos(b1) * torch.cos(b2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a)-1e-5)
    
    c = c*180/math.pi

    # Return mean angular distance (in degrees)
    return c
