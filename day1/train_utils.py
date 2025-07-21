import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


# GPU Selection Logic
def select_gpu(gpu=None,min_free_memory_gb=22):
    if not gpu==None:
        print(f'cuda:{gpu}')
        return f'cuda:{gpu}'
    if not torch.cuda.is_available():
        return 'cpu'
    
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
        if free >= min_free_memory_gb:
            torch.cuda.set_device(i)
            print(f'cuda:{i}')
            return f'cuda:{i}'
    
    print("No suitable GPU found, using CPU")
    return 'cpu'






def calculate_pr_auc(loader, model, device, config):
    model.eval()
    all_targets = []
    all_probs = []
    
    # For expert load trackings
    fusion_expert_loads = []
    classification_expert_loads = []
    import torch.nn.functional as F

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, unit='batch', desc='Evaluating val auprc'):
            metadata = batch['metadata'].to(device)
            image = batch['image'].to(device)

            target = batch['target'].to(device)
            
            outputs = model(metadata, image=image)
            
            # Store expert weights
            try:
                fusion_expert_loads.append(outputs['fusion_weights'].cpu())
                classification_expert_loads.append(outputs['expert_weights'].cpu())
            except:
                None
            probs = F.softmax(outputs, dim=1)
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())
    
    # Calculate PR-AUCs
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()


    # class_fractions = np.average(class_counts.numpy()) / class_counts.numpy() 
    
    pr_aucs = []
    weighted_pr_aucs=[]
    for class_idx in range(len(config['classes'])):  # CLASSES = list of class names/indices
        precision, recall, _ = precision_recall_curve(
            all_targets[:, class_idx],  # Binary targets for this class
            all_probs[:, class_idx]     # Predicted probabilities for this class
        )
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        
        # Weight by class fraction
        # weighted_pr_auc = pr_auc / class_fractions[class_idx]
        # weighted_pr_aucs.append(weighted_pr_auc)




    
    # Calculate expert load statistics
    try:
        fusion_loads = torch.cat(fusion_expert_loads).mean(dim=0).numpy()
        classification_loads = torch.cat(classification_expert_loads).mean(dim=0).numpy()
    
        return np.mean(pr_aucs), pr_aucs, fusion_loads, classification_loads
    except:
        return np.mean(pr_aucs), pr_aucs, [0,0,0], [0,0,0]

def calculate_val_loss(loader, model, criterion, DEVICE):
    """Calculate validation loss using the same criterion as training.
    
    Args:
        loader: DataLoader for validation data
        model: Model to evaluate
        criterion: Loss function (same as used in training)
        DEVICE: Device to run calculations on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating val loss', unit='batch', leave=False):
            metadata = batch['metadata'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            
            outputs = model(metadata, image)
            loss = criterion(outputs, target)
            val_loss += loss.item()
    
    return val_loss / len(loader)




def get_class_counts(dataloader, config):
    num_classes = len(config['classes'])  # Always use coarse classes
    counts = torch.zeros(num_classes, dtype=torch.long)
    
    for batch in dataloader:
        targets = batch['target']
        # For one-hot encoded targets (assuming your targets are one-hot)
        if targets.dim() == 2:
            counts += targets.sum(dim=0).long()
        # For class index targets (alternative format)
        else:
            class_indices = targets.argmax(dim=1) if targets.dim() == 2 else targets
            counts += torch.bincount(class_indices, minlength=num_classes)
    # print(counts)
    
    return counts