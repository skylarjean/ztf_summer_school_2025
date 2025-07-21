import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve
import torch
import numpy as np
from sklearn.metrics import auc
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F


# Config
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define class names (only coarse now)
# CLASSES =['AGN', 'Tidal Disruption Event', 'SN Ia', 'SN Ic', 'SN IIP', 'SN IIn','SN II','SN Ib', 'Cataclysmic']
# CLASSES = ['AGN', 'Tidal Disruption Event', 'SN', 'Cataclysmic']

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#c702b6', '#2ca02c','#1f77b4', '#ff7f0e', '#2ca02c', '#c702b6', '#2ca02c']

def get_model_predictions(loader, model, DEVICE):
    """Get model predictions and targets for coarse classification"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        # model.fusion_router.set_testing_mode(True)
        for batch in loader:
            metadata = batch['metadata'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            
            
            outputs = model(metadata, image)
            
            # Get probabilities from logits using 
            probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
            targets = batch['target'].cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(targets)
    
    return np.concatenate(all_probs), np.concatenate(all_targets)


def plot_pr_curves(ax, probs, targets):
    """Plot precision-recall curves with correct color mapping"""
    colors = plt.cm.tab10.colors  # Use a built-in qualitative colormap
    
    for class_idx, class_name in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(
            targets[:, class_idx],
            probs[:, class_idx]
        )
        ax.plot(recall, precision, 
               color=colors[class_idx % len(colors)],  # Auto-scale to class count
               lw=2, 
               label=f'{class_name} (AUC={auc(recall, precision):.2f}')  # Show AUC in legend
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])



def plot_roc_curves(ax, probs, targets):
    """Plot precision-recall curves with correct color mapping"""
    colors = plt.cm.tab10.colors  # Use a built-in qualitative colormap
    
    for class_idx, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(
            targets[:, class_idx],
            probs[:, class_idx]
        )
        ax.plot(fpr, tpr, 
               color=colors[class_idx % len(colors)],  # Auto-scale to class count
               lw=2, 
               label=f'{class_name} (AUC={auc(fpr, tpr):.2f}')  # Show AUC in legend
    
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])

def plot_confusion_matrix(ax, targets, y_pred):
    """Plot normalized confusion matrix on given axis"""
    # Convert from one-hot to class indices if needed
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    
    cm = confusion_matrix(targets, y_pred, labels=range(len(CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_title("Normalized Confusion Matrix")
    tick_marks = np.arange(len(CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",  
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')



def plot_combined_results(loader, model, DEVICE, seed=None, best_model_path=None):
    """Main function to generate combined plot
    Args:
        loader: Data loader for the model
        model: Either a model object or a filepath to a saved .pth file
        seed: Optional seed for filename differentiation
    """
    # Handle case where model is a filepath to a .pth file
    if isinstance(model, str) and model.endswith('.pth'):
        # Load the model from state dict
        model_obj = torch.load(model, weights_only = False)
        if isinstance(model_obj, torch.nn.Module):
            # If the .pth file contains the full model
            model = model_obj
        else:
            # If the .pth file contains just the state dict
            # Note: You'll need to know the model architecture to properly load the state dict
            # You might want to add model_class parameter to handle this case
            raise ValueError("State dict loading requires model architecture information")
    
    # Get predictions
    probs, targets = get_model_predictions(loader, model, DEVICE)
    
    # For confusion matrix, we need class predictions
    y_pred = np.argmax(probs, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(27, 8))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Plot ROC curves 
    ax0 = plt.subplot(gs[0])
    plot_roc_curves(ax0, probs, targets)


    # Plot PR curves
    ax1 = plt.subplot(gs[1])
    plot_pr_curves(ax1, probs, targets)
    
    # Plot confusion matrix
    ax2 = plt.subplot(gs[2])
    plot_confusion_matrix(ax2, targets, y_pred)
    
    # Save and show
    plt.tight_layout()
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else model.split('/')[-1].replace('.pth', '')
    filename = f'plots/{model_name}_{seed}.png' if seed else f'plots/{model_name}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    pr_auc_mean, pr_aucs = print_metrics(loader, model, DEVICE)
    print(f'\nSaved plots under {filename}\n')
    return pr_auc_mean, pr_aucs, plt




def print_metrics(loader, model, DEVICE):
    """Print evaluation metrics"""
    # Print class distribution
    # print("\nTest set class distribution:")
    # class_counts = {i: 0 for i, name in enumerate(CLASSES)}
    
    # for batch in loader:
    #     targets = batch['target']
    #     if targets.dim() == 2:  # one-hot
    #         batch_indices = torch.argmax(targets, dim=1)
    #     else:  # class indices
    #         batch_indices = targets
        
        # for idx in range(len(CLASSES)):
        #     class_counts[idx] += (batch_indices == idx).sum().item()
    
    # for idx, name in enumerate(CLASSES):
        # print(f"{name}: {class_counts[idx]}")

    # PR-AUC
    pr_auc_mean, pr_aucs, fusion_loads, classification_loads = calculate_pr_auc(loader, model, DEVICE)
    print(f'fusion loads:{fusion_loads}')
    # print(f'classification loads: {classification_loads}')
    print(f"\nMean PR-AUC: {pr_auc_mean:.3f}")
    
    for name, auc in zip(CLASSES, pr_aucs):
        print(f"{name}: {auc:.3f}")
    return pr_auc_mean, pr_aucs



def calculate_pr_auc(loader, model, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    # For expert load trackings
    fusion_expert_loads = []
    classification_expert_loads = []
    import torch.nn.functional as F

    with torch.no_grad():
        for batch in loader:
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
            probs = F.softmax(outputs['logits'], dim=1)
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())
    
    # Calculate PR-AUCs
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()


    # class_fractions = np.average(class_counts.numpy()) / class_counts.numpy() 
    
    pr_aucs = []
    weighted_pr_aucs=[]
    for class_idx in range(len(CLASSES)):  # CLASSES = list of class names/indices
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