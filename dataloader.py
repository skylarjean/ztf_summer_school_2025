import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import unittest
import tempfile
import shutil
from collections import defaultdict
import random
import math
import inspect
from timm.models import convnext
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from tqdm import tqdm
# import h5py



# print(inspect.signature(convnext.ConvNeXt.__init__))
# CLASSES = [['AGN'], ['Tidal Disruption Event'], ['SN Ia'], ['SN Ic'], ['SN IIP'], ['SN IIn'],['SN II'],['SN Ib'], ['Cataclysmic']]
# CLASSES = [['AGN'], ['Tidal Disruption Event'], ['SN Ia'], ['SN Ic', 'SN IIP', 'SN IIn','SN II','SN Ib'], ['Cataclysmic']]
# CLASSES = [ ['SN Ia', 'SN Ic', 'SN Ib'],[ 'SN IIP', 'SN IIn','SN II'], ['Cataclysmic']]
# config['classes'] = [['AGN', 'Tidal Disruption Event'],[ 'SN Ia','SN Ic','SN Ib'], ['SN IIP', 'SN IIn','SN II'], ['Cataclysmic']]
SHOW_CLASSES =["Nuclear", "SN I's", "SN II's", "Cataclysmic"]
# CLASSES = [['AGN', 'Tidal Disruption Event'], ['SN Ia'], ['SN Ic','SN Ib', 'SN IIP', 'SN IIn','SN II'], ['Cataclysmic']]
# CLASSES = [['AGN', 'Tidal Disruption Event'], ['SN Ia', 'SN Ic', 'SN IIP', 'SN IIn','SN II','SN Ib', 'Cataclysmic']]



# SHOW_CLASSES =['AGN', 'Tidal Disruption Event', 'SN Ia', 'SN Ic', 'SN IIP', 'SN IIn','SN II','SN Ib', 'Cataclysmic']

# SHOW_CLASSES =['AGN', 'Tidal Disruption Event', 'SN Ia', 'other SN', 'Cataclysmic']
# SHOW_CLASSES = ['Nuclear', 'everything else']
# SHOW_CLASSES = ['everything else', 'cataclysmic']
SHOW_CLASSES =["Nuclear", "SN I's", "SN II's", "Cataclysmic"]
# SHOW_CLASSES =[ "SN I's", "SN II's", "Cataclysmic"]
# SHOW_CLASSES =['Nuclear', "SN Ia", "other SN", 'Cataclysmic']


class AstroDataset(Dataset):

    def __init__(self, config, transform=None, embedding=False, frequency=20):
        self.npy_dir = config['npy_dir']
        self.file_names = sorted([f for f in os.listdir(config['npy_dir']) if f.endswith('.npy')])
        self.transform = transform
        self.classes = config['classes']
        self.eps = 1e-8  # Small value to prevent division by zero
        self.frequency=frequency
        self.embedding= embedding
        self.config= config
        

        # Metadata normalization statistics
        self.metadata_stats = {
            'sgscore1': {'mean': 0.236, 'std': 0.266},  # 0
            'sgscore2': {'mean': 0.401, 'std': 0.328},   # 1
            'distpsnr1': {'mean': 3.151, 'std': 3.757},  # 2
            'distpsnr2': {'mean': 9.269, 'std': 6.323},  # 3
            'nmtchps': {'mean': 9.231, 'std': 8.089},    # 4
            'sharpnr': {'mean': 0.253, 'std': 0.256},    # 5
            'scorr': {'mean': 22.089, 'std': 16.757},    # 6
            'ra': {'mean': 0, 'std': 1},                 # 7
            'dec': {'mean': 0, 'std': 1},                # 8
            'diffmaglim': {'mean': 20.17, 'std': 0.535}, # 9
            'sky': {'mean': 0.0901, 'std': 2.58},        # 10
            'ndethist': {'mean': 18.7, 'std': 27.83},    # 11
            'ncovhist': {'mean': 1144.9, 'std': 1141.27}, # 12
            'sigmapsf': {'mean': 0.996, 'std': 0.0448},  # 13
            'chinr': {'mean': 3.257, 'std': 21.5},       # 14
            'magpsf': {'mean': 18.64, 'std': 0.936},     # 15
            'nnondet': {'mean': 1126, 'std': 1140},      # 16
            'classtar': {'mean': 0.95, 'std': 0.08},     # 17
            'filter_id': {'mean': 1.62, 'std': 0.57},     # 18
            'days_since_peak': {'mean': 359.59, 'std': 627.91},   # 19
            'days_to_peak': {'mean': 90.93, 'std': 327.51},   # 20
            'age': {'mean': 450.5, 'std': 734.29},          # 21
            'peakmag_so_far': {'mean': 17.945, 'std': 0.896}, # 22
            'maxmag_so_far': {'mean': 20.37, 'std': 0.6325}, # 23
        }


        # Build object index mapping
        self.obj_info = []
        self.obj_id_to_indices = defaultdict(list)
        
        for idx in tqdm(range(len(self.file_names))):
            sample = self._load_sample(idx)
            obj_id = sample.get('obj_id', str(idx))
            self.obj_info.append({
                'obj_id': obj_id,
                'all_indices': []  # Will be populated later
            })
            self.obj_id_to_indices[obj_id].append(idx)
        
        # Populate all_indices
        [self.obj_info[idx].update({'all_indices': indices}) for indices in self.obj_id_to_indices.values() for idx in indices]
        self.samples = [self._load_sample(i) for i in tqdm(range(len(self.file_names)))]

    def _load_sample(self, idx):
        """Load sample directly from disk"""
        file_path = os.path.join(self.npy_dir, self.file_names[idx])
        return np.load(file_path, allow_pickle=True).item()
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        config = self.config
        sample = self._load_sample(idx)
        import torch
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # Assuming sample['metadata']['ra'] and sample['metadata']['dec'] are in degrees
        # ra = sample['metadata']['ra'] 
        # dec = sample['metadata']['dec'] 

        # Convert RA/Dec (equatorial) to galactic coordinates

        # l, b  =   equatorial_to_galactic(ra, dec)# Galactic longitude (0 to 360)

        
        # Normalize metadata
        metadata = torch.tensor([
            (sample['metadata']['sgscore1'] - self.metadata_stats['sgscore1']['mean']) / self.metadata_stats['sgscore1']['std'],
            (sample['metadata']['sgscore2'] - self.metadata_stats['sgscore2']['mean']) / self.metadata_stats['sgscore2']['std'],
            (sample['metadata']['distpsnr1'] - self.metadata_stats['distpsnr1']['mean']) / self.metadata_stats['distpsnr1']['std'],
            (sample['metadata']['distpsnr2'] - self.metadata_stats['distpsnr2']['mean']) / self.metadata_stats['distpsnr2']['std'],
            (sample['metadata']['nmtchps'] - self.metadata_stats['nmtchps']['mean']) / self.metadata_stats['nmtchps']['std'],
            (sample['metadata']['sharpnr'] - self.metadata_stats['sharpnr']['mean']) / self.metadata_stats['sharpnr']['std'],
            (sample['metadata']['scorr'] - self.metadata_stats['scorr']['mean']) / self.metadata_stats['scorr']['std'],
            sample['metadata']['l'] /180 -1,  # Already normalized
            sample['metadata']['b'] /90,  # Already normalized
            (sample['metadata']['diffmaglim'] - self.metadata_stats['diffmaglim']['mean']) / self.metadata_stats['diffmaglim']['std'],
            (sample['metadata']['sky'] - self.metadata_stats['sky']['mean']) / self.metadata_stats['sky']['std'],
            (sample['metadata']['ndethist'] - self.metadata_stats['ndethist']['mean']) / self.metadata_stats['ndethist']['std'],
            (sample['metadata']['ncovhist'] - self.metadata_stats['ncovhist']['mean']) / self.metadata_stats['ncovhist']['std'],
            (sample['metadata']['sigmapsf'] - self.metadata_stats['sigmapsf']['mean']) / self.metadata_stats['sigmapsf']['std'],
            (sample['metadata']['chinr'] - self.metadata_stats['chinr']['mean']) / self.metadata_stats['chinr']['std'],
            (sample['metadata']['magpsf'] - self.metadata_stats['magpsf']['mean']) / self.metadata_stats['magpsf']['std'],
            (sample['metadata']['nnondet'] - self.metadata_stats['nnondet']['mean']) / self.metadata_stats['nnondet']['std'],
            (sample['metadata']['classtar'] - self.metadata_stats['classtar']['mean']) / self.metadata_stats['classtar']['std'],
            (sample['metadata']['filter_id'] - self.metadata_stats['filter_id']['mean']) / self.metadata_stats['filter_id']['std'],
            (sample['metadata']['days_since_peak'] - self.metadata_stats['days_since_peak']['mean']) / self.metadata_stats['days_since_peak']['std'],
            (sample['metadata']['days_to_peak'] - self.metadata_stats['days_to_peak']['mean']) / self.metadata_stats['days_to_peak']['std'],
            (sample['metadata']['age'] - self.metadata_stats['age']['mean']) / self.metadata_stats['age']['std'],
            (sample['metadata']['peakmag_so_far'] - self.metadata_stats['peakmag_so_far']['mean']) / self.metadata_stats['peakmag_so_far']['std'],
            (sample['metadata']['maxmag_so_far'] - self.metadata_stats['maxmag_so_far']['mean']) / self.metadata_stats['maxmag_so_far']['std'],
            (sample['metadata']['maxmag_so_far'] / sample['metadata']['peakmag_so_far']),
        ], dtype=torch.float32)

        # Get image and normalize per-channel

        image = torch.tensor(sample['images'][7:57, 7:57, :], dtype=torch.float32).permute(2, 0, 1)


        # Create radial distance channel [1,H,W] normalized to [0,1]
        if self.embedding:
            height, width = image.shape[1], image.shape[2]
            y_coords = torch.linspace(-1, 1, height, dtype=torch.float32)
            x_coords = torch.linspace(-1, 1, width, dtype=torch.float32)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            radial_distance = torch.sqrt(x_grid**2 + y_grid**2) / math.sqrt(2)  # Normalized [0,1]
            radial_distance = radial_distance.unsqueeze(0)  # [1,H,W]

        
            
            
            # # Combine with radial distance (already normalized [0,1]

            # # Convert RA/Dec (degrees) to Galactic (l, b)
            # ra_deg = sample['metadata']['ra']  # Assuming RA is in degrees
            # dec_deg = sample['metadata']['dec']  # Assuming Dec is in degrees
            # coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
            # galactic = coord.galactic  # Convert to galactic coordinates

            # height, width = image.shape[1], image.shape[2]

            # # Normalized RA/Dec (input coordinates)
            # ra_norm = sample['metadata']['l'] / 360.0  # [0, 1]
            # dec_norm = sample['metadata']['b']  / 90  # [0, 1]

            # # Base grid (normalized image coords)
            # y_coords = torch.linspace(0, 1, height, dtype=torch.float32)
            # x_coords = torch.linspace(0, 1, width, dtype=torch.float32)
            # yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # # Encode RA/Dec as AMPLITUDE + FREQUENCY modulation
            # freq_base = self.frequency  # Base spatial frequency (tune this!)

            # # RA channel: Amplitude = RA, Frequency scales with Dec
            # ra_amplitude = 1  # [0, 1]
            # ra_freq = freq_base * (0.5 + dec_norm)  # Frequency varies with Dec
            # ra_wave = ra_amplitude * torch.sin(2 * np.pi * ra_freq * xx).unsqueeze(0)

            # # Dec channel: Amplitude = Dec, Frequency scales with RA
            # dec_amplitude = 1  # [0, 1]
            # dec_freq = freq_base * (0.5 + ra_norm)  # Frequency varies with RA
            # dec_wave = dec_amplitude * torch.sin(2 * np.pi * dec_freq * yy).unsqueeze(0)


        # Normalize IMAGE channels relative to each other
        # Make sure the tensor is contiguous before reshaping
        image = image.contiguous()
        
        # Compute median of all 3 image channels combined
        combined_median = torch.median(image.reshape(-1))
        
        # Compute per-channel median and normalize
        for c in range(3):
            channel_median = torch.median(image[c].reshape(-1))
            image[c] = image[c] - channel_median
            image[c] = image[c]/(image[c].std()+ self.eps)

        # combined_std = image.std()
        # image = image / (combined_std + self.eps)

        if self.transform:
            image = self.transform(image)

        if self.embedding:
            # Combine all channels: [science, ref, diff, radial_dist, l_pos, b_pos]
            image = torch.cat([image, radial_distance], dim=0)  # Now [6,H,W]
            # image = torch.cat([image, ra_wave], dim=0)
            # image = torch.cat([image, dec_wave], dim=0)


        
        # Target tensor
        target = torch.zeros(len(config['classes']), dtype=torch.float32)
        original_class = sample['target']
        for idx, category in enumerate(config['classes']):
            if original_class in category:
                target[idx] = 1.0
                break
        
        return {
            'metadata': metadata,
            'image': image,
            'target': target,
            'obj_id': sample.get('obj_id', str(idx))
        }

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None, random_alert_per_epoch=False):
        self.subset = subset
        self.transform = transform
        self.random_alert_per_epoch = random_alert_per_epoch
        self.obj_id_to_indices = defaultdict(list)
        
        if random_alert_per_epoch:
            if isinstance(subset, Subset):
                base_dataset = subset.dataset
                for i, idx in enumerate(subset.indices):
                    if hasattr(base_dataset, 'obj_info'):
                        obj_id = base_dataset.obj_info[idx]['obj_id']
                        self.obj_id_to_indices[obj_id].append(i)
            elif hasattr(subset, 'obj_info'):
                for idx in range(len(subset)):
                    obj_id = subset.obj_info[idx]['obj_id']
                    self.obj_id_to_indices[obj_id].append(idx)
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx, config):
        if self.random_alert_per_epoch and self.obj_id_to_indices:
            if isinstance(self.subset, Subset):
                original_idx = self.subset.indices[idx]
                obj_id = self.subset.dataset.obj_info[original_idx]['obj_id']
            else:
                obj_id = self.subset.obj_info[idx]['obj_id']
            
            possible_indices = self.obj_id_to_indices[obj_id]
            selected_idx = random.choice(possible_indices)
            sample = self.subset[selected_idx]
        else:
            sample = self.subset[idx]
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample

def get_augmentation_transforms():
    return transforms.Compose([
        # Geometric transforms first
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=(-2, -1))),  # 90°
            transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=(-2, -1))),  # 270°
        ], p=0.75),

        # Normalization last
        # transforms.Normalize(mean=[0.014770,0.014882,0.000599], std=[0.003890, 0.003294, 0.015838])

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=9, sigma=(0.0005, 0.0005)),
        ], p=1),
    ])



def get_dataloaders(train_npy_dir, batch_size=32, seed=33, include_spectra=False, 
                   include_phot=False, random_alert_per_epoch=False, num_workers=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = get_augmentation_transforms()
    full_dataset = AstroDataset(train_npy_dir, transform=None, include_spectra=include_spectra)
    
    # Create stratified splits
    labels = [torch.argmax(full_dataset[i]['target']).item() for i in range(len(full_dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_val_indices = next(sss.split(np.zeros(len(labels)), labels))
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_indices, test_indices = next(sss.split(np.zeros(len(test_val_indices)), 
                                  [labels[i] for i in test_val_indices]))
    
    # Convert relative indices to absolute indices
    val_indices = [test_val_indices[i] for i in val_indices]
    test_indices = [test_val_indices[i] for i in test_indices]
    
    dataset = TransformedDataset(
        Subset(full_dataset, train_indices),
        transform=transform,
        random_alert_per_epoch=random_alert_per_epoch
    )
    val_dataset = TransformedDataset(
        Subset(full_dataset, val_indices),
        transform=None,
        random_alert_per_epoch=False
    )
    test_dataset = TransformedDataset(
        Subset(full_dataset, test_indices),
        transform=None,
        random_alert_per_epoch=False
    )
    
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=not random_alert_per_epoch,
        num_workers=24, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=24
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=24
    )
    
    return train_loader, val_loader, test_loader, full_dataset.classes


class FullAlertDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        
        # Handle both Subset and direct dataset cases
        if isinstance(subset, Subset):
            self.base_dataset = subset.dataset
            self.indices = subset.indices
        else:
            self.base_dataset = subset
            self.indices = range(len(subset))
        
        # Handle ConcatDataset case
        if isinstance(self.base_dataset, ConcatDataset):
            self.datasets = self.base_dataset.datasets
            # Find the first dataset with obj_info
            for dataset in self.datasets:
                if hasattr(dataset, 'obj_info'):
                    self.obj_info = dataset.obj_info
                    break
            else:
                raise AttributeError("No dataset in ConcatDataset has obj_info")
        else:
            if not hasattr(self.base_dataset, 'obj_info'):
                raise AttributeError("Base dataset has no obj_info")
            self.obj_info = self.base_dataset.obj_info
        
        # Build object index mapping
        self.obj_id_to_indices = defaultdict(list)
        for i, idx in enumerate(self.indices):
            # For ConcatDataset, need to find which dataset the index belongs to
            if isinstance(self.base_dataset, ConcatDataset):
                dataset_idx = 0
                cumulative_size = 0
                for dataset in self.datasets:
                    if idx < cumulative_size + len(dataset):
                        break
                    cumulative_size += len(dataset)
                    dataset_idx += 1
                actual_idx = idx - cumulative_size
                obj_id = self.datasets[dataset_idx].obj_info[actual_idx]['obj_id']
            else:
                obj_id = self.obj_info[idx]['obj_id']
            self.obj_id_to_indices[obj_id].append(i)
        
        # Create flat list of all indices
        self.all_indices = []
        for obj_id in sorted(self.obj_id_to_indices.keys()):
            self.all_indices.extend(self.obj_id_to_indices[obj_id])
    
    def __len__(self):
        return len(self.all_indices)
    
    def __getitem__(self, idx):
        original_idx = self.all_indices[idx]
        return self.subset[original_idx]


def get_dataloaders(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    transform = get_augmentation_transforms()
    # Load both datasets
    print('getting dataset')
    dataset = AstroDataset( config, transform=transform, frequency=20, embedding=False)

    print('got dataset')

    
    # Create a mapping of object IDs to all their indices

    obj_id_to_indices = defaultdict(list)
    obj_id_to_class = {}
    
    # First pass: collect all object IDs and their class information
    
    for i in tqdm(range(len(dataset))):

        sample = dataset[i % len(dataset)]
        
        obj_id = sample['obj_id']
        obj_id_to_indices[obj_id].append(i)
        # Store class for stratification (use first alert's class for the object)
        if obj_id not in obj_id_to_class:
            obj_id_to_class[obj_id] = torch.argmax(sample['target']).item()
    
    # Convert to list of unique object IDs for stratification
    unique_obj_ids = list(obj_id_to_class.keys())
    obj_classes = [obj_id_to_class[obj_id] for obj_id in unique_obj_ids]
    
    # First split: train vs (val + test) at OBJECT level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=config['seed'])
    train_obj_ids, val_test_obj_ids = next(sss.split(np.zeros(len(unique_obj_ids)), obj_classes))
    
    # Convert back to actual object IDs
    train_obj_ids = [unique_obj_ids[i] for i in train_obj_ids]
    val_test_obj_ids = [unique_obj_ids[i] for i in val_test_obj_ids]
    
    # Second split: val vs test from the remaining 30%
    val_test_classes = [obj_id_to_class[obj_id] for obj_id in val_test_obj_ids]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config['seed'])
    val_obj_ids, test_obj_ids = next(sss.split(np.zeros(len(val_test_obj_ids)), val_test_classes))
    
    # Convert back to actual object IDs
    val_obj_ids = [val_test_obj_ids[i] for i in val_obj_ids]
    test_obj_ids = [val_test_obj_ids[i] for i in test_obj_ids]
    
    # Now collect all indices for each split
    train_indices = []
    for obj_id in train_obj_ids:
        train_indices.extend(obj_id_to_indices[obj_id])
    
    val_indices = []
    for obj_id in val_obj_ids:
        val_indices.extend(obj_id_to_indices[obj_id])
    
    test_indices = []
    for obj_id in test_obj_ids:
        test_indices.extend(obj_id_to_indices[obj_id])

    train_subset = FullAlertDataset(Subset(dataset, train_indices))
    val_subset = FullAlertDataset(Subset(dataset, val_indices))
    test_subset = FullAlertDataset(Subset(dataset, test_indices))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']//2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']//4, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']//4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, config['classes']
