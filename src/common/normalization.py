import numpy as np

def get_field_info(field_type):
    """Get normalization parameters for each field type."""
    info = {
        'rho': {
            'center': 1.0,
            'scale': 1.0,
            'range': (0.0, 2.0)
        },
        'ux': {
            'center': 0.0,
            'scale': 160.0,
            'range': (-160.0, 160.0)
        },
        'uy': {
            'center': 0.0,
            'scale': 160.0,
            'range': (-160.0, 160.0)
        },
        'uz': {
            'center': 0.0,
            'scale': 160.0,
            'range': (-160.0, 160.0)
        }
    }
    return info[field_type]

def normalize_field(data, field_type):
    """Normalize field data to [-1, 1] range."""
    info = get_field_info(field_type)
    return (data - info['center']) / info['scale']

def denormalize_field(data, field_type):
    """Denormalize field data back to original range."""
    info = get_field_info(field_type)
    return data * info['scale'] + info['center']

def normalize_batch(batch, field_types):
    """Normalize a batch of data."""
    normalized = np.zeros_like(batch)
    for i, field_type in enumerate(field_types):
        if len(batch.shape) == 4:  # Batch of 3D data
            normalized[:, :, :, i] = normalize_field(batch[:, :, :, i], field_type)
        else:  # Single 3D data
            normalized[:, :, i] = normalize_field(batch[:, :, i], field_type)
    return normalized

def denormalize_batch(batch, field_types):
    """Denormalize a batch of data."""
    denormalized = np.zeros_like(batch)
    for i, field_type in enumerate(field_types):
        if len(batch.shape) == 4:  # Batch of 3D data
            denormalized[:, :, :, i] = denormalize_field(batch[:, :, :, i], field_type)
        else:  # Single 3D data
            denormalized[:, :, i] = denormalize_field(batch[:, :, i], field_type)
    return denormalized