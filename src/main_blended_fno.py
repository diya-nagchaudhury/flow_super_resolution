import jax 
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.ops import index_update, index
from jax.experimental.optimizers import adam, sgd
from jax import lax
from jax import ops
from jax import random
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from ..SciREX.scirex.core.sciml.fno.models.fno_2d import FNO2d

# Path: src/main_blended_fno.py

def mse(pred, true):
    return jnp.mean(jnp.abs(pred - true)**2)

def l2_error(pred, true):
    return jnp.sqrt(jnp.mean(jnp.abs(pred - true)**2))

def l2_error_batch(pred, true):
    return jnp.sqrt(jnp.mean(jnp.abs(pred - true)**2, axis=(1, 2)))

def relative_error(pred, true):
    return jnp.sqrt(jnp.mean(jnp.abs(pred - true)**2)) / jnp.sqrt(jnp.mean(jnp.abs(true)**2))

def relative_error_batch(pred, true):
    return jnp.sqrt(jnp.mean(jnp.abs(pred - true)**2, axis=(1, 2))) / jnp.sqrt(jnp.mean(jnp.abs(true)**2, axis=(1, 2)))

def grad_mse(pred, true):
    return grad(mse)(pred, true)

def grad_l2_error(pred, true):
    return grad(l2_error)(pred, true)

def grad_l2_error_batch(pred, true):
    return grad(l2_error_batch)(pred, true)

def grad_relative_error(pred, true):
    return grad(relative_error)(pred, true)

def grad_relative_error_batch(pred, true):
    return grad(relative_error_batch)(pred, true)

def grad_mse(pred, true):
    return grad(mse)(pred, true)



