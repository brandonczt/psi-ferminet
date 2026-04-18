# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for FermiNet."""

import functools
import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


# Axis name we pmap over.
PMAP_AXIS_NAME = "qmc_pmap_axis"

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

# Shortcut for kfac utils
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
_all_gather_if_pmap = kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather)
all_gather = lambda x: _all_gather_if_pmap(x, PMAP_AXIS_NAME)


def replicate_all_local_devices(obj, axis_name=PMAP_AXIS_NAME):
    try:
        return kfac_jax.utils.replicate_all_local_devices(obj, axis_name=axis_name)
    except AttributeError as exc:
        if "device_put_replicated" not in str(exc):
            raise
        mesh_axis_name = axis_name or "replica_axis"
        devices = jax.local_devices()
        mesh = jax.sharding.Mesh(np.array(devices), (mesh_axis_name,))
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(mesh_axis_name)
        )
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(jnp.stack([x] * len(devices)), sharding), obj
        )
