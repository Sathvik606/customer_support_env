# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Customer Agency Env Environment Client."""

from openenv import EnvClient
from .models import Action, Observation, State

class CustomerAgencyEnv(EnvClient[Action, Observation, State]):
    """
    Client for the Customer Agency Env.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_type = Action
        self.observation_type = Observation
        self.state_type = State

