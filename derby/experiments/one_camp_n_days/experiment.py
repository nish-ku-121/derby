import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.environments import OneCampaignNDaysEnv
from derby.core.pmfs import PMF

logger = logging.getLogger(__name__)


class OneCampNDaysExperiment:

    def __init__(self, seed: int | None = None):
        # Reset class-level UID generators so that spec and campaign IDs are predictable
        # within each experiment run. This avoids cross-run drift when multiple processes
        # or repeated runs occur in the same Python interpreter.
        try:
            import itertools
            from derby.core.basic_structures import AuctionItemSpecification
            from derby.core.ad_structures import Campaign
            AuctionItemSpecification._uid_generator = itertools.count(1)
            Campaign._uid_generator = itertools.count(1)
        except Exception:
            # Best-effort; if modules are not yet imported, continue.
            pass

        # Optional global seeding for reproducibility
        # Only performed if seed is not None to preserve prior stochastic behavior by default.
        self.seed = seed
        if seed is not None:
            try:
                import random
                random.seed(seed)
            except Exception:
                pass
            try:
                np.random.seed(seed)
            except Exception:
                pass
            try:
                tf.random.set_seed(seed)
            except Exception:
                pass
            # Help make hashing deterministic in some Python ops
            os.environ.setdefault('PYTHONHASHSEED', str(seed))
            logger.info("[Experiment] Seed set to %s", seed)
        self.auction_item_specs = [
                        AuctionItemSpecification(name="male", item_type={"male"}),
                        AuctionItemSpecification(name="female", item_type={"female"})
        ]
        self.campaigns = [
                        Campaign(10, 100, self.auction_item_specs[0]),
                        Campaign(10, 100, self.auction_item_specs[1])
        ]
        self.auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0]: 1,
                    self.auction_item_specs[1]: 1
        })
        self.campaign_pmf = PMF({
                    self.campaigns[0]: 1,
                    self.campaigns[1]: 1
        })
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)

    def build_one_segment_setup(self):
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0]: 0,
                    self.auction_item_specs[1]: 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0]: 0,
                    self.campaigns[1]: 1
        })
        for c in campaigns:
            logger.debug("campaign: %s", c)

        env = OneCampaignNDaysEnv(
            self.first_price_auction,
            auction_item_spec_pmf,
            campaign_pmf,
            1,
            2,
        )
        return env, auction_item_spec_ids

    def build_two_segment_setup(self):
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0]: 1,
                    self.auction_item_specs[1]: 1
        })
        auction_item_spec_ids = [spec.uid for spec in self.auction_item_specs]
        campaign_pmf = PMF({
                    self.campaigns[0]: 1,
                    self.campaigns[1]: 1
        })
        for c in campaigns:
            logger.debug("campaign: %s", c)

        env = OneCampaignNDaysEnv(
            self.first_price_auction,
            auction_item_spec_pmf,
            campaign_pmf,
            1,
            2,
        )
        return env, auction_item_spec_ids

    def get_states_scaler_descaler(self, samples):
        """
        :param samples: an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of states of the environment.
        """
        samples_shape = samples.shape
        samples = samples.reshape(-1, samples_shape[-1])
        states_scaler = MinMaxScaler()
        states_scaler.fit(samples)

        def scale_states_func(states):
            states_reshp = states.reshape(-1, samples_shape[-1])
            scaled_states = states_scaler.transform(states_reshp)
            return scaled_states.reshape(states.shape)

        def descale_states_func(states):
            states_reshp = states.reshape(-1, samples_shape[-1])
            descaled_states = states_scaler.inverse_transform(states_reshp)
            return descaled_states.reshape(states.shape)

        return states_scaler, scale_states_func, descale_states_func

    def get_actions_scaler_descaler(self, samples):
        """
        :param samples: an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of states of the environment.
        """
        samples_shape = samples.shape
        samples = samples.reshape(-1, samples_shape[-1])
        budget_samples = samples[:, 1:2]
        actions_scaler = MinMaxScaler()
        actions_scaler.fit(budget_samples)

        def descale_actions_func(scaled_actions):
            sa_without_ais = scaled_actions[:, :, :, 1:]
            sa_reshaped = sa_without_ais.reshape(-1, sa_without_ais.shape[-1])
            bid_per_item = sa_reshaped[:, 0:1]
            total_limit = sa_reshaped[:, 1:2]
            bid_per_item_descaled = actions_scaler.inverse_transform(bid_per_item)
            total_limit_descaled = actions_scaler.inverse_transform(total_limit)
            descaled = np.concatenate([bid_per_item_descaled, total_limit_descaled], axis=1)
            descaled_actions_without_ais = descaled.reshape(sa_without_ais.shape)
            descaled_actions = np.concatenate((scaled_actions[:, :, :, 0:1], descaled_actions_without_ais), axis=3)
            return descaled_actions

        def scale_actions_func(descaled_actions):
            da_without_ais = descaled_actions[:, :, :, 1:]
            da_reshaped = da_without_ais.reshape(-1, da_without_ais.shape[-1])
            bid_per_item = da_reshaped[:, 0:1]
            total_limit = da_reshaped[:, 1:2]
            bid_per_item_scaled = actions_scaler.transform(bid_per_item)
            total_limit_scaled = actions_scaler.transform(total_limit)
            scaled = np.concatenate([bid_per_item_scaled, total_limit_scaled], axis=1)
            scaled_actions_without_ais = scaled.reshape(da_without_ais.shape)
            scaled_actions = np.concatenate((descaled_actions[:, :, :, 0:1], scaled_actions_without_ais), axis=3)
            return scaled_actions

        return actions_scaler, scale_actions_func, descale_actions_func

    def build_env_transforms(self, env):
        samples = env.get_states_samples(10000)
        _, scale_states_func, _ = self.get_states_scaler_descaler(samples)
        actions_scaler, scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(samples)
        return scale_states_func, actions_scaler, scale_actions_func, descale_actions_func
