# federated_core/trainers/fedmgp_learner.py

from typing import Dict, List, Any, Optional
import torch
import copy
import os
from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights

class FedMGPLearner(BaseFederatedLearner):
    """
    FedMGP implementation with similarity-based prompt ranking and aggregation.
    Handles text and vision prompt groups with complex similarity computation.
    """

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        """Get FedMGP-specific attributes."""
        attrs = {}
        attrs['global_text_prompts'] = None
        attrs['global_vision_prompts'] = None
        attrs['text_prompt_ranks'] = {}
        attrs['vision_prompt_ranks'] = {}
        attrs['text_prompt_slots'] = {}
        attrs['vision_prompt_slots'] = {}
        attrs['text_contributions_by_rank'] = {}
        attrs['vision_contributions_by_rank'] = {}
        attrs['tcvp_meta_global_params'] = {}
        attrs['tcvp_meta_adam_m'] = {}
        attrs['tcvp_meta_adam_v'] = {}
        attrs['tcvp_meta_adam_step'] = 0
        return attrs

    def _initialize_components_storage(self) -> None:
        """Initialize FedMGP component storage."""
        self.client_weights['components'] = {
            'text_prompts': [{} for _ in range(self.cfg.DATASET.USERS)],
            'vision_prompts': [{} for _ in range(self.cfg.DATASET.USERS)],
            'tcvp_meta_params': [{} for _ in range(self.cfg.DATASET.USERS)]
        }

    def aggregate_models(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate prompts based on similarity ranking."""
        text_prompts_list = self._extract_text_prompts(client_ids)
        vision_prompts_list = self._extract_vision_prompts(client_ids)

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        group_bind_selection = getattr(self.cfg.TRAINER.FEDMGP, 'GROUP_BIND_SELECTION', False)
        slot_aggregation = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_AGGREGATION', False)
        global_component = {}
        is_first_round = self.current_round == 1

        if group_bind_selection:
            if is_first_round:
                text_prompt_ranks, vision_prompt_ranks = self._first_round_group_bind_ranking(
                    text_prompts_list, vision_prompts_list, client_ids, topk)
            else:
                text_prompt_ranks, vision_prompt_ranks = self._rank_prompt_groups_by_similarity(
                    text_prompts_list, vision_prompts_list, client_ids)
        else:
            if is_first_round:
                text_prompt_ranks, vision_prompt_ranks = self._first_round_ranking(
                    text_prompts_list, vision_prompts_list, client_ids, topk)
            else:
                text_prompt_ranks = self._rank_prompts_by_similarity(
                    text_prompts_list, client_ids, is_vision=False)
                vision_prompt_ranks = self._rank_prompts_by_similarity(
                    vision_prompts_list, client_ids, is_vision=True)

        can_use_slot_aggregation = (
            slot_aggregation
            and not is_first_round
            and self.global_text_prompts
            and self.global_vision_prompts
        )

        if can_use_slot_aggregation:
            if group_bind_selection:
                text_prompt_slots, vision_prompt_slots = self._assign_bound_prompt_slots(
                    text_prompts_list, vision_prompts_list, client_ids,
                    text_prompt_ranks, vision_prompt_ranks)
            else:
                text_prompt_slots = self._assign_prompt_slots(
                    text_prompts_list, client_ids, text_prompt_ranks,
                    self.global_text_prompts, prompt_type="text")
                vision_prompt_slots = self._assign_prompt_slots(
                    vision_prompts_list, client_ids, vision_prompt_ranks,
                    self.global_vision_prompts, prompt_type="vision")

            global_component.update(self._aggregate_text_prompts_by_slot(
                text_prompts_list, client_ids, text_prompt_ranks, text_prompt_slots))
            global_component.update(self._aggregate_vision_prompts_by_slot(
                vision_prompts_list, client_ids, vision_prompt_ranks, vision_prompt_slots))

            self.text_prompt_slots = text_prompt_slots
            self.vision_prompt_slots = vision_prompt_slots
        else:
            if slot_aggregation and is_first_round:
                print("\nSlot aggregation enabled: first round falls back to rank aggregation to seed global prompts")

            global_component.update(self._aggregate_text_prompts(
                text_prompts_list, client_ids, text_prompt_ranks, topk))
            global_component.update(self._aggregate_vision_prompts(
                vision_prompts_list, client_ids, vision_prompt_ranks, topk))

            self.text_prompt_slots = {
                client_id: list(range(len(ranks)))
                for client_id, ranks in text_prompt_ranks.items()
            }
            self.vision_prompt_slots = {
                client_id: list(range(len(ranks)))
                for client_id, ranks in vision_prompt_ranks.items()
            }

        global_component.update(self._aggregate_tcvp_meta_params(client_ids))

        self._update_global_prompts(global_component, topk)

        self.text_prompt_ranks = text_prompt_ranks
        self.vision_prompt_ranks = vision_prompt_ranks

        return global_component

    def update_client_models(self, client_ids: List[int],
                           global_component: Dict[str, torch.Tensor],
                           all_clients: List[int] = None) -> None:
        """Update client prompts based on ranking."""
        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)

        print("\nStarting client prompt update:")
        print(f"Updating selected prompt indices for {len(client_ids)} participating clients")

        for client_id in client_ids:
            if (client_id not in self.text_prompt_ranks and
                client_id not in self.vision_prompt_ranks):
                continue

            client_weight = self.client_weights['local'][client_id]

            self._update_client_text_prompts(client_id, client_weight, global_component, topk)
            self._update_client_vision_prompts(client_id, client_weight, global_component, topk)
            self._update_client_tcvp_meta_params(client_id, client_weight, global_component)

        # Compute pre-training similarities after parameter update
        if self.current_round > 1:
            self._compute_pre_training_similarities(client_ids, global_component)

    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        """Store trained text and vision prompts."""
        text_prompts = {}
        vision_prompts = {}

        # Extract text prompts
        for name, param in trained_state_dict.items():
            if "prompt_learner.ctx" in name:
                clean_name = name.replace('module.', '')
                text_prompts[clean_name] = copy.deepcopy(param)

        # Extract vision prompts
        vision_prompts = self._extract_visual_prompt_with_fallback(client_id, trained_state_dict)
        tcvp_meta_params = self._extract_tcvp_meta_params(trained_state_dict)

        self.client_weights['components']['text_prompts'][client_id] = text_prompts
        self.client_weights['components']['vision_prompts'][client_id] = vision_prompts
        self.client_weights['components']['tcvp_meta_params'][client_id] = tcvp_meta_params

        # Update local model weights
        for name, param in text_prompts.items():
            self.client_weights['local'][client_id][name] = copy.deepcopy(param)
        for name, param in vision_prompts.items():
            self.client_weights['local'][client_id][name] = copy.deepcopy(param)
        for name, param in tcvp_meta_params.items():
            self.client_weights['local'][client_id][name] = copy.deepcopy(param)

    # === Private helper methods ===

    def _extract_text_prompts(self, client_ids: List[int]) -> List[torch.Tensor]:
        """Extract text prompts from all clients."""
        text_prompts_list = []

        for client_id in client_ids:
            local_weights = self.client_weights['local'][client_id]
            if 'prompt_learner.ctx' in local_weights:
                ctx = local_weights['prompt_learner.ctx']
                text_prompts_list.append(ctx)

        return text_prompts_list

    def _extract_vision_prompts(self, client_ids: List[int]) -> List[torch.Tensor]:
        """Extract vision prompts from all clients."""
        vision_prompts_list = []

        for client_id in client_ids:
            local_weights = self.client_weights['local'][client_id]
            vision_key = 'visual.VPT' if 'visual.VPT' in local_weights else 'image_encoder.VPT'
            if vision_key in local_weights:
                vpt = local_weights[vision_key]
                vision_prompts_list.append(vpt)

        return vision_prompts_list

    def _extract_tcvp_meta_params(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract Text-Conditioned Visual Prompt MetaNet parameters."""
        tcvp_meta_params = {}
        for name, param in state_dict.items():
            clean_name = name.replace('module.', '')
            if "tcvp_meta_net" in clean_name:
                tcvp_meta_params[clean_name] = copy.deepcopy(param)
        return tcvp_meta_params

    def _aggregate_tcvp_meta_params(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate TCVP MetaNet parameters across participating clients."""
        if not getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_ENABLED', False):
            return {}
        if not getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_AGGREGATE_META_NET', True):
            print("TCVP MetaNet aggregation disabled: keeping MetaNet parameters client-local")
            return {}

        meta_keys = []
        for client_id in client_ids:
            for key in self.client_weights['local'][client_id].keys():
                if "tcvp_meta_net" in key:
                    meta_keys.append(key)
            if meta_keys:
                break

        if not meta_keys:
            print("TCVP enabled but no MetaNet parameters were found for aggregation")
            return {}

        aggregation = getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_META_AGGREGATION', 'fedavg').lower()

        if self.client_data_sizes and sum(self.client_data_sizes[client_id] for client_id in client_ids) > 0:
            total_weight = sum(self.client_data_sizes[client_id] for client_id in client_ids)
            client_weights = {
                client_id: self.client_data_sizes[client_id] / total_weight
                for client_id in client_ids
            }
        else:
            client_weights = {client_id: 1.0 / len(client_ids) for client_id in client_ids}

        if aggregation == 'fedadam':
            return self._aggregate_tcvp_meta_params_fedadam(client_ids, meta_keys, client_weights)
        if aggregation != 'fedavg':
            print(f"Unknown TCVP MetaNet aggregation '{aggregation}', falling back to fedavg")

        global_component = {}
        for key in meta_keys:
            averaged_param = None
            contributors = []
            for client_id in client_ids:
                client_param = self.client_weights['local'][client_id].get(key)
                if client_param is None:
                    continue

                weighted_param = client_param * client_weights[client_id]
                averaged_param = weighted_param if averaged_param is None else averaged_param + weighted_param
                contributors.append(client_id)

            if averaged_param is not None:
                global_component[key] = averaged_param
                print(f"TCVP MetaNet param aggregated: {key}, contributors={contributors}")

        return global_component

    def _aggregate_tcvp_meta_params_fedadam(self, client_ids: List[int],
                                           meta_keys: List[str],
                                           client_weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """Server-side FedAdam for TCVP MetaNet.

        Delta is defined as avg(client_param - server_param), so the server
        update moves along the average client update direction.
        """
        lr = getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_META_FEDADAM_LR', 0.001)
        beta1 = getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_META_FEDADAM_BETA1', 0.9)
        beta2 = getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_META_FEDADAM_BETA2', 0.999)
        tau = getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_META_FEDADAM_TAU', 0.001)
        bias_correction = getattr(
            self.cfg.TRAINER.FEDMGP, 'TCVP_META_FEDADAM_BIAS_CORRECTION', True)

        self.tcvp_meta_adam_step += 1
        global_component = {}

        for key in meta_keys:
            server_param = self.tcvp_meta_global_params.get(key)
            if server_param is None and key in self.client_weights.get('global', {}):
                server_param = copy.deepcopy(self.client_weights['global'][key]).detach()
            if server_param is None:
                for client_id in client_ids:
                    client_param = self.client_weights['local'][client_id].get(key)
                    if client_param is not None:
                        server_param = copy.deepcopy(client_param).detach()
                        break

            if server_param is None:
                continue

            avg_delta = None
            contributors = []
            for client_id in client_ids:
                client_param = self.client_weights['local'][client_id].get(key)
                if client_param is None:
                    continue

                delta = (client_param.detach() - server_param).to(server_param.device)
                weighted_delta = delta * client_weights[client_id]
                avg_delta = weighted_delta if avg_delta is None else avg_delta + weighted_delta
                contributors.append(client_id)

            if avg_delta is None:
                continue

            if key not in self.tcvp_meta_adam_m:
                self.tcvp_meta_adam_m[key] = torch.zeros_like(server_param)
            if key not in self.tcvp_meta_adam_v:
                self.tcvp_meta_adam_v[key] = torch.zeros_like(server_param)

            m = self.tcvp_meta_adam_m[key]
            v = self.tcvp_meta_adam_v[key]
            m.mul_(beta1).add_(avg_delta, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(avg_delta, avg_delta, value=1.0 - beta2)

            if bias_correction:
                m_hat = m / (1.0 - beta1 ** self.tcvp_meta_adam_step)
                v_hat = v / (1.0 - beta2 ** self.tcvp_meta_adam_step)
            else:
                m_hat = m
                v_hat = v

            updated_param = server_param + lr * m_hat / (v_hat.sqrt() + tau)
            updated_param = updated_param.to(dtype=server_param.dtype)

            self.tcvp_meta_global_params[key] = copy.deepcopy(updated_param).detach()
            global_component[key] = updated_param

            delta_norm = avg_delta.float().norm().item()
            update_norm = (updated_param - server_param).float().norm().item()
            print(
                f"TCVP MetaNet param FedAdam: {key}, contributors={contributors}, "
                f"delta_norm={delta_norm:.6f}, update_norm={update_norm:.6f}"
            )

        return global_component

    def _first_round_ranking(self, text_prompts_list: List[torch.Tensor],
                           vision_prompts_list: List[torch.Tensor],
                           client_ids: List[int], topk: int) -> tuple:
        """Random ranking for first round."""
        text_prompt_ranks = {}
        vision_prompt_ranks = {}

        for client_idx, client_id in enumerate(client_ids):
            # Random text prompt selection
            if client_idx < len(text_prompts_list):
                client_prompt = text_prompts_list[client_idx]
                num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1

                seed = 42 + client_id
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)

                if num_prompts <= topk:
                    selected_indices = list(range(num_prompts))
                else:
                    selected_indices = torch.randperm(num_prompts)[:topk].tolist()

                torch.random.set_rng_state(orig_state)
                text_prompt_ranks[client_id] = selected_indices
                print(f"Client {client_id}: First round random text prompt selection: {selected_indices}")

            # Random vision prompt selection
            if client_idx < len(vision_prompts_list):
                client_prompt = vision_prompts_list[client_idx]
                num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1

                seed = 42 + client_id
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)

                if num_prompts <= topk:
                    selected_indices = list(range(num_prompts))
                else:
                    selected_indices = torch.randperm(num_prompts)[:topk].tolist()

                torch.random.set_rng_state(orig_state)
                vision_prompt_ranks[client_id] = selected_indices
                print(f"Client {client_id}: First round random vision prompt selection: {selected_indices}")

        return text_prompt_ranks, vision_prompt_ranks

    def _first_round_group_bind_ranking(self, text_prompts_list: List[torch.Tensor],
                                      vision_prompts_list: List[torch.Tensor],
                                      client_ids: List[int], topk: int) -> tuple:
        """Randomly select the same group indices for text and vision prompts."""
        text_prompt_ranks = {}
        vision_prompt_ranks = {}

        print(f"\nGroup-bind selection enabled: first round uses shared random prompt groups, top-{topk}")

        for client_idx, client_id in enumerate(client_ids):
            if client_idx >= len(text_prompts_list) or client_idx >= len(vision_prompts_list):
                continue

            text_prompt = text_prompts_list[client_idx]
            vision_prompt = vision_prompts_list[client_idx]
            text_num_prompts = text_prompt.size(0) if text_prompt.dim() == 3 else 1
            vision_num_prompts = vision_prompt.size(0) if vision_prompt.dim() == 3 else 1
            num_groups = min(text_num_prompts, vision_num_prompts)

            seed = 42 + client_id
            orig_state = torch.random.get_rng_state()
            torch.manual_seed(seed)

            if num_groups <= topk:
                selected_indices = list(range(num_groups))
            else:
                selected_indices = torch.randperm(num_groups)[:topk].tolist()

            torch.random.set_rng_state(orig_state)

            text_prompt_ranks[client_id] = selected_indices
            vision_prompt_ranks[client_id] = selected_indices
            print(f"Client {client_id}: First round random bound prompt group selection: {selected_indices}")

        return text_prompt_ranks, vision_prompt_ranks

    def _rank_prompt_groups_by_similarity(self, text_prompts_list: List[torch.Tensor],
                                        vision_prompts_list: List[torch.Tensor],
                                        client_ids: List[int]) -> tuple:
        """Rank bound text-vision prompt groups by combined similarity to global prompt groups."""
        if not text_prompts_list or not vision_prompts_list:
            print("Warning: Missing text or vision prompts for group-bind selection; falling back to independent ranking")
            text_prompt_ranks = self._rank_prompts_by_similarity(text_prompts_list, client_ids, is_vision=False)
            vision_prompt_ranks = self._rank_prompts_by_similarity(vision_prompts_list, client_ids, is_vision=True)
            return text_prompt_ranks, vision_prompt_ranks

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        aggregate_highest = getattr(self.cfg.TRAINER.FEDMGP, 'AGGREGATE_HIGHEST_SIM', False)
        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')
        probabilistic_select = getattr(self.cfg.TRAINER.FEDMGP, 'PROBABILISTIC_SELECTION', False)
        temperature = getattr(self.cfg.TRAINER.FEDMGP, 'TEMPERATURE', 1.0)

        if select_mode == "random":
            selection_mode = "random"
        elif select_mode == "top_fixed":
            selection_mode = f"fixed top-{topk}"
        elif select_mode == "all":
            selection_mode = "all"
        elif probabilistic_select:
            selection_mode = f"probabilistic group-bind (temp={temperature})"
        else:
            sort_direction = "highest" if aggregate_highest else "lowest"
            selection_mode = f"{sort_direction} group-bind similarity"

        print(f"\nRanking bound text-vision prompt groups for {len(client_ids)} clients, {selection_mode} top-{topk}...")

        text_prompt_ranks = {}
        vision_prompt_ranks = {}

        for client_idx, client_id in enumerate(client_ids):
            if client_idx >= len(text_prompts_list) or client_idx >= len(vision_prompts_list):
                default_indices = list(range(min(topk, 2)))
                text_prompt_ranks[client_id] = default_indices
                vision_prompt_ranks[client_id] = default_indices
                continue

            text_prompt = text_prompts_list[client_idx]
            vision_prompt = vision_prompts_list[client_idx]

            if text_prompt.dim() < 2 or vision_prompt.dim() < 2:
                default_indices = list(range(min(topk, 2)))
                text_prompt_ranks[client_id] = default_indices
                vision_prompt_ranks[client_id] = default_indices
                continue

            text_num_prompts = text_prompt.size(0) if text_prompt.dim() == 3 else 1
            vision_num_prompts = vision_prompt.size(0) if vision_prompt.dim() == 3 else 1
            num_groups = min(text_num_prompts, vision_num_prompts)

            if num_groups <= 1:
                selected_indices = [0]
                text_prompt_ranks[client_id] = selected_indices
                vision_prompt_ranks[client_id] = selected_indices
                continue

            if select_mode == "random":
                seed = 42 + client_id + self.current_round
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)
                selected_indices = torch.randperm(num_groups)[:min(topk, num_groups)].tolist()
                torch.random.set_rng_state(orig_state)
                print(f"Client {client_id}: Random bound prompt group selection: {selected_indices}")

            elif select_mode == "top_fixed":
                selected_indices = list(range(min(topk, num_groups)))
                print(f"Client {client_id}: Fixed top-{len(selected_indices)} bound prompt groups: {selected_indices}")

            elif select_mode == "all":
                selected_indices = list(range(num_groups))
                print(f"Client {client_id}: Selected all {num_groups} bound prompt groups: {selected_indices}")

            else:
                similarities = self._compute_bound_group_similarities(text_prompt, vision_prompt)

                if probabilistic_select:
                    selected_indices = self._probabilistic_prompt_selection(
                        similarities, topk, client_id, aggregate_highest, temperature)
                    similarities_str = ", ".join([f"{idx}:{sim:.4f}" for idx, sim in similarities])
                    print(f"Client {client_id}: bound group similarities [{similarities_str}], probabilistic selection: {selected_indices}")
                else:
                    if aggregate_highest:
                        similarities.sort(key=lambda x: x[1], reverse=True)
                    else:
                        similarities.sort(key=lambda x: x[1])
                    selected_indices = [idx for idx, _ in similarities[:min(topk, num_groups)]]
                    similarities_str = ", ".join([f"{idx}:{sim:.4f}" for idx, sim in similarities])
                    print(f"Client {client_id}: bound group similarities [{similarities_str}], selected: {selected_indices}")

            text_prompt_ranks[client_id] = selected_indices
            vision_prompt_ranks[client_id] = selected_indices

        print(f"Completed bound prompt group ranking for all clients, {selection_mode} top-{topk}")
        return text_prompt_ranks, vision_prompt_ranks

    def _rank_prompts_by_similarity(self, prompts_list: List[torch.Tensor],
                                   client_ids: List[int], is_vision: bool = False) -> Dict[int, List[int]]:
        """Rank prompts by similarity to global prompts."""
        if not prompts_list or len(prompts_list) < 1:
            topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
            default_indices = list(range(min(topk, 2)))
            default_ranks = {client_id: default_indices for client_id in client_ids if len(client_ids) > 0}
            prompt_type = "vision" if is_vision else "text"
            print(f"Warning: Insufficient {prompt_type} prompts, using default ranking {default_indices}")
            return default_ranks

        prompt_type = "vision" if is_vision else "text"
        client_prompt_ranks = {}

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        aggregate_highest = getattr(self.cfg.TRAINER.FEDMGP, 'AGGREGATE_HIGHEST_SIM', False)
        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')
        probabilistic_select = getattr(self.cfg.TRAINER.FEDMGP, 'PROBABILISTIC_SELECTION', False)
        temperature = getattr(self.cfg.TRAINER.FEDMGP, 'TEMPERATURE', 1.0)

        global_prompts = self.global_vision_prompts if is_vision else self.global_text_prompts

        # Determine selection mode description
        if select_mode == "random":
            selection_mode = "random"
        elif select_mode == "top_fixed":
            selection_mode = f"fixed top-{topk}"
        elif select_mode == "all":
            selection_mode = "all"
        elif probabilistic_select:
            selection_mode = f"probabilistic (temp={temperature})"
        else:
            sort_direction = "highest" if aggregate_highest else "lowest"
            selection_mode = f"{sort_direction} similarity"

        print(f"\nRanking {prompt_type} prompts for {len(client_ids)} clients, {selection_mode} top-{topk}...")

        for client_idx, client_id in enumerate(client_ids):
            if client_idx >= len(prompts_list):
                default_indices = list(range(min(topk, 2)))
                client_prompt_ranks[client_id] = default_indices
                continue

            client_prompt = prompts_list[client_idx]
            if client_prompt.dim() < 2:
                default_indices = list(range(min(topk, 2)))
                client_prompt_ranks[client_id] = default_indices
                continue

            num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1
            if num_prompts <= 1:
                client_prompt_ranks[client_id] = [0]
                continue

            if select_mode == "random":
                seed = 42 + client_id + self.current_round
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)

                selected_indices = torch.randperm(num_prompts)[:min(topk, num_prompts)].tolist()
                print(f"Client {client_id}: Random {prompt_type} prompt selection: {selected_indices}")

                torch.random.set_rng_state(orig_state)
                client_prompt_ranks[client_id] = selected_indices

            elif select_mode == "top_fixed":
                selected_indices = list(range(min(topk, num_prompts)))
                print(f"Client {client_id}: Fixed top-{len(selected_indices)} {prompt_type} prompts: {selected_indices}")
                client_prompt_ranks[client_id] = selected_indices

            elif select_mode == "all":
                selected_indices = list(range(num_prompts))
                print(f"Client {client_id}: Selected all {num_prompts} {prompt_type} prompts: {selected_indices}")
                client_prompt_ranks[client_id] = selected_indices

            else:  # similarity mode
                similarities = self._compute_prompt_similarities(client_prompt, global_prompts)

                if probabilistic_select:
                    selected_indices = self._probabilistic_prompt_selection(
                        similarities, topk, client_id, aggregate_highest, temperature)

                    similarities_str = ", ".join([f"{idx}:{sim:.4f}" for idx, sim in similarities])
                    print(f"Client {client_id}: {prompt_type} similarities [{similarities_str}], probabilistic selection: {selected_indices}")

                    client_prompt_ranks[client_id] = selected_indices
                else:
                    if aggregate_highest:
                        similarities.sort(key=lambda x: x[1], reverse=True)
                    else:
                        similarities.sort(key=lambda x: x[1])

                    selected_indices = [idx for idx, _ in similarities[:min(topk, num_prompts)]]

                    similarities_str = ", ".join([f"{idx}:{sim:.4f}" for idx, sim in similarities])
                    print(f"Client {client_id}: {prompt_type} similarities [{similarities_str}], selected: {selected_indices}")

                    client_prompt_ranks[client_id] = selected_indices

        print(f"Completed {prompt_type} prompt ranking for all clients, {selection_mode} top-{topk}")
        return client_prompt_ranks

    def _compute_prompt_similarities(self, client_prompt: torch.Tensor,
                                   global_prompts: List[torch.Tensor]) -> List[tuple]:
        """Compute cosine similarity between client and global prompts."""
        similarities = []
        num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1

        for i in range(num_prompts):
            if client_prompt.dim() == 3:
                prompt_i = client_prompt[i].reshape(-1)

                prompt_sims = []
                for global_prompt in global_prompts:
                    global_vec = global_prompt.reshape(-1)

                    norm_i = torch.norm(prompt_i)
                    norm_g = torch.norm(global_vec)

                    if norm_i > 0 and norm_g > 0:
                        sim = torch.dot(prompt_i, global_vec) / (norm_i * norm_g)
                        prompt_sims.append(sim.item())
                    else:
                        prompt_sims.append(0.0)

                if prompt_sims:
                    similarities.append((i, max(prompt_sims)))
                else:
                    similarities.append((i, 0.0))
            else:
                similarities.append((i, 0.0))

        return similarities

    def _compute_bound_group_similarities(self, client_text_prompt: torch.Tensor,
                                        client_vision_prompt: torch.Tensor) -> List[tuple]:
        """Compute combined similarities for bound text-vision prompt groups.

        For each local group i, this computes its best similarity to aligned global
        text/vision prompt slots k:
            score(i) = max_k [w_t sim(text_i, global_text_k)
                              + w_v sim(vision_i, global_vision_k)]
        The best slot is used only for scoring in this first experiment; aggregation
        remains rank-based to keep all other FedMGP behavior unchanged.
        """
        similarities = []
        text_weight = getattr(self.cfg.TRAINER.FEDMGP, 'GROUP_BIND_TEXT_WEIGHT', 0.5)
        vision_weight = getattr(self.cfg.TRAINER.FEDMGP, 'GROUP_BIND_VISION_WEIGHT', 0.5)
        weight_sum = text_weight + vision_weight
        if weight_sum <= 0:
            text_weight = 0.5
            vision_weight = 0.5
            weight_sum = 1.0
        text_weight = text_weight / weight_sum
        vision_weight = vision_weight / weight_sum

        text_num_prompts = client_text_prompt.size(0) if client_text_prompt.dim() == 3 else 1
        vision_num_prompts = client_vision_prompt.size(0) if client_vision_prompt.dim() == 3 else 1
        num_groups = min(text_num_prompts, vision_num_prompts)

        global_text_prompts = self.global_text_prompts or []
        global_vision_prompts = self.global_vision_prompts or []
        num_global_groups = min(len(global_text_prompts), len(global_vision_prompts))

        for i in range(num_groups):
            text_i = client_text_prompt[i] if client_text_prompt.dim() == 3 else client_text_prompt
            vision_i = client_vision_prompt[i] if client_vision_prompt.dim() == 3 else client_vision_prompt

            group_scores = []
            for k in range(num_global_groups):
                text_sim = self._cosine_flatten(text_i, global_text_prompts[k])
                vision_sim = self._cosine_flatten(vision_i, global_vision_prompts[k])
                group_scores.append(text_weight * text_sim + vision_weight * vision_sim)

            if group_scores:
                similarities.append((i, max(group_scores)))
            else:
                # Fallback for malformed state: use whichever modality has global prompts.
                fallback_scores = []
                for global_text_prompt in global_text_prompts:
                    fallback_scores.append(self._cosine_flatten(text_i, global_text_prompt))
                for global_vision_prompt in global_vision_prompts:
                    fallback_scores.append(self._cosine_flatten(vision_i, global_vision_prompt))
                similarities.append((i, max(fallback_scores) if fallback_scores else 0.0))

        return similarities

    @staticmethod
    def _cosine_flatten(left: torch.Tensor, right: torch.Tensor) -> float:
        """Compute cosine similarity between flattened prompt tensors."""
        left_vec = left.reshape(-1)
        right_vec = right.reshape(-1)
        norm_left = torch.norm(left_vec)
        norm_right = torch.norm(right_vec)
        if norm_left > 0 and norm_right > 0:
            return (torch.dot(left_vec, right_vec) / (norm_left * norm_right)).item()
        return 0.0

    def _probabilistic_prompt_selection(self, similarities: List[tuple],
                                      topk: int, client_id: int,
                                      aggregate_highest: bool, temperature: float) -> List[int]:
        """Select prompts based on probability distribution."""
        sim_values = torch.tensor([sim for _, sim in similarities])

        if not aggregate_highest:
            sim_values = -sim_values

        probs = torch.nn.functional.softmax(sim_values / temperature, dim=0)

        seed = 42 + client_id + self.current_round
        orig_state = torch.random.get_rng_state()
        torch.manual_seed(seed)

        # Sample topk indices without replacement
        selected_indices = []
        remaining_indices = list(range(len(similarities)))

        for _ in range(min(topk, len(similarities))):
            if not remaining_indices:
                break

            curr_probs = probs[remaining_indices]
            curr_probs = curr_probs / curr_probs.sum()

            idx = torch.multinomial(curr_probs, 1).item()
            selected_idx = remaining_indices.pop(idx)
            selected_indices.append(selected_idx)

        torch.random.set_rng_state(orig_state)

        return selected_indices

    def _assign_prompt_slots(self, prompts_list: List[torch.Tensor],
                           client_ids: List[int], prompt_ranks: Dict[int, List[int]],
                           global_prompts: List[torch.Tensor],
                           prompt_type: str) -> Dict[int, List[Optional[int]]]:
        """Assign selected local prompts to their most similar global prompt slots."""
        num_slots = len(global_prompts)
        prompt_slots = {
            client_id: [None] * len(prompt_ranks.get(client_id, []))
            for client_id in client_ids
            if client_id in prompt_ranks
        }

        if num_slots == 0:
            return prompt_slots

        candidates = []
        assign_highest = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_ASSIGN_HIGHEST_SIM', True)

        for client_idx, client_id in enumerate(client_ids):
            if client_id not in prompt_ranks or client_idx >= len(prompts_list):
                continue

            client_prompt = prompts_list[client_idx]
            for rank_pos, prompt_idx in enumerate(prompt_ranks[client_id]):
                selected_prompt = self._get_prompt_by_index(client_prompt, prompt_idx)
                if selected_prompt is None:
                    continue

                slot_sims = [self._cosine_flatten(selected_prompt, global_prompt)
                             for global_prompt in global_prompts]
                if assign_highest:
                    slot = max(range(num_slots), key=lambda idx: slot_sims[idx])
                else:
                    slot = min(range(num_slots), key=lambda idx: slot_sims[idx])

                candidates.append({
                    'client_id': client_id,
                    'rank_pos': rank_pos,
                    'prompt_idx': prompt_idx,
                    'slot_sims': slot_sims,
                    'slot': slot,
                })

        self._rebalance_slot_assignments(candidates, num_slots, prompt_type)

        for candidate in candidates:
            client_id = candidate['client_id']
            rank_pos = candidate['rank_pos']
            if client_id in prompt_slots and rank_pos < len(prompt_slots[client_id]):
                prompt_slots[client_id][rank_pos] = candidate['slot']

        self._print_slot_assignments(prompt_slots, prompt_ranks, prompt_type)
        return prompt_slots

    def _assign_bound_prompt_slots(self, text_prompts_list: List[torch.Tensor],
                                 vision_prompts_list: List[torch.Tensor],
                                 client_ids: List[int],
                                 text_prompt_ranks: Dict[int, List[int]],
                                 vision_prompt_ranks: Dict[int, List[int]]) -> tuple:
        """Assign selected bound text-vision prompt groups to shared global slots."""
        num_slots = min(len(self.global_text_prompts or []), len(self.global_vision_prompts or []))
        if num_slots == 0:
            return {}, {}

        text_prompt_slots = {
            client_id: [None] * len(text_prompt_ranks.get(client_id, []))
            for client_id in client_ids
            if client_id in text_prompt_ranks
        }
        vision_prompt_slots = {
            client_id: [None] * len(vision_prompt_ranks.get(client_id, []))
            for client_id in client_ids
            if client_id in vision_prompt_ranks
        }

        text_weight = getattr(self.cfg.TRAINER.FEDMGP, 'GROUP_BIND_TEXT_WEIGHT', 0.5)
        vision_weight = getattr(self.cfg.TRAINER.FEDMGP, 'GROUP_BIND_VISION_WEIGHT', 0.5)
        weight_sum = text_weight + vision_weight
        if weight_sum <= 0:
            text_weight = 0.5
            vision_weight = 0.5
            weight_sum = 1.0
        text_weight = text_weight / weight_sum
        vision_weight = vision_weight / weight_sum

        candidates = []
        assign_highest = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_ASSIGN_HIGHEST_SIM', True)

        for client_idx, client_id in enumerate(client_ids):
            if (client_id not in text_prompt_ranks or client_id not in vision_prompt_ranks or
                client_idx >= len(text_prompts_list) or client_idx >= len(vision_prompts_list)):
                continue

            text_prompt = text_prompts_list[client_idx]
            vision_prompt = vision_prompts_list[client_idx]
            num_positions = min(len(text_prompt_ranks[client_id]), len(vision_prompt_ranks[client_id]))

            for rank_pos in range(num_positions):
                text_idx = text_prompt_ranks[client_id][rank_pos]
                vision_idx = vision_prompt_ranks[client_id][rank_pos]
                selected_text = self._get_prompt_by_index(text_prompt, text_idx)
                selected_vision = self._get_prompt_by_index(vision_prompt, vision_idx)
                if selected_text is None or selected_vision is None:
                    continue

                slot_sims = []
                for slot in range(num_slots):
                    text_sim = self._cosine_flatten(selected_text, self.global_text_prompts[slot])
                    vision_sim = self._cosine_flatten(selected_vision, self.global_vision_prompts[slot])
                    slot_sims.append(text_weight * text_sim + vision_weight * vision_sim)

                if assign_highest:
                    slot = max(range(num_slots), key=lambda idx: slot_sims[idx])
                else:
                    slot = min(range(num_slots), key=lambda idx: slot_sims[idx])

                candidates.append({
                    'client_id': client_id,
                    'rank_pos': rank_pos,
                    'prompt_idx': text_idx,
                    'vision_prompt_idx': vision_idx,
                    'slot_sims': slot_sims,
                    'slot': slot,
                })

        self._rebalance_slot_assignments(candidates, num_slots, "bound group")

        for candidate in candidates:
            client_id = candidate['client_id']
            rank_pos = candidate['rank_pos']
            if client_id in text_prompt_slots and rank_pos < len(text_prompt_slots[client_id]):
                text_prompt_slots[client_id][rank_pos] = candidate['slot']
            if client_id in vision_prompt_slots and rank_pos < len(vision_prompt_slots[client_id]):
                vision_prompt_slots[client_id][rank_pos] = candidate['slot']

        self._print_slot_assignments(text_prompt_slots, text_prompt_ranks, "bound text")
        self._print_slot_assignments(vision_prompt_slots, vision_prompt_ranks, "bound vision")
        return text_prompt_slots, vision_prompt_slots

    def _rebalance_slot_assignments(self, candidates: List[Dict[str, Any]],
                                  num_slots: int, prompt_type: str) -> None:
        """Reassign prompts so each global slot receives a minimum number of updates."""
        if not candidates or num_slots <= 0:
            return

        min_prompts = max(0, int(getattr(
            self.cfg.TRAINER.FEDMGP, 'SLOT_MIN_PROMPTS_PER_SLOT', 1)))
        if min_prompts == 0:
            return

        effective_min = min(min_prompts, len(candidates) // num_slots)
        if effective_min < min_prompts:
            print(
                f"Slot aggregation warning: only {len(candidates)} {prompt_type} candidates "
                f"for {num_slots} slots, effective minimum is {effective_min}"
            )
        if effective_min == 0:
            return

        def count_slots() -> List[int]:
            counts = [0] * num_slots
            for item in candidates:
                if item['slot'] is not None:
                    counts[item['slot']] += 1
            return counts

        counts = count_slots()
        changed = True
        while changed:
            changed = False
            underfilled_slots = [slot for slot, count in enumerate(counts) if count < effective_min]
            if not underfilled_slots:
                break

            for target_slot in underfilled_slots:
                donors = [
                    item for item in candidates
                    if item['slot'] != target_slot and counts[item['slot']] > effective_min
                ]
                if not donors:
                    continue

                donor = max(donors, key=lambda item: item['slot_sims'][target_slot])
                counts[donor['slot']] -= 1
                donor['slot'] = target_slot
                counts[target_slot] += 1
                changed = True

        print(f"Slot aggregation {prompt_type} counts after quota rebalance: {counts}")

    def _print_slot_assignments(self, prompt_slots: Dict[int, List[Optional[int]]],
                              prompt_ranks: Dict[int, List[int]],
                              prompt_type: str) -> None:
        """Print selected prompt to global slot mappings."""
        print(f"\nSlot aggregation {prompt_type} assignments:")
        for client_id, slots in prompt_slots.items():
            pairs = []
            ranks = prompt_ranks.get(client_id, [])
            for rank_pos, slot in enumerate(slots):
                if slot is None or rank_pos >= len(ranks):
                    continue
                pairs.append(f"local {ranks[rank_pos]} -> global {slot}")
            if pairs:
                print(f"Client {client_id}: {', '.join(pairs)}")

    @staticmethod
    def _get_prompt_by_index(prompt_tensor: torch.Tensor, prompt_idx: int) -> Optional[torch.Tensor]:
        """Return a single prompt group tensor from a prompt bank."""
        if prompt_tensor.dim() == 3 and prompt_idx < prompt_tensor.size(0):
            return prompt_tensor[prompt_idx]
        if prompt_tensor.dim() == 2 and prompt_idx == 0:
            return prompt_tensor
        return None

    def _aggregate_text_prompts(self, text_prompts_list: List[torch.Tensor],
                              client_ids: List[int], text_prompt_ranks: Dict[int, List[int]],
                              topk: int) -> Dict[str, torch.Tensor]:
        """Aggregate text prompts by rank."""
        global_component = {}
        text_contributions_by_rank = {}

        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_prompts = 0
            for client_idx, client_id in enumerate(client_ids):
                if client_id in text_prompt_ranks and client_idx < len(text_prompts_list):
                    max_prompts = max(max_prompts, len(text_prompt_ranks[client_id]))

            for rank in range(max_prompts):
                rank_text_values = []
                rank_text_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in text_prompt_ranks and client_idx < len(text_prompts_list):
                        if rank < len(text_prompt_ranks[client_id]):
                            prompt_idx = text_prompt_ranks[client_id][rank]
                            client_prompt = text_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_text_values.append(selected_prompt)
                                rank_text_contributions.append((client_id, prompt_idx))

                if rank_text_values:
                    try:
                        all_prompts = torch.cat(rank_text_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'prompt_learner.ctx_{rank}'] = avg_prompt
                        text_contributions_by_rank[rank] = rank_text_contributions

                        contributors = [cid for cid, _ in rank_text_contributions]
                        prompt_indices = [idx for _, idx in rank_text_contributions]
                        print(f"Text prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Text prompt rank {rank+1} aggregation failed: {str(e)}")
        else:
            for rank in range(topk):
                rank_text_values = []
                rank_text_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in text_prompt_ranks and client_idx < len(text_prompts_list):
                        if rank < len(text_prompt_ranks[client_id]):
                            prompt_idx = text_prompt_ranks[client_id][rank]
                            client_prompt = text_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_text_values.append(selected_prompt)
                                rank_text_contributions.append((client_id, prompt_idx))

                if rank_text_values:
                    try:
                        all_prompts = torch.cat(rank_text_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'prompt_learner.ctx_{rank}'] = avg_prompt
                        text_contributions_by_rank[rank] = rank_text_contributions

                        contributors = [cid for cid, _ in rank_text_contributions]
                        prompt_indices = [idx for _, idx in rank_text_contributions]
                        print(f"Text prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Text prompt rank {rank+1} aggregation failed: {str(e)}")

        self.text_contributions_by_rank = text_contributions_by_rank
        return global_component

    def _aggregate_vision_prompts(self, vision_prompts_list: List[torch.Tensor],
                                client_ids: List[int], vision_prompt_ranks: Dict[int, List[int]],
                                topk: int) -> Dict[str, torch.Tensor]:
        """Aggregate vision prompts by rank."""
        global_component = {}
        vision_contributions_by_rank = {}

        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_prompts = 0
            for client_idx, client_id in enumerate(client_ids):
                if client_id in vision_prompt_ranks and client_idx < len(vision_prompts_list):
                    max_prompts = max(max_prompts, len(vision_prompt_ranks[client_id]))

            for rank in range(max_prompts):
                rank_vision_values = []
                rank_vision_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in vision_prompt_ranks and client_idx < len(vision_prompts_list):
                        if rank < len(vision_prompt_ranks[client_id]):
                            prompt_idx = vision_prompt_ranks[client_id][rank]
                            client_prompt = vision_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, prompt_idx))
                            elif client_prompt.dim() == 2 and rank == 0:
                                selected_prompt = client_prompt.unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, 0))

                if rank_vision_values:
                    try:
                        all_prompts = torch.cat(rank_vision_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'visual.VPT_{rank}'] = avg_prompt
                        vision_contributions_by_rank[rank] = rank_vision_contributions

                        contributors = [cid for cid, _ in rank_vision_contributions]
                        prompt_indices = [idx for _, idx in rank_vision_contributions]
                        print(f"Vision prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Vision prompt rank {rank+1} aggregation failed: {str(e)}")
        else:
            for rank in range(topk):
                rank_vision_values = []
                rank_vision_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in vision_prompt_ranks and client_idx < len(vision_prompts_list):
                        if rank < len(vision_prompt_ranks[client_id]):
                            prompt_idx = vision_prompt_ranks[client_id][rank]
                            client_prompt = vision_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, prompt_idx))
                            elif client_prompt.dim() == 2 and rank == 0:
                                selected_prompt = client_prompt.unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, 0))

                if rank_vision_values:
                    try:
                        all_prompts = torch.cat(rank_vision_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'visual.VPT_{rank}'] = avg_prompt
                        vision_contributions_by_rank[rank] = rank_vision_contributions

                        contributors = [cid for cid, _ in rank_vision_contributions]
                        prompt_indices = [idx for _, idx in rank_vision_contributions]
                        print(f"Vision prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Vision prompt rank {rank+1} aggregation failed: {str(e)}")

        self.vision_contributions_by_rank = vision_contributions_by_rank
        return global_component

    def _aggregate_text_prompts_by_slot(self, text_prompts_list: List[torch.Tensor],
                                      client_ids: List[int],
                                      text_prompt_ranks: Dict[int, List[int]],
                                      text_prompt_slots: Dict[int, List[Optional[int]]]) -> Dict[str, torch.Tensor]:
        """Aggregate selected text prompts into their assigned global prompt slots."""
        global_component = {}
        text_contributions_by_rank = {}
        num_slots = len(self.global_text_prompts or [])
        keep_empty = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_KEEP_EMPTY_PROMPTS', True)

        print(f"\nAggregating text prompts by similarity slots ({num_slots} global slots)")

        for slot in range(num_slots):
            slot_text_values = []
            slot_text_contributions = []

            for client_idx, client_id in enumerate(client_ids):
                if (client_id not in text_prompt_ranks or
                    client_id not in text_prompt_slots or
                    client_idx >= len(text_prompts_list)):
                    continue

                client_prompt = text_prompts_list[client_idx]
                for rank_pos, prompt_idx in enumerate(text_prompt_ranks[client_id]):
                    slots = text_prompt_slots[client_id]
                    if rank_pos >= len(slots) or slots[rank_pos] != slot:
                        continue

                    selected_prompt = self._get_prompt_by_index(client_prompt, prompt_idx)
                    if selected_prompt is not None:
                        slot_text_values.append(selected_prompt.unsqueeze(0))
                        slot_text_contributions.append((client_id, prompt_idx))

            if slot_text_values:
                try:
                    all_prompts = torch.cat(slot_text_values, dim=0)
                    avg_prompt = all_prompts.mean(dim=0, keepdim=True)
                    global_component[f'prompt_learner.ctx_{slot}'] = avg_prompt
                    text_contributions_by_rank[slot] = slot_text_contributions

                    contributors = [cid for cid, _ in slot_text_contributions]
                    prompt_indices = [idx for _, idx in slot_text_contributions]
                    print(f"Text prompt slot {slot} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                except Exception as e:
                    print(f"Text prompt slot {slot} aggregation failed: {str(e)}")
            elif keep_empty and slot < len(self.global_text_prompts):
                global_component[f'prompt_learner.ctx_{slot}'] = self.global_text_prompts[slot].unsqueeze(0)
                text_contributions_by_rank[slot] = []
                print(f"Text prompt slot {slot}: no contributors, keeping previous global prompt")

        self.text_contributions_by_rank = text_contributions_by_rank
        return global_component

    def _aggregate_vision_prompts_by_slot(self, vision_prompts_list: List[torch.Tensor],
                                        client_ids: List[int],
                                        vision_prompt_ranks: Dict[int, List[int]],
                                        vision_prompt_slots: Dict[int, List[Optional[int]]]) -> Dict[str, torch.Tensor]:
        """Aggregate selected vision prompts into their assigned global prompt slots."""
        global_component = {}
        vision_contributions_by_rank = {}
        num_slots = len(self.global_vision_prompts or [])
        keep_empty = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_KEEP_EMPTY_PROMPTS', True)

        print(f"\nAggregating vision prompts by similarity slots ({num_slots} global slots)")

        for slot in range(num_slots):
            slot_vision_values = []
            slot_vision_contributions = []

            for client_idx, client_id in enumerate(client_ids):
                if (client_id not in vision_prompt_ranks or
                    client_id not in vision_prompt_slots or
                    client_idx >= len(vision_prompts_list)):
                    continue

                client_prompt = vision_prompts_list[client_idx]
                for rank_pos, prompt_idx in enumerate(vision_prompt_ranks[client_id]):
                    slots = vision_prompt_slots[client_id]
                    if rank_pos >= len(slots) or slots[rank_pos] != slot:
                        continue

                    selected_prompt = self._get_prompt_by_index(client_prompt, prompt_idx)
                    if selected_prompt is not None:
                        slot_vision_values.append(selected_prompt.unsqueeze(0))
                        slot_vision_contributions.append((client_id, prompt_idx))

            if slot_vision_values:
                try:
                    all_prompts = torch.cat(slot_vision_values, dim=0)
                    avg_prompt = all_prompts.mean(dim=0, keepdim=True)
                    global_component[f'visual.VPT_{slot}'] = avg_prompt
                    vision_contributions_by_rank[slot] = slot_vision_contributions

                    contributors = [cid for cid, _ in slot_vision_contributions]
                    prompt_indices = [idx for _, idx in slot_vision_contributions]
                    print(f"Vision prompt slot {slot} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                except Exception as e:
                    print(f"Vision prompt slot {slot} aggregation failed: {str(e)}")
            elif keep_empty and slot < len(self.global_vision_prompts):
                global_component[f'visual.VPT_{slot}'] = self.global_vision_prompts[slot].unsqueeze(0)
                vision_contributions_by_rank[slot] = []
                print(f"Vision prompt slot {slot}: no contributors, keeping previous global prompt")

        self.vision_contributions_by_rank = vision_contributions_by_rank
        return global_component

    def _update_global_prompts(self, global_component: Dict[str, torch.Tensor], topk: int) -> None:
        """Update global prompts for next round."""
        global_text_prompts = []
        global_vision_prompts = []

        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_text_rank = -1
            max_vision_rank = -1

            for key in global_component.keys():
                if key.startswith('prompt_learner.ctx_'):
                    rank = int(key.split('_')[-1])
                    max_text_rank = max(max_text_rank, rank)
                elif key.startswith('visual.VPT_'):
                    rank = int(key.split('_')[-1])
                    max_vision_rank = max(max_vision_rank, rank)

            for rank in range(max_text_rank + 1):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    global_text_prompts.append(global_component[text_key][0])

            for rank in range(max_vision_rank + 1):
                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    global_vision_prompts.append(global_component[vision_key][0])
        else:
            for rank in range(topk):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    global_text_prompts.append(global_component[text_key][0])

                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    global_vision_prompts.append(global_component[vision_key][0])

        self.global_text_prompts = global_text_prompts
        self.global_vision_prompts = global_vision_prompts

    def _update_client_text_prompts(self, client_id: int, client_weight: Dict,
                                  global_component: Dict[str, torch.Tensor], topk: int) -> None:
        """Update client text prompts."""
        text_prompt_indices = []
        updated_slots = []
        slot_aggregation = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_AGGREGATION', False)

        if client_id in self.text_prompt_ranks and 'prompt_learner.ctx' in client_weight:
            ctx = client_weight['prompt_learner.ctx']
            for rank, idx in enumerate(self.text_prompt_ranks[client_id]):
                slot = rank
                if slot_aggregation and client_id in self.text_prompt_slots:
                    slots = self.text_prompt_slots[client_id]
                    if rank < len(slots) and slots[rank] is not None:
                        slot = slots[rank]

                key = f'prompt_learner.ctx_{slot}'
                if key in global_component and idx < ctx.size(0):
                    ctx[idx] = global_component[key][0]
                    text_prompt_indices.append(idx)
                    updated_slots.append(slot)

        if text_prompt_indices:
            print(f"Client {client_id}:")
            if slot_aggregation:
                print(f"  - Text prompt update: indices {text_prompt_indices} (global slots {updated_slots})")
            else:
                print(f"  - Text prompt update: indices {text_prompt_indices} (rank order {list(range(len(text_prompt_indices)))})")
            print(f"  - Text prompt index mapping: {dict(enumerate(self.text_prompt_ranks[client_id]))}")

    def _update_client_vision_prompts(self, client_id: int, client_weight: Dict,
                                    global_component: Dict[str, torch.Tensor], topk: int) -> None:
        """Update client vision prompts."""
        vision_prompt_indices = []
        updated_slots = []
        slot_aggregation = getattr(self.cfg.TRAINER.FEDMGP, 'SLOT_AGGREGATION', False)

        vision_key = 'visual.VPT' if 'visual.VPT' in client_weight else 'image_encoder.VPT'
        if client_id in self.vision_prompt_ranks and vision_key in client_weight:
            vpt = client_weight[vision_key]
            if vpt.dim() == 3:
                for rank, idx in enumerate(self.vision_prompt_ranks[client_id]):
                    slot = rank
                    if slot_aggregation and client_id in self.vision_prompt_slots:
                        slots = self.vision_prompt_slots[client_id]
                        if rank < len(slots) and slots[rank] is not None:
                            slot = slots[rank]

                    key = f'visual.VPT_{slot}'
                    if key in global_component and idx < vpt.size(0):
                        vpt[idx] = global_component[key][0]
                        vision_prompt_indices.append(idx)
                        updated_slots.append(slot)
            elif vpt.dim() == 2:
                slot = 0
                if (slot_aggregation and client_id in self.vision_prompt_slots and
                    self.vision_prompt_slots[client_id]):
                    assigned_slot = self.vision_prompt_slots[client_id][0]
                    if assigned_slot is not None:
                        slot = assigned_slot
                key = f'visual.VPT_{slot}'
                if key in global_component:
                    client_weight[vision_key] = global_component[key][0]
                    vision_prompt_indices.append(0)
                    updated_slots.append(slot)

        if vision_prompt_indices:
            if slot_aggregation:
                print(f"  - Vision prompt update: indices {vision_prompt_indices} (global slots {updated_slots})")
            else:
                print(f"  - Vision prompt update: indices {vision_prompt_indices} (rank order {list(range(len(vision_prompt_indices)))})")
            print(f"  - Vision prompt index mapping: {dict(enumerate(self.vision_prompt_ranks[client_id]))}")

    def _update_client_tcvp_meta_params(self, client_id: int, client_weight: Dict,
                                      global_component: Dict[str, torch.Tensor]) -> None:
        """Update participating clients with aggregated TCVP MetaNet parameters."""
        if not getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_ENABLED', False):
            return
        if not getattr(self.cfg.TRAINER.FEDMGP, 'TCVP_AGGREGATE_META_NET', True):
            return

        updated_params = []
        for key, value in global_component.items():
            if "tcvp_meta_net" in key and key in client_weight:
                client_weight[key] = copy.deepcopy(value)
                updated_params.append(key)

        if updated_params:
            print(f"  - TCVP MetaNet update: {len(updated_params)} parameters")

    def _extract_visual_prompt_with_fallback(self, client_id: int,
                                           trained_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract vision prompt with fallback mechanism."""
        vision_prompts = {}
        found_visual_vpt = False

        # Try extracting from state_dict first
        for name, param in trained_state_dict.items():
            if any(pattern in name for pattern in ["visual.VPT", "image_encoder.VPT", "VPT"]):
                found_visual_vpt = True
                clean_name = "visual.VPT"
                vision_prompts[clean_name] = copy.deepcopy(param)
                break

        # Fallback: try extracting from model directly
        if not found_visual_vpt:
            possible_paths = [
                (self.trainer.model, 'image_encoder.VPT'),
                (self.trainer.model, 'visual.VPT'),
                (self.trainer.model, 'VPT'),
                (getattr(self.trainer.model, 'module', None), 'image_encoder.VPT'),
                (getattr(self.trainer.model, 'module', None), 'visual.VPT'),
                (getattr(self.trainer.model, 'module', None), 'VPT'),
                (getattr(self.trainer.model, 'image_encoder', None), 'VPT'),
                (getattr(self.trainer.model, 'visual', None), 'VPT'),
                (getattr(getattr(self.trainer.model, 'module', None), 'image_encoder', None), 'VPT'),
                (getattr(getattr(self.trainer.model, 'module', None), 'visual', None), 'VPT'),
            ]

            vpt_param = None
            for obj, attr_path in possible_paths:
                if obj is None:
                    continue

                parts = attr_path.split('.')
                current = obj

                try:
                    for part in parts:
                        current = getattr(current, part)

                    if isinstance(current, torch.nn.Parameter):
                        vpt_param = current
                        break
                except (AttributeError, KeyError):
                    continue

            if vpt_param is not None:
                vision_prompts['visual.VPT'] = copy.deepcopy(vpt_param.data)

        return vision_prompts

    def _compute_pre_training_similarities(self, client_ids: List[int],
                                         global_component: Dict[str, torch.Tensor]) -> None:
        """Compute pre-training similarities after parameter update."""
        print(f"\nComputing round {self.current_round} pre-training similarities:")

        text_prompts_list = self._extract_text_prompts(client_ids)
        vision_prompts_list = self._extract_vision_prompts(client_ids)

        current_global_text_prompts = []
        current_global_vision_prompts = []

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_text_rank = -1
            max_vision_rank = -1

            for key in global_component.keys():
                if key.startswith('prompt_learner.ctx_'):
                    rank = int(key.split('_')[-1])
                    max_text_rank = max(max_text_rank, rank)
                elif key.startswith('visual.VPT_'):
                    rank = int(key.split('_')[-1])
                    max_vision_rank = max(max_vision_rank, rank)

            for rank in range(max_text_rank + 1):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    current_global_text_prompts.append(global_component[text_key][0])

            for rank in range(max_vision_rank + 1):
                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    current_global_vision_prompts.append(global_component[vision_key][0])
        else:
            for rank in range(topk):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    current_global_text_prompts.append(global_component[text_key][0])

                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    current_global_vision_prompts.append(global_component[vision_key][0])

        for client_idx, client_id in enumerate(client_ids):
            updated_text_indices = self.text_prompt_ranks.get(client_id, [])
            updated_vision_indices = self.vision_prompt_ranks.get(client_id, [])

            # Compute text prompt similarities
            if client_idx < len(text_prompts_list) and current_global_text_prompts:
                client_text_prompt = text_prompts_list[client_idx]
                text_similarities = self._compute_prompt_similarities(
                    client_text_prompt, current_global_text_prompts)

                text_sim_parts = []
                for idx, sim in text_similarities:
                    if idx in updated_text_indices:
                        text_sim_parts.append(f"{idx}:{sim:.4f}*")
                    else:
                        text_sim_parts.append(f"{idx}:{sim:.4f}")

                text_sim_str = ", ".join(text_sim_parts)
                print(f"Client {client_id}: Pre-training text similarities [{text_sim_str}] (updated: {updated_text_indices})")

            # Compute vision prompt similarities
            if client_idx < len(vision_prompts_list) and current_global_vision_prompts:
                client_vision_prompt = vision_prompts_list[client_idx]
                vision_similarities = self._compute_prompt_similarities(
                    client_vision_prompt, current_global_vision_prompts)

                vision_sim_parts = []
                for idx, sim in vision_similarities:
                    if idx in updated_vision_indices:
                        vision_sim_parts.append(f"{idx}:{sim:.4f}*")
                    else:
                        vision_sim_parts.append(f"{idx}:{sim:.4f}")

                vision_sim_str = ", ".join(vision_sim_parts)
                print(f"Client {client_id}: Pre-training vision similarities [{vision_sim_str}] (updated: {updated_vision_indices})")
