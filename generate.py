# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate_dual_branch(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                         remasking='low_confidence', mask_id=126336, factor=None,
                         main_threshold=0.9, spec_threshold=0.6, merge_window=3, evolution_interval=4):
    '''
    Generating using a dual-branch (main and speculative) strategy with a single batched forward pass.
    '''
    # Initialize two sequences: main and speculative
    main_x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    main_x[:, :prompt.shape[1]] = prompt.clone()
    spec_x = main_x.clone()

    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    batch_threshold = torch.tensor([main_threshold, spec_threshold], device=model.device).unsqueeze(1)
    nfe = 0
    
    # Process generation block by block
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        i = 0
        while True:
            # If the main branch has completed the current block, move to the next
            if (main_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            
            nfe += 1
            i += 1
            
            # Check if the spec branch is already complete for this block
            if (spec_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                # 1. Forward pass for only the main branch
                logits = model(main_x).logits
                
                # 2. Update the main branch
                mask_index = (main_x == mask_id)
                mask_index[:, current_block_end:] = False

                main_x = _update_branch(
                    logits, temperature, remasking, mask_index,
                    main_x, num_transfer_tokens=None, factor=factor,
                    threshold=batch_threshold[0:1]
                )

            else:
                # 1. Batched Forward Pass
                x_batch = torch.cat([main_x, spec_x], dim=0)
                logits_batch = model(x_batch).logits
                
                # 2. Batched Branch Update
                batch_mask_index = (x_batch == mask_id)
                batch_mask_index[:, current_block_end:] = False
                
                x_batch = _update_branch(
                    logits_batch, temperature, remasking, batch_mask_index,
                    x_batch, num_transfer_tokens=None, factor=factor,
                    threshold=batch_threshold
                )
                main_x, spec_x = x_batch.chunk(2, dim=0)

            # --- Vectorized Merge and Reset Logic ---
            if i > 0 and i % evolution_interval == 0:
                decoded_main = main_x != mask_id
                decoded_spec = spec_x != mask_id
                match_positions = (main_x == spec_x) & decoded_main & decoded_spec
                
                match_indices = match_positions[0].nonzero().squeeze(-1)
                
                if match_indices.numel() > 0:
                    half_window = merge_window // 2
                    offsets = torch.arange(-half_window, half_window + 1, device=model.device)
                    
                    potential_merge_pos = match_indices[:, None] + offsets[None, :]
                    
                    merge_indices = torch.unique(potential_merge_pos.flatten())
                    merge_indices = merge_indices[(merge_indices >= current_block_start) & (merge_indices < current_block_end)]
                    
                    update_mask = torch.zeros_like(main_x, dtype=torch.bool)
                    update_mask[0, merge_indices] = True
                    update_mask &= (main_x == mask_id)
                    update_mask &= (spec_x != mask_id)
                    # Don't overwrite the original match positions themselves
                    update_mask[0, match_indices] = False
                    
                    main_x[update_mask] = spec_x[update_mask]

            # Reset the speculative branch if it is not making more progress than the main branch
            main_decoded_count = (main_x[:, current_block_start:current_block_end] != mask_id).sum()
            spec_decoded_count = (spec_x[:, current_block_start:current_block_end] != mask_id).sum()
            
            if spec_decoded_count <= main_decoded_count:
                spec_x = main_x.clone()

    return main_x, nfe


@ torch.no_grad()
def generate_with_prefix_cache_dual_branch(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., 
                                            remasking='low_confidence', mask_id=126336, factor=None,
                                            main_threshold=0.9, spec_threshold=0.6, merge_window=3, evolution_interval=4):
    '''
    Generating using a dual-branch strategy with a shared prefix KV cache.
    '''
    # Initialize two sequences: main and speculative
    main_x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    main_x[:, :prompt.shape[1]] = prompt.clone()
    spec_x = main_x.clone()

    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    batch_threshold = torch.tensor([main_threshold, spec_threshold], device=model.device).unsqueeze(1)

    nfe = 0
    
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # --- First forward pass for the block to establish the cache ---
        output = model(main_x, use_cache=True)
        past_key_values = output.past_key_values
        logits = output.logits

        main_logits = logits
        spec_logits = logits.clone()
        nfe += 1

        # --- Batched update for both branches --- .expand(current_batch_size, -1).clone()
        x_batch = torch.cat([main_x, spec_x], dim=0)
        logits_batch = torch.cat([main_logits, spec_logits], dim=0)
        
        batch_mask_index = (x_batch == mask_id)
        batch_mask_index[:, current_block_end:] = False
        
        x_batch = _update_branch(
                logits_batch, temperature, remasking, batch_mask_index,
                x_batch, num_transfer_tokens=None, factor=factor,
                threshold=batch_threshold
            )
        main_x, spec_x = x_batch.chunk(2, dim=0)

        # The cache before the current block is identical for both branches
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values

        i = 1
        while True:
            if (main_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            
            nfe += 1
            i += 1

            if (spec_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                # 1. Prepare input and mask for the main branch's suffix
                main_input = main_x[:, current_block_start:]
                mask_index = (main_input == mask_id)
                mask_index[:, block_length:] = False
                
                # 2. Forward pass for only the main branch using the prefix cache
                logits = model(main_input, past_key_values=past_key_values, use_cache=True, refresh=False).logits
                
                # 3. Update the main branch
                updated_main_input = _update_branch(
                    logits, temperature, remasking, mask_index,
                    main_input, num_transfer_tokens=None, factor=factor,
                    threshold=batch_threshold[0:1]
                )
                main_x[:, current_block_start:] = updated_main_input

            else:
                # 1. Batched Forward Pass with Prefix Cache
                x_batch = torch.cat([main_x[:, current_block_start:], spec_x[:, current_block_start:]], dim=0)
                batch_mask_index = (x_batch == mask_id)
                batch_mask_index[:, block_length:] = False
                
                logits_batch = model(x_batch, past_key_values=past_key_values, use_cache=True, refresh=False).logits
                
                # 2. Batched update for the current block
                x_batch = _update_branch(
                    logits_batch, temperature, remasking, batch_mask_index,
                    x_batch, num_transfer_tokens=None, factor=factor,
                    threshold=batch_threshold
                )
                main_x[:, current_block_start:], spec_x[:, current_block_start:] = x_batch.chunk(2, dim=0)

            # ---Vectorized Merge and Reset Logic ---
            if i > 0 and i % evolution_interval == 0:
                decoded_main = main_x != mask_id
                decoded_spec = spec_x != mask_id
                match_positions = (main_x == spec_x) & decoded_main & decoded_spec
                
                match_indices = match_positions[0].nonzero().squeeze(-1)
                
                if match_indices.numel() > 0:
                    half_window = merge_window // 2
                    offsets = torch.arange(-half_window, half_window + 1, device=model.device)
                    
                    potential_merge_pos = match_indices[:, None] + offsets[None, :]
                    
                    merge_indices = torch.unique(potential_merge_pos.flatten())
                    merge_indices = merge_indices[(merge_indices >= current_block_start) & (merge_indices < current_block_end)]
                    
                    update_mask = torch.zeros_like(main_x, dtype=torch.bool)
                    update_mask[0, merge_indices] = True
                    update_mask &= (main_x == mask_id)
                    update_mask &= (spec_x != mask_id)
                    # Don't overwrite the original match positions themselves
                    update_mask[0, match_indices] = False
                    
                    main_x[update_mask] = spec_x[update_mask]
            
            main_decoded_count = (main_x[:, current_block_start:current_block_end] != mask_id).sum()
            spec_decoded_count = (spec_x[:, current_block_start:current_block_end] != mask_id).sum()
            
            if spec_decoded_count <= main_decoded_count:
                spec_x = main_x.clone()

    return main_x, nfe


@ torch.no_grad()
def generate_with_dual_cache_dual_branch(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., 
                                        remasking='low_confidence', mask_id=126336, factor=None,
                                        main_threshold=0.9, spec_threshold=0.6, merge_window=3, evolution_interval=4):
    '''
    Generating using a dual-branch strategy with a shared KV cache,
    '''
    # Initialize two sequences: main and speculative
    main_x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    main_x[:, :prompt.shape[1]] = prompt.clone()
    spec_x = main_x.clone()

    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    batch_threshold = torch.tensor([main_threshold, spec_threshold], device=model.device).unsqueeze(1)
    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # --- First forward pass ---
        # Since main and spec inputs are exactly the same, only forward main_x once
        output = model(main_x, use_cache=True)
        past_key_values = output.past_key_values
        logits = output.logits

        # main branch logits
        main_logits = logits
        # spec branch logits are just a copy of main branch
        spec_logits = main_logits.clone()
        nfe += 1

        x_batch = torch.cat([main_x, spec_x], dim=0)
        logits_batch = torch.cat([main_logits, spec_logits], dim=0)
        
        batch_mask_index = (x_batch == mask_id)
        batch_mask_index[:, current_block_end:] = False
        
        # --- Batched update for both branches ---
        x_batch = _update_branch(
                logits_batch, temperature, remasking, batch_mask_index,
                x_batch, num_transfer_tokens=None, factor=factor,
                threshold=batch_threshold
            )
        main_x, spec_x = x_batch.chunk(2, dim=0)

        i = 1
        replace_position = torch.zeros_like(main_x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        while True:
            # If the main branch has completed the current block, move to the next
            if (main_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            
            nfe += 1
            i += 1

            if (spec_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                # 1. Prepare input and mask for the main branch's current block
                main_input_block = main_x[:, current_block_start:current_block_end]
                mask_index_block = (main_input_block == mask_id)

                # 2. Forward pass for the main branch's block, providing the full cache and replace_position
                logits = model(
                    main_input_block, past_key_values=past_key_values, use_cache=True, 
                    replace_position=replace_position, refresh=False
                ).logits

                # 3. Update the main branch
                updated_main_block = _update_branch(
                    logits, temperature, remasking, mask_index_block,
                    main_input_block, num_transfer_tokens=None, factor=factor,
                    threshold=batch_threshold[0:1]
                )
                main_x[:, current_block_start:current_block_end] = updated_main_block

            else:
                # 1. Batched Forward Pass with Full Cache and Replace Position
                x_batch = torch.cat([
                    main_x[:, current_block_start:current_block_end], 
                    spec_x[:, current_block_start:current_block_end]
                ], dim=0)

                logits_batch = model(
                    x_batch, past_key_values=past_key_values, use_cache=True, 
                    replace_position=replace_position, refresh=False
                ).logits

                # 2. Batched update for the current block
                batch_mask_index_block = (x_batch == mask_id)
                x_batch = _update_branch(
                    logits_batch, temperature, remasking, batch_mask_index_block,
                    x_batch, num_transfer_tokens=None, factor=factor,
                    threshold=batch_threshold
                )
                main_x[:, current_block_start:current_block_end], spec_x[:, current_block_start:current_block_end] = x_batch.chunk(2, dim=0)


            # --- Vectorized Merge and Reset Logic ---
            if i > 0 and i % evolution_interval == 0:
                decoded_main = main_x != mask_id
                decoded_spec = spec_x != mask_id
                match_positions = (main_x == spec_x) & decoded_main & decoded_spec
                
                match_indices = match_positions[0].nonzero().squeeze(-1)
                
                if match_indices.numel() > 0:
                    # Create all window offsets at once
                    half_window = merge_window // 2
                    offsets = torch.arange(-half_window, half_window + 1, device=model.device)
                    
                    # Use broadcasting to create all potential merge positions
                    # Shape: (num_matches, merge_window_size)
                    potential_merge_pos = match_indices[:, None] + offsets[None, :]
                    
                    # Get unique indices within the current block
                    merge_indices = torch.unique(potential_merge_pos.flatten())
                    merge_indices = merge_indices[(merge_indices >= current_block_start) & (merge_indices < current_block_end)]
                    
                    # Create a boolean mask for the final update conditions
                    update_mask = torch.zeros_like(main_x, dtype=torch.bool)
                    update_mask[0, merge_indices] = True
                    update_mask &= (main_x == mask_id)
                    update_mask &= (spec_x != mask_id)
                    # Don't overwrite the original match positions themselves
                    update_mask[0, match_indices] = False
                    
                    # Perform the merge in one vectorized operation
                    main_x[update_mask] = spec_x[update_mask]

            main_decoded_count = (main_x[:, current_block_start:current_block_end] != mask_id).sum()
            spec_decoded_count = (spec_x[:, current_block_start:current_block_end] != mask_id).sum()
            
            if spec_decoded_count <= main_decoded_count:
                spec_x = main_x.clone()

    return main_x, nfe


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
            
            # # seems not use anymore
            # logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe

def _update_branch(logits, temperature, remasking, mask_index, x, num_transfer_tokens=None, threshold=None, factor=None):
    """Helper function for main and spec branch updates."""
    if factor is None:
        x0, transfer_index = get_transfer_index_parallel(
            logits, temperature, remasking, mask_index, x, 
            num_transfer_tokens=num_transfer_tokens, threshold=threshold
        )
    else:
        raise NotImplementedError("Factor-based dynamic updates not integrated into this parallel example.")
    
    x[transfer_index] = x0[transfer_index]
    return x


def get_transfer_index_parallel(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    # --- Step 1: Initial Token Prediction and Confidence Calculation (Unchanged) ---
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == 'random':
        x0_p = torch.rand_like(x0, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -torch.inf)

    # --- Step 2: Parallel Token Selection ---
    if threshold is not None:
        num_to_consider = mask_index.sum(dim=1, keepdim=True)
    else:
        num_to_consider = num_transfer_tokens

    max_k = num_to_consider.max().item()
    if max_k == 0:
        return x0, torch.zeros_like(x0, dtype=torch.bool)

    top_confidences, top_indices = torch.topk(confidence, k=max_k, dim=-1)

    # --- Step 3: Create Masks for Parallel Filtering (Core Correction) ---
    k_range = torch.arange(max_k, device=x0.device)[None, :]
    valid_k_mask = k_range < num_to_consider

    if threshold is not None:
        # Mask 2.1: Standard threshold check
        threshold_mask = top_confidences >= threshold # (b, max_k)

        # Mask 2.2: Create a mask that is only True for the first column (k=0), Top-1 Token
        is_top1_mask = torch.zeros_like(threshold_mask, dtype=torch.bool)
        if max_k > 0:
            is_top1_mask[:, 0] = True
        corrected_threshold_mask = threshold_mask | is_top1_mask

        final_selection_mask = valid_k_mask & corrected_threshold_mask
    else:
        final_selection_mask = valid_k_mask

    # --- Step 4: Use scatter_ to Parallelly Update Final Result ---
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    transfer_index.scatter_(dim=1, index=top_indices, src=final_selection_mask)

    return x0, transfer_index


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
