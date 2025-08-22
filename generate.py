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
            
            # --- Batched Forward Pass ---
            # Stack main and speculative sequences into a single batch of size 2
            x_batch = torch.cat([main_x, spec_x], dim=0)
            
            # Run a single forward pass to get logits for both branches
            logits_batch = model(x_batch).logits
            
            main_logits = logits_batch[0:1]
            spec_logits = logits_batch[1:2]

            # --- Main Branch Update ---
            main_mask_index = (main_x == mask_id)
            main_mask_index[:, current_block_end:] = False

            if factor is None:
                main_x0, main_transfer_index = get_transfer_index(
                    main_logits, temperature, remasking, main_mask_index, main_x, 
                    num_transfer_tokens=None, threshold=main_threshold
                )
            else:
                main_x0, main_transfer_index = get_transfer_index_dynamic(
                    main_logits, temperature, remasking, main_mask_index, main_x, 
                    num_transfer_tokens=None, factor=factor
                )
            main_x[main_transfer_index] = main_x0[main_transfer_index]

            # --- Speculative Branch Update ---
            # Only update the speculative branch if it still has masked tokens in the current block
            if (spec_x[:, current_block_start:current_block_end] == mask_id).sum() > 0:
                spec_mask_index = (spec_x == mask_id)
                spec_mask_index[:, current_block_end:] = False

                if factor is None:
                    spec_x0, spec_transfer_index = get_transfer_index(
                        spec_logits, temperature, remasking, spec_mask_index, spec_x, 
                        num_transfer_tokens=None, threshold=spec_threshold
                    )
                else:
                    spec_x0, spec_transfer_index = get_transfer_index_dynamic(
                        spec_logits, temperature, remasking, spec_mask_index, spec_x, 
                        num_transfer_tokens=None, factor=factor
                    )
                spec_x[spec_transfer_index] = spec_x0[spec_transfer_index]

            # --- Merge and Reset Logic ---
            # Periodically merge high-confidence tokens from the speculative branch to the main branch
            if i > 0 and i % evolution_interval == 0:
                decoded_main = main_x != mask_id
                decoded_spec = spec_x != mask_id
                # Find positions where tokens match and both are decoded
                match_positions = (main_x == spec_x) & decoded_main & decoded_spec
                
                if match_positions.sum() > 0:
                    match_indices = torch.where(match_positions[0])[0]
                    
                    for match_idx in match_indices:
                        # Skip matches outside the current working block
                        if not (current_block_start <= match_idx < current_block_end):
                            continue

                        # Define the merge window around the matching token
                        merge_start = max(current_block_start, int(match_idx - merge_window // 2))
                        merge_end = min(current_block_end, int(match_idx + merge_window // 2 + 1))
                        
                        for pos in range(merge_start, merge_end):
                            # Merge if main is [MASK] and spec has a decoded token
                            if (main_x[0, pos] == mask_id and spec_x[0, pos] != mask_id and pos != match_idx):
                                main_x[0, pos] = spec_x[0, pos]

            # Reset the speculative branch if it is not making more progress than the main branch
            main_decoded_count = (main_x[:, current_block_start:current_block_end] != mask_id).sum()
            spec_decoded_count = (spec_x[:, current_block_start:current_block_end] != mask_id).sum()
            
            if spec_decoded_count <= main_decoded_count:
                spec_x = main_x.clone()

    # Return the completed sequence from the main branch and the number of forward evaluations
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

    nfe = 0
    
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # --- First forward pass for the block to establish the cache ---
        # Stack main and speculative sequences into a single batch of size 2
        x_batch = torch.cat([main_x, spec_x], dim=0)
        
        # Run a single forward pass to get logits and the initial KV cache for the entire sequence
        output = model(x_batch, use_cache=True)
        past_key_values = output.past_key_values
        logits_batch = output.logits
        
        main_logits = logits_batch[0:1]
        spec_logits = logits_batch[1:2]
        nfe += 1
        
        # --- Update both branches based on the first full pass ---
        # Main Branch Update
        main_mask_index = (main_x == mask_id)
        main_mask_index[:, current_block_end:] = False
        if factor is None:
            main_x0, main_transfer_index = get_transfer_index(
                main_logits, temperature, remasking, main_mask_index, main_x, 
                num_transfer_tokens=None, threshold=main_threshold
            )
        else:
            main_x0, main_transfer_index = get_transfer_index_dynamic(
                main_logits, temperature, remasking, main_mask_index, main_x, 
                num_transfer_tokens=None, factor=factor
            )
        main_x[main_transfer_index] = main_x0[main_transfer_index]

        # Speculative Branch Update
        spec_mask_index = (spec_x == mask_id)
        spec_mask_index[:, current_block_end:] = False
        if factor is None:
            spec_x0, spec_transfer_index = get_transfer_index(
                spec_logits, temperature, remasking, spec_mask_index, spec_x, 
                num_transfer_tokens=None, threshold=spec_threshold
            )
        else:
            spec_x0, spec_transfer_index = get_transfer_index_dynamic(
                spec_logits, temperature, remasking, spec_mask_index, spec_x, 
                num_transfer_tokens=None, factor=factor
            )
        spec_x[spec_transfer_index] = spec_x0[spec_transfer_index]

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

            # --- Batched Forward Pass with Prefix Cache ---
            x_batch_block = torch.cat([main_x[:, current_block_start:], spec_x[:, current_block_start:]], dim=0)

            # Run forward pass on the current block using the shared prefix cache
            logits_batch_block = model(x_batch_block, past_key_values=past_key_values, use_cache=True).logits
            
            main_logits_block = logits_batch_block[0:1]
            spec_logits_block = logits_batch_block[1:2]

            # --- Main Branch Update (for the current block) ---
            main_mask_index_block = (main_x[:, current_block_start:] == mask_id)
            main_mask_index_block[:, block_length:] = False
            if factor is None:
                main_x0_block, main_transfer_index_block = get_transfer_index(
                    main_logits_block, temperature, remasking, main_mask_index_block, 
                    main_x[:, current_block_start:], num_transfer_tokens=None, threshold=main_threshold
                )
            else:
                main_x0_block, main_transfer_index_block = get_transfer_index_dynamic(
                    main_logits_block, temperature, remasking, main_mask_index_block, 
                    main_x[:, current_block_start:], num_transfer_tokens=None, factor=factor
                )
            main_x[:, current_block_start:][main_transfer_index_block] = main_x0_block[main_transfer_index_block]

            # --- Speculative Branch Update (for the current block) ---
            if (spec_x[:, current_block_start:current_block_end] == mask_id).sum() > 0:
                spec_mask_index_block = (spec_x[:, current_block_start:] == mask_id)
                spec_mask_index_block[:, block_length:] = False
                if factor is None:
                    spec_x0_block, spec_transfer_index_block = get_transfer_index(
                        spec_logits_block, temperature, remasking, spec_mask_index_block, 
                        spec_x[:, current_block_start:], num_transfer_tokens=None, threshold=spec_threshold
                    )
                else:
                    spec_x0_block, spec_transfer_index_block = get_transfer_index_dynamic(
                        spec_logits_block, temperature, remasking, spec_mask_index_block, 
                        spec_x[:, current_block_start:], num_transfer_tokens=None, factor=factor
                    )
                spec_x[:, current_block_start:][spec_transfer_index_block] = spec_x0_block[spec_transfer_index_block]

            # --- Merge and Reset Logic ---
            if i > 0 and i % evolution_interval == 0:
                decoded_main = main_x != mask_id
                decoded_spec = spec_x != mask_id
                match_positions = (main_x == spec_x) & decoded_main & decoded_spec
                
                if match_positions.sum() > 0:
                    match_indices = torch.where(match_positions[0])[0]
                    for match_idx in match_indices:
                        if not (current_block_start <= match_idx < current_block_end):
                            continue
                        merge_start = max(current_block_start, int(match_idx - merge_window // 2))
                        merge_end = min(current_block_end, int(match_idx + merge_window // 2 + 1))
                        for pos in range(merge_start, merge_end):
                            if (main_x[0, pos] == mask_id and spec_x[0, pos] != mask_id and pos != match_idx):
                                main_x[0, pos] = spec_x[0, pos]
            
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

    nfe = 0
    
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        # --- First forward pass ---
        x_batch = torch.cat([main_x, spec_x], dim=0)
        
        # Run a single forward pass to get logits and the initial KV cache for the entire sequence
        output = model(x_batch, use_cache=True)
        past_key_values = output.past_key_values
        logits_batch = output.logits
        
        main_logits = logits_batch[0:1]
        spec_logits = logits_batch[1:2]
        nfe += 1
        
        # Main Branch Update
        main_mask_index = (main_x == mask_id)
        main_mask_index[:, current_block_end:] = False
        if factor is None:
            main_x0, main_transfer_index = get_transfer_index(
                main_logits, temperature, remasking, main_mask_index, main_x, 
                num_transfer_tokens=None, threshold=main_threshold
            )
        else:
            main_x0, main_transfer_index = get_transfer_index_dynamic(
                main_logits, temperature, remasking, main_mask_index, main_x, 
                num_transfer_tokens=None, factor=factor
            )
        main_x[main_transfer_index] = main_x0[main_transfer_index]

        # Speculative Branch Update
        spec_mask_index = (spec_x == mask_id)
        spec_mask_index[:, current_block_end:] = False
        if factor is None:
            spec_x0, spec_transfer_index = get_transfer_index(
                spec_logits, temperature, remasking, spec_mask_index, spec_x, 
                num_transfer_tokens=None, threshold=spec_threshold
            )
        else:
            spec_x0, spec_transfer_index = get_transfer_index_dynamic(
                spec_logits, temperature, remasking, spec_mask_index, spec_x, 
                num_transfer_tokens=None, factor=factor
            )
        spec_x[spec_transfer_index] = spec_x0[spec_transfer_index]
        
        i = 1
        replace_position = torch.zeros_like(main_x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        while True:
            # If the main branch has completed the current block, move to the next
            if (main_x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            
            nfe += 1
            i += 1

            # --- Batched Forward Pass with Full Cache and Replace Position ---
            x_batch_block = torch.cat([
                main_x[:, current_block_start:current_block_end], 
                spec_x[:, current_block_start:current_block_end]
            ], dim=0)

            # Run forward pass on the current block, providing the full cache and the position to update it
            logits_batch_block = model(x_batch_block, past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits
            
            main_logits_block = logits_batch_block[0:1]
            spec_logits_block = logits_batch_block[1:2]

            # --- Main Branch Update (for the current block) ---
            main_mask_index_block = (main_x[:, current_block_start:current_block_end] == mask_id)
            if factor is None:
                main_x0_block, main_transfer_index_block = get_transfer_index(
                    main_logits_block, temperature, remasking, main_mask_index_block, 
                    main_x[:, current_block_start:current_block_end], num_transfer_tokens=None, threshold=main_threshold
                )
            else:
                main_x0_block, main_transfer_index_block = get_transfer_index_dynamic(
                    main_logits_block, temperature, remasking, main_mask_index_block, 
                    main_x[:, current_block_start:current_block_end], num_transfer_tokens=None, factor=factor
                )
            main_x[:, current_block_start:current_block_end][main_transfer_index_block] = main_x0_block[main_transfer_index_block]

            # --- Speculative Branch Update (for the current block) ---
            if (spec_x[:, current_block_start:current_block_end] == mask_id).sum() > 0:
                spec_mask_index_block = (spec_x[:, current_block_start:current_block_end] == mask_id)
                if factor is None:
                    spec_x0_block, spec_transfer_index_block = get_transfer_index(
                        spec_logits_block, temperature, remasking, spec_mask_index_block, 
                        spec_x[:, current_block_start:current_block_end], num_transfer_tokens=None, threshold=spec_threshold
                    )
                else:
                    spec_x0_block, spec_transfer_index_block = get_transfer_index_dynamic(
                        spec_logits_block, temperature, remasking, spec_mask_index_block, 
                        spec_x[:, current_block_start:current_block_end], num_transfer_tokens=None, factor=factor
                    )
                spec_x[:, current_block_start:current_block_end][spec_transfer_index_block] = spec_x0_block[spec_transfer_index_block]

            # --- Merge and Reset Logic ---
            if i > 0 and i % evolution_interval == 0:
                decoded_main = main_x != mask_id
                decoded_spec = spec_x != mask_id
                match_positions = (main_x == spec_x) & decoded_main & decoded_spec
                
                if match_positions.sum() > 0:
                    match_indices = torch.where(match_positions[0])[0]
                    for match_idx in match_indices:
                        if not (current_block_start <= match_idx < current_block_end):
                            continue
                        merge_start = max(current_block_start, int(match_idx - merge_window // 2))
                        merge_end = min(current_block_end, int(match_idx + merge_window // 2 + 1))
                        for pos in range(merge_start, merge_end):
                            if (main_x[0, pos] == mask_id and spec_x[0, pos] != mask_id and pos != match_idx):
                                main_x[0, pos] = spec_x[0, pos]
            
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
