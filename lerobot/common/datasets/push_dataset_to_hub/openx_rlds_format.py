#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
For https://github.com/google-deepmind/open_x_embodiment (OPENX) datasets.

Example:
    python lerobot/scripts/push_dataset_to_hub.py \
        --raw-dir /hdd/tensorflow_datasets/bridge_dataset/1.0.0/ \
        --repo-id youliangtan/sampled_bridge_data_v2 \
        --raw-format openx_rlds.bridge_orig \
        --episodes 3 4 5 8 9

Exact dataset fps defined in openx/config.py, obtained from:
    https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0&range=R:R
"""

import math
import functools

import os
import numpy as np
import tensorflow_datasets as tfds
import torch
import tqdm
import datasets
import multiprocessing
from typing import Any

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)

from lerobot.common.datasets.push_dataset_to_hub.openx.openx_utils import (
    OPENX_DATASET_CONFIGS,
    make_tf_dataset,
    verify_hf_dataset_correctness,
    shard_index_to_name,
    extract_episode_data,
    extract_episode_metadata,
    ep_dict_to_hf_dataset,
    metadata_dict_to_hf_dataset,
)


def from_raw_to_lerobot_format(
    source_path: str,
    output_path: str,
    dataset_name: str,
    episodes_per_shard: int = 256,
    num_workers: int = 32,
    split: str = 'train',
):
    if "fps" not in OPENX_DATASET_CONFIGS[dataset_name]:
        raise ValueError(
            "fps for this dataset is not specified in openx/configs.py yet," "means it is not yet tested"
        )
    fps = OPENX_DATASET_CONFIGS[dataset_name]["fps"]

    hf_dataset, metadata_dataset = convert_rlds_to_hf_dataset(
        source_path=source_path,
        output_path=output_path,
        dataset_name=dataset_name,
        episodes_per_shard=episodes_per_shard,
        num_workers=num_workers,
        split=split,
    )

    if "index" in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns("index")
    hf_dataset = hf_dataset.add_column("index", np.arange(0, len(hf_dataset), 1))

    hf_dataset.set_transform(hf_transform_to_torch)

    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": False,
    }

    return hf_dataset, metadata_dataset, episode_data_index, info


def convert_rlds_to_hf_dataset(
    source_path: str,
    output_path: str,
    dataset_name: str,
    episodes_per_shard: int = 16,
    num_workers: int = 32,
    split: str = 'train',
):
    ds_builder = tfds.builder_from_directory(source_path)

    if split == 'test' and 'val' in ds_builder.info.splits:
        split = 'val'

    dataset_info = ds_builder.info    
    ds_length = ds_builder.info.splits[split].num_examples

    print("dataset_info: ", dataset_info)
    print("num episodes: ", ds_length)

    tmp_shards_dir = os.path.join(output_path, 'tmp_hf_shards')  # Stores shards
    tmp_metadata_dir = os.path.join(output_path, 'tmp_hf_metadata')  # Stores episode metadata
 
    os.makedirs(tmp_shards_dir, exist_ok=True)
    os.makedirs(tmp_metadata_dir, exist_ok=True)

    # Use for debugging or when the dataset is very small and saving shards to disk will slow things down
    if num_workers == 0:
        hf_dataset, metadata_dataset = episodes_to_hf_dataset_shard(
            start_idx=0,
            end_idx=ds_length,
            dataset_output_path=None,
            metadata_output_path=None,
            dataset_name=dataset_name,
            source_path=source_path,
            split=split,
            save=False, 
        )
    
    else:
        config = make_mp_processing_configuration(
            source_path=source_path,
            dataset_name=dataset_name,
            episodes_per_shard=episodes_per_shard,
            num_workers=num_workers,
            ds_length=ds_length,
            output_path=output_path,
            split=split,
        )
        start_indices = config['start_indices']
        end_indices = config['end_indices']
        shard_paths = config['shard_paths']
        metadata_shard_paths = config['metadata_shard_paths']

        with multiprocessing.Pool(num_workers, maxtasksperchild=1) as pool:
            pool.starmap(
                functools.partial(
                    episodes_to_hf_dataset_shard,
                    dataset_name=dataset_name,
                    source_path=source_path,
                    split=split,
                    save=True,
                ),
                list(zip(start_indices, end_indices, shard_paths, metadata_shard_paths))
            )

        data_shard_paths = [
            os.path.join(tmp_shards_dir, shard_name) for shard_name in sorted(os.listdir(tmp_shards_dir))
        ]
        metadata_shard_paths = [
            os.path.join(tmp_metadata_dir, metadata_name) for metadata_name in sorted(os.listdir(tmp_metadata_dir))
        ]

        # Load and concatenate all shards and all metadata
        print("Loading datasets from disk")
        hf_dataset = datasets.concatenate_datasets(
            [datasets.Dataset.load_from_disk(tmp_path) for tmp_path in data_shard_paths]
        )
        metadata_dataset = datasets.concatenate_datasets(
            [datasets.Dataset.load_from_disk(tmp_path) for tmp_path in metadata_shard_paths]
        )

    verify_hf_dataset_correctness(
        hf_dataset, metadata_dataset, dataset_name=dataset_name, source_path=source_path, split=split
    )

    print("HF dataset created")

    return hf_dataset, metadata_dataset


def episodes_to_hf_dataset_shard(
    start_idx: int,
    end_idx: int,
    dataset_output_path: str,
    metadata_output_path: str,
    dataset_name: str,
    source_path: str,
    split: str,
    save: bool = True,
):
    fps = OPENX_DATASET_CONFIGS[dataset_name]["fps"]

    tf_dataset = make_tf_dataset(source_path, dataset_name, split=split)

    # Cut the dataset to the specified shard
    tf_dataset = tf_dataset.skip(start_idx).take(end_idx - start_idx)
    assert len(tf_dataset) == end_idx - start_idx, f"{len(tf_dataset)} != {end_idx - start_idx}"

    it = iter(tf_dataset)
    hf_datasets: list[datasets.Dataset] = []
    metadata_datasets: list[datasets.Dataset] = []

    tf_image_keys = OPENX_DATASET_CONFIGS[dataset_name]["image_obs_keys"]
    tf_depth_keys = OPENX_DATASET_CONFIGS[dataset_name]["depth_obs_keys"]
    # tf_state_obs_keys = OPENX_DATASET_CONFIGS[dataset_name]["state_obs_keys"]

    for ep_idx in tqdm.tqdm(range(start_idx, end_idx)):
        episode = next(it)

        num_frames = episode["action"].shape[0]

        ep_dict = extract_episode_data(
            episode,
            tf_image_keys=tf_image_keys,
            tf_depth_keys=tf_depth_keys,
            element_spec=tf_dataset.element_spec,
            skip_imgs=False,
        )

        metadata: dict[str, Any] = extract_episode_metadata(episode, ep_idx)

        # Make sure all entries are of the same length
        assert len(set([len(v) for v in ep_dict.values()] + [num_frames])) == 1

        # Add extra information for each transition
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict['index'] = torch.arange(num_frames)

        # Convert to hf dataset - takes significantly less memory
        hf_dataset = ep_dict_to_hf_dataset(ep_dict, element_spec=tf_dataset.element_spec)
        metadata_dataset = metadata_dict_to_hf_dataset(metadata, element_spec=tf_dataset.element_spec)

        hf_datasets.append(hf_dataset)
        metadata_datasets.append(metadata_dataset)

    hf_dataset = datasets.concatenate_datasets(hf_datasets)
    metadata_dataset = datasets.concatenate_datasets(metadata_datasets)

    if save:
        hf_dataset.save_to_disk(dataset_output_path)
        metadata_dataset.save_to_disk(metadata_output_path)
        return None, None
    return hf_dataset, metadata_dataset


def make_mp_processing_configuration(
    source_path: str,
    dataset_name: str,
    episodes_per_shard: int,
    num_workers: int,
    ds_length: int,
    output_path: str,
    split: str,
) -> dict[str, list[Any]]:
    """
    Get the configuration for processing the dataset - how to split into shards and which shards remain
    to be processed. Supports continuing from a previous run
    """

    tmp_shards_dir = os.path.join(output_path, 'tmp_hf_shards')  # Stores shards
    tmp_metadata_dir = os.path.join(output_path, 'tmp_hf_metadata')  # Stores episode metadata

    # See if anything has already been processed
    processed_shards_indices: list[int] = [
        int(filename.replace('shard_', '')) for filename in os.listdir(tmp_shards_dir)
    ]

    if len(processed_shards_indices) > 0:
        # Find episodes_per_shard for the shards already written
        tmp_dataset = datasets.Dataset.load_from_disk(
            os.path.join(tmp_shards_dir, os.listdir(tmp_shards_dir)[0])
        )
        written_episodes_per_shard = len(np.unique(tmp_dataset['episode_index']))

        last_shard = max(processed_shards_indices) + 1  # Exclusive
        last_episode = last_shard * written_episodes_per_shard

        missing_shard_indices: list[int] = [
            i for i in range(max(processed_shards_indices)) if i not in processed_shards_indices
        ]

        # Everything was already processed
        if len(missing_shard_indices) == 0 and last_episode >= ds_length:
            return {
                'start_indices': [],
                'end_indices': [],
                'source_paths': [],
                'names': [],
                'splits': [],
                'metadata_shard_paths': [],
                'shard_paths': [],
            }

        # Get start and end indices for missing shards. Use written_episodes_per_shard
        missing_start_indices = [
            shard * written_episodes_per_shard for shard in missing_shard_indices
        ]
        missing_end_indices = [
            (shard + 1) * written_episodes_per_shard for shard in missing_shard_indices
        ]

        # Get indices for remaining shard beyond the max written shard
        split_indices = np.arange(last_episode, ds_length, episodes_per_shard)
        if len(split_indices) > 0:
            if ds_length > split_indices[-1]:
                split_indices = split_indices.tolist() + [ds_length]
            else:
                split_indices = split_indices.tolist()

            beyond_start_indices = split_indices[:-1]
            beyond_end_indices = split_indices[1:]
            num_shards_beyond = len(start_indices)
        else:
            beyond_start_indices = []
            beyond_end_indices = []
            num_shards_beyond = 0
        
        # Combine missing and beyond indices
        start_indices = missing_start_indices + beyond_start_indices
        end_indices = missing_end_indices + beyond_end_indices

        # Get the shards that still need to be processed
        shard_indices = missing_shard_indices + (np.arange(num_shards_beyond) + last_shard).tolist()
        shard_paths = [
            os.path.join(tmp_shards_dir, shard_index_to_name(shard)) for shard in shard_indices
        ]
        metadata_paths = [
            os.path.join(tmp_metadata_dir, shard_index_to_name(shard)) for shard in shard_indices
        ]
        num_shards = len(shard_paths)

    else:
        episodes_per_shard = min(episodes_per_shard, math.ceil(ds_length / num_workers))

        split_indices = np.arange(0, ds_length // episodes_per_shard + 1) * episodes_per_shard
        if ds_length % episodes_per_shard != 0:
            split_indices = split_indices.tolist() + [ds_length]
        else:
            split_indices = split_indices.tolist()

        start_indices, end_indices = split_indices[:-1], split_indices[1:]
        num_shards = len(start_indices)

        shard_paths = [
            os.path.join(tmp_shards_dir, shard_index_to_name(shard)) for shard in range(num_shards)
        ]
        metadata_paths = [
            os.path.join(tmp_metadata_dir, shard_index_to_name(shard)) for shard in range(num_shards)
        ]

    names = [dataset_name] * num_shards
    source_paths = [source_path] * num_shards
    splits = [split] * num_shards

    return {
        'start_indices': start_indices,
        'end_indices': end_indices,
        'source_paths': source_paths,
        'names': names,
        'splits': splits,
        'shard_paths': shard_paths,
        'metadata_shard_paths': metadata_paths,
    }
