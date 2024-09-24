from typing import Any, Tuple

import io
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
import datasets

from lerobot.common.datasets.push_dataset_to_hub.openx.min_transforms import OPENX_STANDARDIZATION_TRANSFORMS

with open("lerobot/common/datasets/push_dataset_to_hub/openx/configs.yaml") as f:
    _openx_list = yaml.safe_load(f)

OPENX_DATASET_CONFIGS = _openx_list["OPENX_DATASET_CONFIGS"]

SKIP_KEYS = [
    'is_first',
    'is_last',
    'is_terminal',
    'natural_language_embedding',
]
METADATA_SKIP_KEYS = [
    'file_path',
    'episode_metadata.file_path',
]



def tf_to_torch(data):
    return torch.from_numpy(data.numpy())


def tf_img_to_np(img):
    if img.dtype == tf.string:
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    elif img.dtype != tf.uint8:
        raise ValueError(f"Unsupported image dtype: found with dtype {img.dtype}")
    return img.numpy()


def _broadcast_metadata_rlds(i: tf.Tensor, traj: dict) -> dict:
    """
    In the RLDS format, each trajectory has some top-level metadata that is explicitly separated out, and a "steps"
    entry. This function moves the "steps" entry to the top level, broadcasting any metadata to the length of the
    trajectory. This function also adds the extra metadata fields `_len`, `_traj_index`, and `_frame_index`.

    NOTE: adapted from DLimp library https://github.com/kvablack/dlimp/
    """
    steps = traj.pop("steps")

    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]

    # broadcast metadata to the length of the trajectory
    # metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)
    metadata = traj

    # put steps back in
    assert "traj_metadata" not in steps
    traj = {**steps, "traj_metadata": metadata}

    assert "_len" not in traj
    assert "_traj_index" not in traj
    assert "_frame_index" not in traj
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)

    return traj


def make_tf_dataset(source_path: str, dataset_name: str, split: str) -> tf.data.Dataset:
    ds_builder = tfds.builder_from_directory(str(source_path))

    if split == 'test' and 'val' in ds_builder.info.splits:
        split = 'val'
    elif split == 'val' and 'test' in ds_builder.info.splits:
        split = 'test'

    dataset = ds_builder.as_dataset(
        split=split,
        decoders={"steps": tfds.decode.SkipDecoding()},
    )

    ds_length = len(dataset)
    dataset = dataset.take(ds_length)
    # "flatten" the dataset as such we can apply trajectory level map() easily
    # each [obs][key] has a shape of (frame_size, ...)
    dataset = dataset.enumerate().map(_broadcast_metadata_rlds)

    # we will apply the standardization transform if the dataset_name is provided
    # if the dataset name is not provided and the goal is to convert any rlds formatted dataset
    # search for 'image' keys in the observations
    assert dataset_name in OPENX_STANDARDIZATION_TRANSFORMS
    transform_fn = OPENX_STANDARDIZATION_TRANSFORMS[dataset_name]
    dataset = dataset.map(transform_fn)

    ram_budget_gb = 8
    options = tf.data.Options()
    options.threading.private_threadpool_size = 8
    options.autotune.ram_budget = ram_budget_gb * 1024 * 1024 * 1024  # GB --> Bytes
    dataset = dataset.with_options(options)

    return dataset


def get_tf_dataset_num_transitions(source_path: str, dataset_name: str, split: str):
    tf_dataset = make_tf_dataset(source_path, dataset_name, split=split)
    total_transitions = 0

    for episode in iter(tf_dataset):
        total_transitions += episode["action"].shape[0]

    return total_transitions


def verify_hf_dataset_correctness(
    hf_dataset, metadata_dataset, dataset_name: str, source_path: str, split: str
):
    ds_builder = tfds.builder_from_directory(str(source_path))

    if split == 'test' and 'val' in ds_builder.info.splits:
        split = 'val'

    num_eps = ds_builder.info.splits[split].num_examples

    episode_index = np.array(hf_dataset['episode_index'])
    if not np.all(np.diff(episode_index) >= 0):
        raise ValueError("Corrupt episode_index")

    if (num_eps := len(np.unique(episode_index))) != num_eps:
        raise ValueError(
            f"hf_dataset contains {num_eps} episodes, but source dataset contains {num_eps} episodes"
        )

    gt_num_transitions = get_tf_dataset_num_transitions(
        source_path=source_path,
        dataset_name=dataset_name,
        split=split,
    )

    if len(hf_dataset) != gt_num_transitions:
        raise ValueError(
            f"hf_dataset contains {len(hf_dataset)} transitions, "
            f"but source dataset contains {gt_num_transitions} transitions"
        )
    
    if metadata_dataset is not None:
        assert len(metadata_dataset) == num_eps, f"Ep metadata: {len(metadata_dataset)} != {num_eps}"


def tf_element_spec_to_hf_features(data_dict: dict[str, Any], element_spec):
    features = {}

    for key in data_dict.keys():
        # Extra data added by us
        if key in ['episode_index', 'frame_index', 'index']:
            features[key] = datasets.Value(dtype="int64", id=None)
            continue
        if key in ['timestamp', 'next.reward']:
            features[key] = datasets.Value(dtype="float32", id=None)
            continue

        if key in ['next.done']:
            features[key] = datasets.Value(dtype="bool", id=None)
            continue

        # Images
        if key.startswith("observation.images"):
            features[key] = datasets.Image()
            continue

        # Rename control -> action for the search in spec
        if key.startswith('control'):
            spec = get_key_recursive(key.replace('control', 'action'), element_spec)
        else:
            spec = get_key_recursive(key, element_spec)

        shape = spec.shape[1:]

        # Scalar
        if shape.ndims == 0:
            features[key] = datasets.Value(dtype=spec.dtype.name, id=None)
        # 1D Array
        elif shape.ndims == 1:
            features[key] = datasets.Sequence(
                length=shape.num_elements(), feature=datasets.Value(dtype=spec.dtype.name, id=None)
            )
        # 2D Array
        elif shape.ndim == 2:
            features[key] = datasets.Array2D(shape=tuple(shape.as_list()), dtype=spec.dtype.name)
        else:        
            raise ValueError(f"Uknown formatting for key {key} with shape {spec.shape}")

    return features


def metadata_dict_to_hf_dataset(metadata_dict: dict[str, Any], element_spec: dict[str, Any]) -> datasets.Dataset:
    # metadata_spec = dict(element_spec).pop('steps')
    metadata_spec = element_spec['traj_metadata']
    features = tf_element_spec_to_hf_features(metadata_dict, metadata_spec)
    hf_dataset = datasets.Dataset.from_dict(metadata_dict, features=datasets.Features(features))
    return hf_dataset


def ep_dict_to_hf_dataset(ep_dict: dict[str, Any], element_spec: dict[str, Any]) -> datasets.Dataset:
    # steps is already expanded in the element spec
    features = tf_element_spec_to_hf_features(ep_dict, element_spec)
    hf_dataset = datasets.Dataset.from_dict(ep_dict, features=datasets.Features(features))
    return hf_dataset


def extract_episode_data(
    episode: dict[str, Any],
    tf_image_keys: list[str],
    tf_depth_keys: list[str],
    element_spec: dict[str, Any],
    skip_imgs: bool = False,
) -> dict[str, Any]:
    ep_dict = {}

    num_frames = episode["reward"].shape[0]

    for section in ['observation', 'action']:
        if section == 'action' and isinstance(episode[section], tf.Tensor):
            ep_dict[f"control"] = tf_to_torch(episode[section])
            continue

        for key, value in episode[section].items():
            # Skip any keys we don't want in the final dataset
            if key in SKIP_KEYS:
                continue

            # Process RGB images
            if key in tf_image_keys:
                import ipdb; ipdb.set_trace()
                if skip_imgs:
                    continue
                data = [PIL.Image.fromarray(tf_img_to_np(image)) for image in value]

            # Process depth images
            elif key in tf_depth_keys:
                if skip_imgs:
                    continue
                # TODO: Process depth image
                raise NotImplementedError("Can't process depth images yet")

            else:
                # Process tensor data
                if element_spec[section][key].dtype == tf.string:
                    data = [str(x.numpy().decode()) for x in value]
                else:
                    data = tf_to_torch(value)

            # Make it easy to identify images in the observation
            if key in tf_image_keys or key in tf_depth_keys:
                key = f"images.{key}"

            if section == 'action':
                ep_dict[f"control.{key}"] = data
            else:
                ep_dict[f"observation.{key}"] = data

    # last step of demonstration is considered done
    done = torch.zeros(num_frames, dtype=torch.bool)
    done[-1] = True

    ep_dict["next.reward"] = tf_to_torch(episode["reward"]).float()
    ep_dict["next.done"] = done

    return ep_dict


def extract_episode_metadata(episode: dict[str, Any], ep_idx: int) -> dict[str, Any]:
    assert 'traj_metadata' in episode, episode.keys()

    metadata = {}
    ep_metadata = episode['traj_metadata']

    for key, value in iterate_metadata(ep_metadata):
        if key in METADATA_SKIP_KEYS:
            continue
        assert value.shape.ndims == 0
        if value.dtype == tf.string:
            value = [str(value.numpy().decode())]
        else:
            # TODO: Is this correct
            value = [tf_to_torch(value).item()]
        metadata[key] = [value]
    
    metadata['episode_index'] = [ep_idx]

    return metadata


def pil_to_jpeg(image: PIL.Image.Image) -> PIL.Image.Image:
    with io.BytesIO() as f:
        image.save(f, format='JPEG')
        f.seek(0)
        image = PIL.Image.open(f)
        image.load()
    return image


def shard_index_to_name(index: int) -> str:
    return "shard_" + "0" * (10 - len(str(index))) + str(index)


def iterate_metadata(d, parent_key=''):
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            # Recursively yield from nested dicts
            yield from iterate_metadata(value, full_key)
        else:
            yield full_key, value


def get_key_recursive(key, spec: dict[str, Any]) -> Any:
    if '.' not in key:
        return spec[key]
    prefix, nest_key = key.split('.', 1)
    return get_key_recursive(nest_key, spec[prefix])
