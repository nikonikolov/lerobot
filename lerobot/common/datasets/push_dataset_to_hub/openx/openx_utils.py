import io
import numpy as np
import torch
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml

from lerobot.common.datasets.push_dataset_to_hub.openx.transforms import OPENX_STANDARDIZATION_TRANSFORMS

with open("lerobot/common/datasets/push_dataset_to_hub/openx/configs.yaml") as f:
    _openx_list = yaml.safe_load(f)

OPENX_DATASET_CONFIGS = _openx_list["OPENX_DATASET_CONFIGS"]


def tf_to_torch(data):
    return torch.from_numpy(data.numpy())


def tf_img_convert(img):
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


def make_tf_dataset(raw_dir: str, openx_dataset_name: str, split: str) -> tf.data.Dataset:
    ds_builder = tfds.builder_from_directory(str(raw_dir))

    if split == 'test' and 'val' in ds_builder.info.splits:
        split = 'val'
    elif split == 'val' and 'test' in ds_builder.info.splits:
        split = 'test'

    dataset = ds_builder.as_dataset(
        split=split,
        decoders={"steps": tfds.decode.SkipDecoding()},
    )

    dataset_info = ds_builder.info

    ds_length = len(dataset)
    dataset = dataset.take(ds_length)
    # "flatten" the dataset as such we can apply trajectory level map() easily
    # each [obs][key] has a shape of (frame_size, ...)
    dataset = dataset.enumerate().map(_broadcast_metadata_rlds)

    # we will apply the standardization transform if the dataset_name is provided
    # if the dataset name is not provided and the goal is to convert any rlds formatted dataset
    # search for 'image' keys in the observations
    if openx_dataset_name is not None:
        assert openx_dataset_name in OPENX_STANDARDIZATION_TRANSFORMS
        transform_fn = OPENX_STANDARDIZATION_TRANSFORMS[openx_dataset_name]
        dataset = dataset.map(transform_fn)

        image_keys = OPENX_DATASET_CONFIGS[openx_dataset_name]["image_obs_keys"]
    else:
        obs_keys = dataset_info.features["steps"]["observation"].keys()
        image_keys = [key for key in obs_keys if "image" in key]

    ram_budget_gb = 8
    options = tf.data.Options()
    options.threading.private_threadpool_size = 8
    options.autotune.ram_budget = ram_budget_gb * 1024 * 1024 * 1024  # GB --> Bytes
    dataset = dataset.with_options(options)

    return dataset, image_keys


def get_tf_dataset_num_transitions(raw_dir: str, openx_dataset_name: str, split: str):
    tf_dataset, _ = make_tf_dataset(raw_dir, openx_dataset_name, split=split)
    total_transitions = 0

    for episode in iter(tf_dataset):
        total_transitions += episode["action"].shape[0]

    return total_transitions


def verify_hf_dataset_correctness(
    hf_dataset, gt_num_eps: int, openx_dataset_name: str, raw_dir: str, split: str
):
    episode_index = np.array(hf_dataset['episode_index'])
    if not np.all(np.diff(episode_index) >= 0):
        raise ValueError("Corrupt episode_index")

    if (num_eps := len(np.unique(episode_index))) != gt_num_eps:
        raise ValueError(
            f"hf_dataset contains {num_eps} episodes, but source dataset contains {gt_num_eps} episodes"
        )

    gt_num_transitions = get_tf_dataset_num_transitions(
        raw_dir=raw_dir,
        openx_dataset_name=openx_dataset_name,
        split=split,
    )

    if len(hf_dataset) != gt_num_transitions:
        raise ValueError(
            f"hf_dataset contains {len(hf_dataset)} transitions, "
            f"but source dataset contains {gt_num_transitions} transitions"
        )


def pil_to_jpeg(image: PIL.Image.Image) -> PIL.Image.Image:
    with io.BytesIO() as f:
        image.save(f, format='JPEG')
        f.seek(0)
        image = PIL.Image.open(f)
        image.load()
    return image
