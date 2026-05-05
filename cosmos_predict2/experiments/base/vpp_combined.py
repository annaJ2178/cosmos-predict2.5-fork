# SPDX-License-Identifier: Apache-2.0
"""
Cosmos-Predict2.5 video2world post-training experiment entry for the
libero_mem + rmbench_v3 + real_robot combined baseline.

Install this file at:
    cosmos_predict2/experiments/base/vpp_combined.py

It mirrors cosmos_predict2/experiments/base/cosmos_nemo_assets.py and points
VideoDataset at /root/datasets/vpp_combined/ which is produced by
preprocess_combined.py.

Launch:
    torchrun --nproc_per_node=8 --master_port=12341 scripts/train.py \\
        --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- \\
        experiment=predict2_video2world_training_2b_vpp_combined \\
        job.wandb_mode=disabled
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import checkpoint_db
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Local paths to already-rsynced HF hub files. The on-disk layout is a plain
# copy of the snapshot directories (no blobs/symlink structure), so the stock
# `uvx hf download` path can't find them — we short-circuit `_hf_download` to
# return the local path for the exact cmd_args the checkpoint registry uses.
_LOCAL_HF_PATHS: dict[tuple[str, ...], str] = {
    # 2B pre-trained base model (3.9 GB, used as load_path below)
    (
        "nvidia/Cosmos-Predict2.5-2B",
        "--repo-type",
        "model",
        "--revision",
        "15a82a2ec231bc318692aa0456a36537c806e7d4",
        "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
    ): (
        "/root/models/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/"
        "snapshots/1a7f55340992562b20e81a93238e6722345c855d/"
        "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
    ),
    # Wan2.1 VAE tokenizer (485 MB)
    (
        "nvidia/Cosmos-Predict2.5-2B",
        "--repo-type",
        "model",
        "--revision",
        "f176dc95b4a70f53ce01c4b302851595e7322b00",
        "tokenizer.pth",
    ): (
        "/root/models/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/"
        "snapshots/6787e176dce74a101d922174a95dba29fa5f0c55/tokenizer.pth"
    ),
    # Qwen2.5-VL text encoder (13 GB, served via nvidia/Cosmos-Reason1-7B)
    (
        "nvidia/Cosmos-Reason1-7B",
        "--repo-type",
        "model",
        "--revision",
        "3210bec0495fdc7a8d3dbb8d58da5711eab4b423",
        "--include",
        "*",
    ): (
        "/root/models/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/"
        "snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
    ),
}


def _patched_hf_download(cmd_args: list[str]) -> str:
    """Offline replacement for checkpoint_db._hf_download.

    Looks up cmd_args in _LOCAL_HF_PATHS. Unknown args raise so we fail loud
    rather than trying to hit HuggingFace with a dead token.
    """
    key = tuple(cmd_args)
    if key in _LOCAL_HF_PATHS:
        return _LOCAL_HF_PATHS[key]
    raise RuntimeError(
        f"vpp_combined: no local mapping for HF download args {cmd_args}. "
        "Add it to _LOCAL_HF_PATHS in vpp_combined.py."
    )


checkpoint_db._hf_download = _patched_hf_download  # type: ignore[attr-defined]

LOCAL_CKPT_PATH = _LOCAL_HF_PATHS[
    (
        "nvidia/Cosmos-Predict2.5-2B",
        "--repo-type",
        "model",
        "--revision",
        "15a82a2ec231bc318692aa0456a36537c806e7d4",
        "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
    )
]


# --- Monkey-patch EveryNDrawSample to save annotated per-sample mp4s ---
# Stock cosmos writes a jpg grid locally (one jpg with 3 frames from each
# sample×guidance). For training-time eval we want libero_mem-style videos:
# GT on top, Pred on bottom, task description in a text bar, per sample.
from cosmos_predict2._src.predict2.callbacks.every_n_draw_sample import (  # noqa: E402
    EveryNDrawSample as _EveryNDrawSample,
)
from cosmos_predict2._src.imaginaire.visualize.video import (  # noqa: E402
    save_img_or_video as _save_img_or_video,
)
import torch as _torch  # noqa: E402
import numpy as _np  # noqa: E402
from einops import rearrange as _rearrange  # noqa: E402


def _put_text_outlined(frame, text, position, scale=0.5, color=(255, 255, 255), thickness=1):
    import cv2
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame


def _save_annotated_sample_mp4(
    pred_ct_h_w: _torch.Tensor,
    gt_ct_h_w: _torch.Tensor,
    caption: str,
    guidance_label: str,
    out_path_wo_ext: str,
    fps: int = 16,
    text_bar_h: int = 50,
) -> None:
    """Save a GT-top / Pred-bottom comparison mp4 with text bar.

    Args:
        pred_ct_h_w: [C, T, H, W] in [0, 1]
        gt_ct_h_w:   [C, T, H, W] in [0, 1]
        caption: raw task description string
        guidance_label: e.g. "Pred (g=7.0)"
        out_path_wo_ext: full path without extension
    """
    pred = (pred_ct_h_w.cpu().float().clamp(0, 1).numpy() * 255).astype(_np.uint8)  # [C, T, H, W]
    gt = (gt_ct_h_w.cpu().float().clamp(0, 1).numpy() * 255).astype(_np.uint8)
    pred = _np.transpose(pred, (1, 2, 3, 0))  # [T, H, W, C]
    gt = _np.transpose(gt, (1, 2, 3, 0))
    T, H, W, _ = pred.shape
    frame_h = 2 * H + text_bar_h

    canvas_seq = _np.zeros((T, frame_h, W, 3), dtype=_np.uint8)
    for t in range(T):
        canvas = canvas_seq[t]
        canvas[:H] = gt[t]
        canvas[H : 2 * H] = pred[t]
        _put_text_outlined(canvas, "GT", (8, 22), scale=0.6)
        _put_text_outlined(canvas, guidance_label, (8, H + 22), scale=0.6)
        _put_text_outlined(canvas, f"t={t}/{T - 1}", (W - 90, 22), scale=0.5)
        text_y = 2 * H + 18
        max_chars = max(8, W // 8)
        for line_idx in range(0, min(len(caption), 3 * max_chars), max_chars):
            line = caption[line_idx : line_idx + max_chars]
            _put_text_outlined(canvas, line, (8, text_y), scale=0.42, color=(200, 230, 255))
            text_y += 16

    # Cosmos's easy_io dumps mp4 via imageio; we hand off a [C, T, H, W] float
    # tensor in [0, 1] and it takes care of the encoder.
    canvas_tensor = _torch.from_numpy(canvas_seq).permute(3, 0, 1, 2).float() / 255.0
    _save_img_or_video(canvas_tensor, out_path_wo_ext, fps=fps)


_orig_run_save = _EveryNDrawSample.run_save


def _patched_run_save(self, to_show, batch_size, base_fp_wo_ext):
    local_path = _orig_run_save(self, to_show, batch_size, base_fp_wo_ext)
    if self.rank != 0:
        return local_path
    try:
        stacked = (1.0 + _torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        if stacked.shape[3] <= 1:
            return local_path  # single-frame image → jpg already written by orig
        n_viz = min(self.n_viz_sample, batch_size)
        # to_show order: [pred_guidance0, pred_guidance1, ..., raw_gt]
        gt_b = stacked[-1, :n_viz]  # [n_viz, c, t, h, w]
        preds_n = stacked[:-1, :n_viz]  # [n_guidance, n_viz, c, t, h, w]
        captions = getattr(self, "_current_captions", None) or ["(no caption)"] * n_viz
        guidance_list = list(getattr(self, "guidance", [7.0]))
        for g_idx, g_val in enumerate(guidance_list):
            for s_idx in range(n_viz):
                cap = captions[s_idx] if s_idx < len(captions) else "(no caption)"
                _save_annotated_sample_mp4(
                    pred_ct_h_w=preds_n[g_idx, s_idx],
                    gt_ct_h_w=gt_b[s_idx],
                    caption=cap,
                    guidance_label=f"Pred (g={g_val:.1f})",
                    out_path_wo_ext=f"{self.local_dir}/{base_fp_wo_ext}_s{s_idx}_g{g_val:.1f}",
                    fps=self.fps,
                )
    except Exception as _e:
        import traceback
        print(f"[vpp_combined] annotated mp4 save failed: {_e}\n{traceback.format_exc()}", flush=True)
    return local_path


_EveryNDrawSample.run_save = _patched_run_save


# Slice data_batch to bound sample-gen memory and stash ai_caption on the
# callback instance so run_save can label the mp4s. cosmos uses
# n_sample=bs[0], which at bs=16 pushes per-GPU peak to ~140 GB when colocated
# with other tenants. Cap at 4 samples for in-training viz.
_SAMPLE_GEN_MAX_BS = 4
_orig_every_n_impl = _EveryNDrawSample.every_n_impl


def _patched_every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
    bs = None
    for _v in data_batch.values():
        if _torch.is_tensor(_v) and _v.dim() >= 1:
            bs = _v.shape[0]
            break
    if bs and bs > _SAMPLE_GEN_MAX_BS:
        sliced = {}
        for k, v in data_batch.items():
            if _torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == bs:
                sliced[k] = v[:_SAMPLE_GEN_MAX_BS]
            elif isinstance(v, list) and len(v) == bs:
                sliced[k] = v[:_SAMPLE_GEN_MAX_BS]
            else:
                sliced[k] = v
        data_batch = sliced
    caps = data_batch.get("ai_caption")
    if isinstance(caps, (list, tuple)):
        self._current_captions = [str(c) for c in caps[:_SAMPLE_GEN_MAX_BS]]
    elif isinstance(caps, str):
        self._current_captions = [caps]
    else:
        self._current_captions = None
    return _orig_every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration)


_EveryNDrawSample.every_n_impl = _patched_every_n_impl


# Make run_at_start fire on the first post-resume step too (stock EveryN
# gates it on `iteration == 1`, so it never fires when resuming from a
# checkpoint). Fires exactly once per callback instance per process.
from cosmos_predict2._src.imaginaire.callbacks.every_n import EveryN as _EveryN  # noqa: E402
from cosmos_predict2._src.imaginaire.utils import distributed as _distributed  # noqa: E402

_orig_on_step_end = _EveryN.on_training_step_end


def _patched_on_step_end(self, model, data_batch, output_batch, loss, iteration=0):
    if getattr(self, "run_at_start", False) and not getattr(self, "_run_at_start_fired", False):
        self._run_at_start_fired = True
        if getattr(self, "every_n", 0) != 0:
            try:
                trainer = self.trainer
                self.every_n_impl(trainer, model, data_batch, output_batch, loss, iteration)
                if getattr(self, "barrier_after_run", True):
                    _distributed.barrier()
            except Exception as _e:
                print(f"[vpp_combined] run_at_start fire failed: {_e}", flush=True)
    return _orig_on_step_end(self, model, data_batch, output_batch, loss, iteration)


_EveryN.on_training_step_end = _patched_on_step_end


# libero_mem + rmbench_v3 + real_robot unified video dataset
example_video_dataset_vpp_combined = L(VideoDataset)(
    dataset_dir="/root/datasets/vpp_combined",
    num_frames=17,
    video_size=(256, 256),
)

dataloader_train_vpp_combined = L(get_generic_dataloader)(
    dataset=example_video_dataset_vpp_combined,
    sampler=L(get_sampler)(dataset=example_video_dataset_vpp_combined),
    batch_size=16,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

predict2_video2world_training_2b_vpp_combined = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_vpp_stage1",
        group="video2world",
        name="2b_vpp_combined",
    ),
    dataloader_train=dataloader_train_vpp_combined,
    checkpoint=dict(
        save_iter=10_000,
        load_path=LOCAL_CKPT_PATH,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[50_000],
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=50_000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(
                every_n=2_000,
                save_s3=False,
                n_viz_sample=2,
                num_sampling_step=20,
                guidance=[7.0],
                run_at_start=True,
            ),
            every_n_sample_ema=dict(
                every_n=2_000,
                save_s3=False,
                n_viz_sample=2,
                num_sampling_step=20,
                guidance=[7.0],
                run_at_start=True,
            ),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
)

cs = ConfigStore.instance()
for _item in [predict2_video2world_training_2b_vpp_combined]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
