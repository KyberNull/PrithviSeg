"""Phase-specific checkpointing and dataloader helpers."""

import torch
from torch import optim
from torch.utils.data import DataLoader


def _pad_and_stack_batch(batch):
    """Pad variable-size (image, mask) pairs to the max H/W in batch, then stack."""
    images, masks = zip(*batch)
    images = [torch.as_tensor(img) for img in images]
    masks = [torch.as_tensor(mask) for mask in masks]

    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images = []
    padded_masks = []
    for img, mask in zip(images, masks):
        pad_h = max_h - img.shape[-2]
        pad_w = max_w - img.shape[-1]
        pad = (0, pad_w, 0, pad_h)
        padded_images.append(torch.nn.functional.pad(img, pad, mode="constant", value=0.0))
        padded_masks.append(torch.nn.functional.pad(mask, pad, mode="constant", value=255.0))

    return torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0)


def load_checkpoint_train(*, path, model, start_epoch_default, logger, optimizer=None, scheduler=None, scaler=None):
    start_epoch = start_epoch_default
    new_segmentation_head = False

    try:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["model_state"]

        for k in list(state_dict.keys()):
            if "segmentation_head.0.weight" in k:
                model_state = model.state_dict()
                if k in model_state:
                    old_num_classes = state_dict[k].shape[0]
                    new_num_classes = model_state[k].shape[0]
                    if old_num_classes != new_num_classes:
                        logger.warning(f"Shape mismatch at {k}: {old_num_classes} -> {new_num_classes}. Dropping head.")
                        new_segmentation_head = True
                        del state_dict[k]
                        bias_key = k.replace("weight", "bias")
                        if bias_key in state_dict:
                            del state_dict[bias_key]
                break

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        has_train_state = all(k in ckpt for k in ("optim_state", "scheduler_state", "scaler_state"))
        if has_train_state and not new_segmentation_head and optimizer and scheduler and scaler:
            optimizer.load_state_dict(ckpt["optim_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")
        else:
            logger.info("Loaded model weights only (fresh optimizer/scheduler).")

    except FileNotFoundError:
        logger.warning("Checkpoint not found. Starting from scratch.")
    except (RuntimeError, KeyError) as err:
        logger.error(f"Incompatible checkpoint: {err}")
        logger.warning("Starting from scratch.")

    return start_epoch


def load_checkpoint_pretrain(
    *,
    model_path,
    model,
    train_loader,
    setup_scheduler_fn,
    get_adamw_param_groups_fn,
    learning_rate,
    weight_decay,
    grad_accum_steps,
    total_epochs,
    warmup_epochs,
    device,
    num_classes,
    logger,
):
    start_epoch = 0
    optimizer = optim.AdamW(get_adamw_param_groups_fn(model, weight_decay), lr=learning_rate)
    scheduler = setup_scheduler_fn(
        train_loader=train_loader,
        optimizer=optimizer,
        grad_accum_steps=grad_accum_steps,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        learning_rate=learning_rate,
        warmup_start_factor=0.1,
    )
    scaler = torch.GradScaler(enabled=(device.type == "cuda"))

    try:
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt["model_state"]
        ckpt_epoch = ckpt.get("epoch", -1) + 1

        keys_to_remove = []
        for k, v in list(state_dict.items()):
            if k.endswith("head.weight") and v.shape[0] != num_classes:
                keys_to_remove.append(k)
                bias_k = k[:-len("weight")] + "bias"
                if bias_k in state_dict:
                    keys_to_remove.append(bias_k)

        if keys_to_remove:
            try:
                sample_key = next(key for key in keys_to_remove if key.endswith("head.weight"))
                orig_classes = state_dict[sample_key].shape[0]
            except StopIteration:
                orig_classes = "unknown"
            logger.warning(
                f"Pre-trained head has {orig_classes} classes, but current task has {num_classes}. "
                f"Excluding head parameters: {keys_to_remove}"
            )
            for k in keys_to_remove:
                state_dict.pop(k, None)

        model.load_state_dict(state_dict, strict=False)
        is_pretrain_resume = 0 < ckpt_epoch < total_epochs
        has_train_state = all(k in ckpt for k in ("optim_state", "scheduler_state", "scaler_state"))

        if keys_to_remove:
            logger.warning("Head class mismatch detected earlier; resetting optimizer/scheduler/scaler.")
            start_epoch = 0
        elif is_pretrain_resume and has_train_state:
            optimizer.load_state_dict(ckpt["optim_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt_epoch
            logger.info("Resuming pretrain optimizer/scheduler state.")
        else:
            start_epoch = 0
            if ckpt_epoch >= total_epochs:
                logger.info("Pretrain checkpoint already complete; starting with fresh optimizer/scheduler.")
            else:
                logger.info("Using checkpoint model weights with fresh optimizer/scheduler/scaler.")

    except FileNotFoundError:
        logger.warning("Pretrain checkpoint not found. Starting from scratch.")
    except (RuntimeError, KeyError) as err:
        logger.error(f"Checkpoint incompatible with current model architecture: {err}")
        logger.warning("Starting from scratch.")

    logger.info(f"Resuming pretraining from epoch {start_epoch+1}")
    return model, optimizer, scheduler, scaler, start_epoch, train_loader


def get_train_dataloaders(
    *,
    geospatial_dataset_cls,
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    train_transform,
    eval_transform,
    batch_size,
    num_workers,
    prefetch_factor,
    pin_memory,
    val_batch_size,
):
    train_dataset = geospatial_dataset_cls(img_dir=train_img_dir, img_mask=train_mask_dir, transform=train_transform)
    val_dataset = geospatial_dataset_cls(img_dir=val_img_dir, img_mask=val_mask_dir, transform=eval_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        pin_memory=pin_memory,
        collate_fn=_pad_and_stack_batch,
    )
    return train_dataloader, val_dataloader


def get_pretrain_dataloaders(
    *,
    loveda_cls,
    root,
    scenes,
    train_transform,
    eval_transform,
    batch_size,
    num_workers,
    prefetch_factor,
    pin_memory,
    val_batch_size,
):
    train_dataset = loveda_cls(root=root, split="train", scene=scenes, transforms=train_transform, download=False)
    val_dataset = loveda_cls(root=root, split="val", scene=scenes, transforms=eval_transform, download=False)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=_pad_and_stack_batch,
    )
    return train_dataloader, val_dataloader
