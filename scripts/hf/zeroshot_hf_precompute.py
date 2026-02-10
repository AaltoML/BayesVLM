import argparse
from typing import Tuple

import torch
import torch.distributions as dists
from torchmetrics.classification import MulticlassCalibrationError
from transformers import AutoConfig, AutoModel, AutoTokenizer

from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.data.common import default_transform

SUPPORTED_DATASETS = [
    "flowers102", "food101", "cifar10", "cifar100", "imagenet-r", "ucf101", "sun397",
]


def evaluate_prediction(prediction: torch.Tensor, label: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm="l1")
    one_hot_pred = prediction.argmax(1)
    acc = (one_hot_pred == label).float().cpu().numpy()
    nlpd = -dists.Categorical(prediction).log_prob(label).cpu().numpy()
    ece = ece_metric(prediction, label).item()
    return acc, nlpd, ece


def _get_image_size(config) -> int:
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "image_size"):
        return int(config.vision_config.image_size)
    if hasattr(config, "image_size"):
        return int(config.image_size)
    return 224


def main(
    dataset: str,
    model: str,
    batch_size: int,
    num_workers: int,
    device: str,
):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}, must be one of {SUPPORTED_DATASETS}")

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    image_size = _get_image_size(config)
    transform = default_transform(image_size)

    dm_factory = DataModuleFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=transform,
        test_transform=transform,
        shuffle_train=True,
    )
    dm = dm_factory.create(dataset)
    dm.setup()

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModel.from_pretrained(model, trust_remote_code=True)
    model = model.to(device).eval()

    class_prompts = dm.class_prompts
    text_inputs = tokenizer(
        class_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            return_dict=True,
        )
        text_pooled = text_outputs[1]
        text_embeds = model.text_projection(text_pooled)

    logits_means = []
    logits_vars = []
    labels = []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            pixel_values = batch["image"].to(device)
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            image_pooled = vision_outputs[1]
            image_embeds = model.visual_projection(image_pooled)

            logits_mean, logits_var = model._expected_logits_and_var(
                image_embeds,
                text_embeds,
                image_pooled,
                text_pooled,
            )
            if logits_var is None:
                logits_var = torch.zeros_like(logits_mean)

            logits_means.append(logits_mean.cpu())
            logits_vars.append(logits_var.cpu())
            labels.append(batch["class_id"].cpu())

    logits_mean = torch.cat(logits_means, dim=0)
    logits_var = torch.cat(logits_vars, dim=0)
    labels = torch.cat(labels, dim=0)

    kappa = 1 / torch.sqrt(1.0 + torch.pi / 8 * logits_var)
    pred = torch.softmax(kappa * logits_mean, dim=-1)

    acc, nlpd, ece = evaluate_prediction(pred, labels, num_classes=len(class_prompts))

    print(f"Zero shot CLIP on {dataset}")
    print(f"ACC: {acc.mean()}, {acc.std()}")
    print(f"NLPD: {nlpd.mean()}, {nlpd.std()}")
    print(f"ECE: {ece}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="food101", help="The dataset to use")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
