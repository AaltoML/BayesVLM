import argparse
from typing import Tuple

import torch
import torch.distributions as dists
from torchmetrics.classification import MulticlassCalibrationError

from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.hessians import load_hessians, compute_covariances, optimize_prior_precision
from bayesvlm.precompute import precompute_image_features, precompute_text_features, make_predictions
from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.utils import get_model_type_and_size, get_image_size, get_transform, load_model

SUPPORTED_DATASETS = [
    "flowers102", "food101", "cifar10", "cifar100", "imagenet-r", "ucf101", "sun397",
]

    
def evaluate_prediction(prediction: torch.Tensor, label: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm='l1')
    one_hot_pred = prediction.argmax(1)
    acc = (one_hot_pred == label).float().cpu().numpy()
    nlpd = -dists.Categorical(prediction).log_prob(label).cpu().numpy()
    ece = ece_metric(prediction, label).item()
    return acc, nlpd, ece

def main(
    dataset: str,
    hessian_dir: str,
    model_str: str = "clip-base",
    pseudo_data_count: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
):
    if model_str not in MODEL_NAME_MAP:
        raise ValueError(f"Invalid model name: {model_str}, must be one of {MODEL_NAME_MAP.keys()}")
    
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}, must be one of {SUPPORTED_DATASETS}")
    
    model_type, model_size = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    f = DataModuleFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=transform,
        test_transform=transform,
        shuffle_train=True,
    )
    dm = f.create(dataset)
    dm.setup()

    image_encoder, text_encoder, vlm = load_model(model_str, device)

    A_img, B_img = load_hessians(hessian_dir, tag='img', return_info=False)
    A_txt, B_txt = load_hessians(hessian_dir, tag='txt', return_info=False)

    info = {
        'n_img': pseudo_data_count,
        'n_txt': pseudo_data_count,
    }

    print("[1] Optimizing prior precision...")
    info['lambda_img'] = optimize_prior_precision(
        image_encoder.vision_projection,
        A=A_img,
        B=B_img,
        lmbda_init=300,
        n=info['n_img'],
        lr=1e-2,
        num_steps=1000,
        device=device,
        verbose=True,
    ).item()

    info['lambda_txt'] = optimize_prior_precision(
        text_encoder.text_projection,
        A=A_txt,
        B=B_txt,
        lmbda_init=300,
        n=info['n_txt'],
        lr=1e-2,
        num_steps=1000,
        device=device,
        verbose=True,
    ).item()
    print("\tn_img:", info['n_img'])
    print("\tn_txt:", info['n_txt'])
    print("\tlambda_img:", info['lambda_img'])
    print("\tlambda_txt:", info['lambda_txt'])

    cov_img, cov_txt = compute_covariances(A_img, B_img, A_txt, B_txt, info)
    vlm.set_covariances(cov_img, cov_txt)

    print("[2] Precomputing features...")
    with torch.no_grad():
        image_outputs_test, image_class_ids_test, image_ids_test = precompute_image_features(
            image_encoder=image_encoder,
            loader=dm.test_dataloader(),
        )

        label_outputs = precompute_text_features(
            text_encoder=text_encoder,
            class_prompts=dm.class_prompts,
            batch_size=batch_size,
        )

        print("[3] Making predictions...")
        prob_logits_test = make_predictions(
            clip=vlm,
            image_outputs=image_outputs_test,
            text_outputs=label_outputs,
            batch_size=batch_size,
            device=device,
        )

    # probit approximation
    kappa = 1 / torch.sqrt(1. + torch.pi / 8 * prob_logits_test.var)
    pred = torch.softmax(kappa * prob_logits_test.mean, dim=-1)

    print("[4] Evaluate model ...")
    acc, nlpd, ece = evaluate_prediction(pred, image_class_ids_test, num_classes=len(dm.class_prompts))

    print(f"Zero shot CLIP on {dataset}")
    print(f"ACC: {acc.mean()}, {acc.std()}")
    print(f"NLPD: {nlpd.mean()}, {nlpd.std()}")
    print(f"ECE: {ece}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='food101', help="The dataset to use")
    parser.add_argument("--hessian_dir", type=str, default='hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K', help="The directory containing the hessian files")
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,
        model_str=args.model,
        pseudo_data_count=args.pseudo_data_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
