import argparse
import shutil
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

from bayesvlm.hessians import load_hessians, compute_covariances, optimize_prior_precision
from bayesvlm.hf.modeling_bayesvlm_clip import (
    BayesVLMModel,
    BayesVLMTextModel,
    BayesVLMVisionModel,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "bayesvlm" / "hf" / "modeling_bayesvlm_clip.py"
        if candidate.exists():
            return parent
    raise FileNotFoundError("Could not locate bayesvlm/hf/modeling_bayesvlm_clip.py")


def _copy_code(dst_dir: Path) -> None:
    root = _repo_root()
    src = root / "bayesvlm" / "hf" / "modeling_bayesvlm_clip.py"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "modeling_bayesvlm_clip.py")


def _set_auto_map(config, class_name: str) -> None:
    config.auto_map = {
        "AutoModel": f"modeling_bayesvlm_clip.{class_name}",
        "AutoProcessor": "transformers.CLIPProcessor",
    }
    config.architectures = [class_name]


def _save_text_and_vision(
    base: CLIPModel,
    cov_img,
    cov_txt,
    output_dir: Path,
    safe_serialization: bool,
) -> None:
    text_dir = output_dir / "text"
    vision_dir = output_dir / "vision"

    text_config = base.text_model.config
    text_config.projection_dim = base.config.projection_dim
    text_model = BayesVLMTextModel(text_config)
    text_model.text_model.load_state_dict(base.text_model.state_dict())
    text_model.text_projection.load_state_dict(base.text_projection.state_dict())
    text_model.set_covariance(cov_txt.A_inv, cov_txt.B_inv)
    _set_auto_map(text_model.config, "BayesVLMTextModel")
    text_model.save_pretrained(text_dir, safe_serialization=safe_serialization)
    _copy_code(text_dir)

    vision_config = base.vision_model.config
    vision_config.projection_dim = base.config.projection_dim
    vision_model = BayesVLMVisionModel(vision_config)
    vision_model.vision_model.load_state_dict(base.vision_model.state_dict())
    vision_model.visual_projection.load_state_dict(base.visual_projection.state_dict())
    vision_model.set_covariance(cov_img.A_inv, cov_img.B_inv)
    _set_auto_map(vision_model.config, "BayesVLMVisionModel")
    vision_model.save_pretrained(vision_dir, safe_serialization=safe_serialization)
    _copy_code(vision_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a BayesVLM HF-compatible CLIP model with Hessian covariances."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HF model id (e.g., laion/CLIP-ViT-B-32-laion2B-s34B-b79K)",
    )
    parser.add_argument(
        "--hessian-dir",
        required=True,
        help="Path to Hessian/covariance folder (A_*, B_* and prior_precision_analytic.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the HF model.",
    )
    parser.add_argument(
        "--no-text-vision",
        action="store_true",
        help="Skip saving standalone text/vision encoder subfolders.",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save weights in safetensors format.",
    )
    parser.add_argument(
        "--n-img",
        type=float,
        default=None,
        help="Override pseudo data count for image covariance (n_img).",
    )
    parser.add_argument(
        "--n-txt",
        type=float,
        default=None,
        help="Override pseudo data count for text covariance (n_txt).",
    )
    parser.add_argument(
        "--no-optimize-prior",
        action="store_true",
        help="Skip prior precision optimization and use stored lambdas.",
    )
    parser.add_argument(
        "--prior-lambda-init",
        type=float,
        default=1000.0,
        help="Initial lambda value for prior precision optimization.",
    )
    parser.add_argument(
        "--prior-lr",
        type=float,
        default=1e-2,
        help="Learning rate for prior precision optimization.",
    )
    parser.add_argument(
        "--prior-steps",
        type=int,
        default=1000,
        help="Number of steps for prior precision optimization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for prior precision optimization (cpu or cuda).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = CLIPModel.from_pretrained(args.base_model)

    a_img, b_img, info = load_hessians(args.hessian_dir, "img", return_info=True)
    a_txt, b_txt, _ = load_hessians(args.hessian_dir, "txt", return_info=True)

    if args.n_img is not None:
        info["n_img"] = args.n_img
    if args.n_txt is not None:
        info["n_txt"] = args.n_txt

    if not args.no_optimize_prior:
        device = args.device
        vision_projection = torch.nn.Linear(
            base.visual_projection.in_features,
            base.visual_projection.out_features,
            bias=base.visual_projection.bias is not None,
        )
        vision_projection.load_state_dict(base.visual_projection.state_dict())
        vision_projection = vision_projection.to(device)

        text_projection = torch.nn.Linear(
            base.text_projection.in_features,
            base.text_projection.out_features,
            bias=base.text_projection.bias is not None,
        )
        text_projection.load_state_dict(base.text_projection.state_dict())
        text_projection = text_projection.to(device)

        info["lambda_img"] = optimize_prior_precision(
            vision_projection,
            A=a_img,
            B=b_img,
            lmbda_init=args.prior_lambda_init,
            n=info["n_img"],
            lr=args.prior_lr,
            num_steps=args.prior_steps,
            device=device,
            verbose=True,
        ).item()

        info["lambda_txt"] = optimize_prior_precision(
            text_projection,
            A=a_txt,
            B=b_txt,
            lmbda_init=args.prior_lambda_init,
            n=info["n_txt"],
            lr=args.prior_lr,
            num_steps=args.prior_steps,
            device=device,
            verbose=True,
        ).item()

    cov_img, cov_txt = compute_covariances(a_img, b_img, a_txt, b_txt, info)

    model = BayesVLMModel(base.config)
    model.load_state_dict(base.state_dict(), strict=False)
    model.set_covariances(
        image_a_inv=cov_img.A_inv,
        image_b_inv=cov_img.B_inv,
        text_a_inv=cov_txt.A_inv,
        text_b_inv=cov_txt.B_inv,
    )
    _set_auto_map(model.config, "BayesVLMModel")

    model.save_pretrained(output_dir, safe_serialization=args.safe_serialization)
    _copy_code(output_dir)

    processor = CLIPProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(output_dir)

    if not args.no_text_vision:
        _save_text_and_vision(base, cov_img, cov_txt, output_dir, args.safe_serialization)


if __name__ == "__main__":
    main()
