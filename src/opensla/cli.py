"""Run OpenSLA inference on a user-provided preprocessed batch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/template.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--domain", choices=("clinical", "or", "cgm"))
    parser.add_argument("--device")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("checkpoint", "input", "output", "domain", "device", "max_new_tokens"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from .inference import OpenSLA, load_prepared_batch
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output path.")
    batch = load_prepared_batch(config["input"], domain=config["domain"])
    model = OpenSLA.from_checkpoint(
        config["checkpoint"], domain=config["domain"], device=config.get("device", "cuda"),
        model_id=config.get("model_id"), dino_checkpoint=config.get("dino_checkpoint"),
        numeric_config=config.get("numeric_config"), action_group_types=config.get("action_group_types"),
    )
    results = model.predict(
        batch["input_text"], batch["sensors"],
        max_new_tokens=config.get("max_new_tokens", 512),
        necessity_threshold=config.get("necessity_threshold", 0.5),
        category_threshold=config.get("category_threshold", 0.5),
        labels_per_category=config.get("labels_per_category", 3),
        max_categories=config.get("max_categories", 5),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for sample_id, result in zip(batch["sample_ids"], results):
            handle.write(json.dumps({"sample_id": sample_id, **result}) + "\n")
    print(f"Wrote {len(results)} predictions to {output}")


if __name__ == "__main__":
    main()
