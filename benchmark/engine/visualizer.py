from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont


class BenchmarkVisualizer:
    """
    Generates reproducible visual comparison grids and research summaries.
    Compares: Input LR | Bicubic Baseline | Model Output | Ground Truth Reference.
    """

    @classmethod
    def create_side_by_side_grid(
        cls,
        lr_image: Image.Image,
        bicubic_image: Image.Image,
        model_image: Image.Image,
        ground_truth_image: Image.Image,
        labels: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Creates a 4-panel horizontal comparison image:
          [Input LR (upscaled)] | [Bicubic Baseline] | [Model Prediction] | [Ground Truth HR]
        """
        target_size = ground_truth_image.size
        w, h = target_size

        # Resize all to match HR dimensions for visual alignment
        p1 = lr_image.resize((w, h), resample=Image.NEAREST).convert("RGB")
        p2 = bicubic_image.resize((w, h), resample=Image.BICUBIC).convert("RGB")
        p3 = model_image.resize((w, h), resample=Image.LANCZOS).convert("RGB")
        p4 = ground_truth_image.resize((w, h), resample=Image.LANCZOS).convert("RGB")

        panel_w = w
        panel_h = h
        header_h = 36
        grid_w = panel_w * 4 + 15
        grid_h = panel_h + header_h

        grid = Image.new("RGB", (grid_w, grid_h), color=(20, 24, 33))
        draw = ImageDraw.Draw(grid)

        default_labels = ["Input LR (Nearest)", "Bicubic Baseline", "Model Prediction", "Ground Truth HR"]
        active_labels = labels or default_labels

        panels = [p1, p2, p3, p4]
        for i, (panel, lbl) in enumerate(zip(panels, active_labels)):
            x_offset = i * (panel_w + 5)
            # Paste header text
            draw.text((x_offset + 10, 10), lbl, fill=(230, 235, 245))
            # Paste image
            grid.paste(panel, (x_offset, header_h))

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(output_path)

        return grid

    @classmethod
    def generate_markdown_summary_table(cls, aggregation: Dict[str, Any]) -> str:
        """Generates a GitHub-flavored markdown table summarizing the benchmark run."""
        lines = [
            "### Benchmark Matrix Summary",
            "",
            "| Model & Configuration | N | PSNR Mean ± Std (dB) | SSIM Mean ± Std | Laplacian Sharpness | Mean Inference (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        # Group by Model x Scale
        by_scale = aggregation.get("by_model_scale", {})
        for config_name, stats in sorted(by_scale.items()):
            n = stats["sample_count_N"]
            psnr = stats["psnr"]
            ssim = stats["ssim"]
            sharp = stats["sharpness"]
            inf_ms = stats["inference_ms"]

            psnr_str = f"{psnr['mean']} ± {psnr['std']}" if psnr['mean'] is not None else "N/A"
            ssim_str = f"{ssim['mean']} ± {ssim['std']}" if ssim['mean'] is not None else "N/A"
            sharp_str = f"{sharp['mean']}" if sharp['mean'] is not None else "N/A"
            time_str = f"{inf_ms['mean']}" if inf_ms['mean'] is not None else "N/A"

            lines.append(f"| **{config_name}** | {n} | {psnr_str} | {ssim_str} | {sharp_str} | {time_str} |")

        lines.append("")
        return "\n".join(lines)
