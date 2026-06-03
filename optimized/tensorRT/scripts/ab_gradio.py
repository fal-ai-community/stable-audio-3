"""Six-way comparison gradio for SA3 medium DiT:
  SAME-L decoder side:
    - PT eager + SAME-L  (PyTorch FP32 DiT + PyTorch FP32 SAME-L)
    - TRT canon          (FP16-mixed DiT + BF16/Triton-SWA SAME-L)
    - TRT FP32           (pure-FP32 DiT + pure-FP32 SAME-L)
  SAME-S decoder side:
    - PT eager + SAME-S  (PyTorch FP32 DiT + PyTorch FP32 SAME-S)
    - TRT canon          (FP16-mixed DiT + BF16 SAME-S)
    - TRT FP32           (pure-FP32 DiT + FP32 SAME-S)

Same prompt + seed + seconds + steps → all six pipelines run → six columns
of audio + spectrogram + per-stage timing + a 6×6 cos-sim matrix.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Backends loaded lazily on first generate (or eagerly via --preload).
PT_EAGER         = None  # type: ignore  (single instance; holds both SAME-L and SAME-S decoders)
TRT_SAMEL_CANON  = None  # type: ignore
TRT_SAMEL_FP32   = None  # type: ignore
TRT_SAMES_CANON  = None  # type: ignore
TRT_SAMES_FP32   = None  # type: ignore


SAMPLE_RATE = 44100
OUTPUT_DIR = SCRIPTS_DIR.parent / "output" / "ab_gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _stats(pcm: np.ndarray) -> dict:
    p = pcm.astype(np.float64)
    return {
        "rms":  int(np.sqrt((p ** 2).mean())),
        "peak": int(np.abs(pcm).max()),
        "clip_pct": float((np.abs(pcm) >= 32700).mean() * 100),
    }


def _save_wav(pcm: np.ndarray, path: str) -> None:
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.astype(np.int16).tobytes())


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    af = a[:n].astype(np.float64).flatten()
    bf = b[:n].astype(np.float64).flatten()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float((af @ bf) / denom)


def _ensure_pt():
    global PT_EAGER
    if PT_EAGER is None:
        from pt_inference import PTInference
        PT_EAGER = PTInference(load_samel=True, load_sames=True)


def run_pt_samel(prompt: str, seconds: float, steps: int, seed: int):
    _ensure_pt()
    return PT_EAGER.generate(prompt, seconds=seconds, steps=steps, seed=seed, decoder="same-l")


def run_pt_sames(prompt: str, seconds: float, steps: int, seed: int):
    _ensure_pt()
    return PT_EAGER.generate(prompt, seconds=seconds, steps=steps, seed=seed, decoder="same-s")


def _build_sa3_with(dit_engine_name: str, decoder_name: str, dec_engine_name: str,
                     seconds: float, steps: int):
    """Construct SA3Inference (CUDA graph mode) with the named engines."""
    import sa3_trt_core as canon
    from sa3_trt import SA3Inference
    arch_dir = Path(canon.ARCH_DIR)
    dit_path = arch_dir / "sa3-m"      / dit_engine_name
    dec_path = arch_dir / decoder_name / dec_engine_name
    if not dit_path.exists() or not dec_path.exists():
        raise FileNotFoundError(f"missing engine(s): {dit_path}  /  {dec_path}")
    orig_dit = canon.DIT_CHOICES["medium"]["engine"]
    orig_dec = canon.DECODER_PATHS[decoder_name]
    try:
        canon.DIT_CHOICES["medium"]["engine"] = dit_path
        canon.DECODER_PATHS[decoder_name] = dec_path
        import math
        T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / 4096))
        return SA3Inference(
            dit="medium", decoder=decoder_name,
            default_T_lat=T_lat, default_steps=steps,
            default_seconds=seconds, with_encoder=False, quiet=True,
        )
    finally:
        canon.DIT_CHOICES["medium"]["engine"] = orig_dit
        canon.DECODER_PATHS[decoder_name] = orig_dec


def run_trt_samel_canon(prompt: str, seconds: float, steps: int, seed: int):
    global TRT_SAMEL_CANON
    if TRT_SAMEL_CANON is None:
        TRT_SAMEL_CANON = _build_sa3_with("dit_fp16mixed.trt", "same-l", "dec_dynamic_triton_swa.trt", seconds, steps)
    return TRT_SAMEL_CANON.generate(prompt, seconds=seconds, steps=steps, seed=seed)


def run_trt_samel_fp32(prompt: str, seconds: float, steps: int, seed: int):
    global TRT_SAMEL_FP32
    if TRT_SAMEL_FP32 is None:
        TRT_SAMEL_FP32 = _build_sa3_with("dit_fp32.trt", "same-l", "dec_dynamic_fp32.trt", seconds, steps)
    return TRT_SAMEL_FP32.generate(prompt, seconds=seconds, steps=steps, seed=seed)


def run_trt_sames_canon(prompt: str, seconds: float, steps: int, seed: int):
    global TRT_SAMES_CANON
    if TRT_SAMES_CANON is None:
        TRT_SAMES_CANON = _build_sa3_with("dit_fp16mixed.trt", "same-s", "dec_dynamic_bf16.trt", seconds, steps)
    return TRT_SAMES_CANON.generate(prompt, seconds=seconds, steps=steps, seed=seed)


def run_trt_sames_fp32(prompt: str, seconds: float, steps: int, seed: int):
    global TRT_SAMES_FP32
    if TRT_SAMES_FP32 is None:
        TRT_SAMES_FP32 = _build_sa3_with("dit_fp32.trt", "same-s", "dec_dynamic_fp32.trt", seconds, steps)
    return TRT_SAMES_FP32.generate(prompt, seconds=seconds, steps=steps, seed=seed)


PANEL_BACKENDS = [
    # (short label, title, runner)  — order is the column order in the UI
    ("pt_samel",    "PT eager + SAME-L (FP32)",                            "run_pt_samel"),
    ("samel_canon", "TRT SAME-L canon (FP16-mixed DiT + BF16/Triton SAME-L)", "run_trt_samel_canon"),
    ("samel_fp32",  "TRT SAME-L FP32 (FP32 DiT + FP32 SAME-L)",            "run_trt_samel_fp32"),
    ("pt_sames",    "PT eager + SAME-S (FP32)",                            "run_pt_sames"),
    ("sames_canon", "TRT SAME-S canon (FP16-mixed DiT + BF16 SAME-S)",     "run_trt_sames_canon"),
    ("sames_fp32",  "TRT SAME-S FP32 (FP32 DiT + FP32 SAME-S)",            "run_trt_sames_fp32"),
]
# Short labels for the cos-sim table column/row headers
PANEL_SHORT = ["PT-L", "TRT-L can", "TRT-L fp32", "PT-S", "TRT-S can", "TRT-S fp32"]


def generate_all(prompt: str, seconds: float, steps: int, seed: int):
    """Run all six backends; return artifacts for the six columns + cos-sim matrix."""
    if not prompt.strip():
        prompt = "music"
    seconds = float(seconds); steps = int(steps); seed = int(seed)

    # Resolve runner names to actual functions in this module
    g = globals()
    backends = [(short, title, g[runner_name]) for short, title, runner_name in PANEL_BACKENDS]

    results = []
    for short, title, runner in backends:
        t0 = time.time()
        pcm, timing = runner(prompt, seconds, steps, seed)
        wall_ms = (time.time() - t0) * 1000
        results.append({"short": short, "title": title, "pcm": pcm,
                         "timing": timing, "wall_ms": wall_ms})

    # Save WAVs + spectrograms
    stamp = int(time.time() * 1000)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:32]
    from spec import render_spectrogram_png
    for r in results:
        r["wav_path"]  = str(OUTPUT_DIR / f"{stamp}_{r['short']}_{safe_prompt}_s{seed}.wav")
        r["spec_path"] = r["wav_path"].replace(".wav", "_spec.png")
        _save_wav(r["pcm"], r["wav_path"])
        with open(r["spec_path"], "wb") as f:
            f.write(render_spectrogram_png(r["pcm"], sample_rate=SAMPLE_RATE, width=900, height=200))
        r["stats"] = _stats(r["pcm"])

    def fmt_info(r):
        rt = (seconds * 1000.0) / r["wall_ms"] if r["wall_ms"] > 0 else 0.0
        T_lat = r["timing"].get("T_lat", "?")
        lines = [
            f"<b>{r['title']}</b>",
            f"T_lat={T_lat}  samples={r['timing'].get('samples', '?')}",
            f"PCM: RMS={r['stats']['rms']}  peak={r['stats']['peak']}  clip={r['stats']['clip_pct']:.2f}%",
            f"wall: {r['wall_ms']:.0f} ms  ({rt:.1f}× real-time)",
        ]
        for k in ("t5_ms", "sampling_ms", "decode_ms", "graph_build_ms"):
            if k in r["timing"] and r["timing"][k] > 0.1:
                lines.append(f"{k}: {r['timing'][k]:.0f} ms")
        return "<br>".join(lines)

    # --- Cos-sim matrix (6×6, upper triangular shown; diagonal = 1.00) ---
    def colorize(cos):
        return "#5ad" if cos > 0.99 else ("#7fa" if cos > 0.95 else ("#fa3" if cos > 0.5 else "#f55"))

    n = len(results)
    rows_html = []
    # header row
    header_cells = "<th></th>" + "".join(f"<th>{lbl}</th>" for lbl in PANEL_SHORT)
    rows_html.append(f"<tr>{header_cells}</tr>")
    for i in range(n):
        cells = [f"<th style='text-align:right;padding-right:6px'>{PANEL_SHORT[i]}</th>"]
        for j in range(n):
            if j < i:
                cells.append("<td style='color:#444'>·</td>")  # blank below-diagonal
            elif j == i:
                cells.append("<td style='color:#666'>1.0000</td>")
            else:
                c = _cos_sim(results[i]["pcm"], results[j]["pcm"])
                cells.append(f"<td style='color:{colorize(c)};font-weight:bold'>{c:.4f}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table_html = (
        "<div id='cos-table'><table>"
        "<thead>" + rows_html[0] + "</thead>"
        "<tbody>" + "".join(rows_html[1:]) + "</tbody></table></div>"
    )
    # All backends now use the same natural T_lat — pull from any one
    T_lat_used = results[0]["timing"].get("T_lat", "?")
    header = (f"<div style='text-align:center'>"
              f"<div style='font-size:0.9em; color:#888; margin-bottom:6px'>"
              f"prompt=<code>{prompt}</code>  ·  seed={seed}  ·  {seconds:.1f}s  ·  {steps} steps"
              f"  ·  T_lat=<b>{T_lat_used}</b>"
              f"</div>{table_html}</div>")

    out = [header]
    for r in results:
        out += [r["wav_path"], r["spec_path"], fmt_info(r)]
    return tuple(out)


def build_ui():
    import gradio as gr

    # Must match PANEL_BACKENDS order. Six entries.
    panel_titles = [
        "PT eager + SAME-L (FP32)",
        "TRT SAME-L canon (FP16-mixed)",
        "TRT SAME-L FP32",
        "PT eager + SAME-S (FP32)",
        "TRT SAME-S canon (BF16)",
        "TRT SAME-S FP32",
    ]

    with gr.Blocks(title="SA3 6-way: PT eager × SAME-L/S × TRT canon/FP32") as demo:
        gr.Markdown(
            "# SA3 — six-way comparison\n"
            "Same prompt + seed + duration + steps, all six backends. **medium DiT** + "
            "SAME-L (PT, TRT canon, TRT FP32) and SAME-S (PT, TRT canon, TRT FP32).\n\n"
            "> ⚠️ **First-run / duration-change is slow.** Each TRT backend builds a CUDA "
            "graph keyed by `(T_lat, steps)`, which adds 2–5s on the first call at a given "
            "duration. Generate **2–3 times** at the same duration to see the real "
            "warmed-up inference timing. Changing the **seconds** value triggers a new graph "
            "build on each TRT backend. DiT runs at the natural T_lat = ⌈seconds·44100 / 4096⌉; "
            "SAME-S decoder accepts any L (no even-bump)."
        )
        with gr.Row():
            prompt = gr.Textbox(label="prompt", value="Death Metal", lines=1, scale=4)
            seed = gr.Number(label="seed", value=1, precision=0, scale=1)
            seconds = gr.Number(label="seconds", value=30.0, precision=1, scale=1)
            steps = gr.Number(label="steps", value=8, precision=0, scale=1)
            generate_btn = gr.Button("Generate", variant="primary", scale=2)

        header = gr.HTML()

        out_widgets = []
        with gr.Row():
            for title in panel_titles:
                with gr.Column():
                    gr.Markdown(f"### {title}")
                    a = gr.Audio(label="audio", type="filepath", autoplay=False)
                    s = gr.Image(label="spectrogram", show_label=False)
                    i = gr.HTML()
                    out_widgets += [a, s, i]

        generate_btn.click(
            generate_all,
            inputs=[prompt, seconds, steps, seed],
            outputs=[header] + out_widgets,
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true", help="gradio.live tunnel")
    ap.add_argument("--preload", action="store_true",
                    help="Eagerly load all three backends at startup (default is lazy on first generate)")
    args = ap.parse_args()

    if args.preload:
        # Use seconds=15 → T_lat=162 (bumped to even), inside the engines' [32, 4096] profile range.
        # PyTorch eager loads the DiT + both SAME-{L,S} decoders on the first call;
        # subsequent PT calls reuse the cached module.
        for label, runner in [
            ("PT eager + SAME-L", run_pt_samel),
            ("TRT SAME-L canon",  run_trt_samel_canon),
            ("TRT SAME-L FP32",   run_trt_samel_fp32),
            ("PT eager + SAME-S", run_pt_sames),
            ("TRT SAME-S canon",  run_trt_sames_canon),
            ("TRT SAME-S FP32",   run_trt_sames_fp32),
        ]:
            print(f">> preloading {label}…", flush=True)
            runner("warmup", seconds=15.0, steps=2, seed=0)
        print(">> all 6 loaded", flush=True)

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(OUTPUT_DIR)],
        css="""
            #cos-table table { margin: 0 auto; border-collapse: collapse;
                                font-family: monospace; font-size: 0.95em; }
            #cos-table th { color: #fff !important; padding: 4px 8px; background: #2a2a2a; }
            #cos-table td { padding: 4px 8px; }
        """,
    )


if __name__ == "__main__":
    main()
