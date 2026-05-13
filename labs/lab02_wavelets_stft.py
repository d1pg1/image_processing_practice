from __future__ import annotations

"""Lab 02 (skeleton): Wavelets (Haar) + STFT bridge."""

import argparse
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal as _scipy_signal

ThresholdMode = Literal["soft", "hard"]


def haar_dwt1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-level 1D Haar DWT.

    For odd-length inputs, pad one sample (edge policy).

    Args:
        x: 1D numeric signal.

    Returns:
        (approx, detail): each length ~N/2.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) % 2 != 0:
        x = np.pad(x, (0, 1), mode="edge")
    s = np.sqrt(2.0)
    approx = (x[0::2] + x[1::2]) / s
    detail = (x[0::2] - x[1::2]) / s
    return approx, detail


def haar_idwt1(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """
    Invert one-level 1D Haar DWT.

    Args:
        approx: Approximation coefficients.
        detail: Detail coefficients.

    Returns:
        Reconstructed signal.
    """
    approx = np.asarray(approx, dtype=np.float64)
    detail = np.asarray(detail, dtype=np.float64)
    s = np.sqrt(2.0)
    out = np.empty(len(approx) * 2, dtype=np.float64)
    out[0::2] = (approx + detail) / s
    out[1::2] = (approx - detail) / s
    return out


def haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute one-level 2D separable Haar DWT for grayscale images.

    Args:
        image: 2D grayscale image.

    Returns:
        LL, (LH, HL, HH).
    """
    image = np.asarray(image, dtype=np.float64)
    
    row_a = np.stack([haar_dwt1(row)[0] for row in image])
    row_d = np.stack([haar_dwt1(row)[1] for row in image])
    
    LL = np.stack([haar_dwt1(row_a[:, c])[0] for c in range(row_a.shape[1])], axis=1)
    LH = np.stack([haar_dwt1(row_a[:, c])[1] for c in range(row_a.shape[1])], axis=1)
    HL = np.stack([haar_dwt1(row_d[:, c])[0] for c in range(row_d.shape[1])], axis=1)
    HH = np.stack([haar_dwt1(row_d[:, c])[1] for c in range(row_d.shape[1])], axis=1)
    return LL, (LH, HL, HH)


def haar_idwt2(LL: np.ndarray, bands: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Invert one-level 2D Haar DWT.

    Args:
        LL: Low-low sub-band.
        bands: Tuple `(LH, HL, HH)`.

    Returns:
        Reconstructed image. If the original had odd dimensions the IDWT produces
        the padded (even) size; callers should crop to the desired shape.
    """
    LH, HL, HH = bands
    
    row_a = np.stack([haar_idwt1(LL[:, c], LH[:, c]) for c in range(LL.shape[1])], axis=1)
    row_d = np.stack([haar_idwt1(HL[:, c], HH[:, c]) for c in range(HL.shape[1])], axis=1)
    
    rows = row_a.shape[0]
    out = np.stack([haar_idwt1(row_a[r], row_d[r]) for r in range(rows)])
    return out


def wavelet_threshold(coeffs: Any, threshold: float, mode: ThresholdMode = "soft") -> Any:
    """
    Apply thresholding to coefficient arrays.

    Args:
        coeffs: Array or nested tuples/lists of arrays.
        threshold: Non-negative threshold value.
        mode: `"soft"` or `"hard"`.

    Returns:
        Thresholded coefficients with same structure.
    """
    if isinstance(coeffs, np.ndarray):
        if mode == "hard":
            return np.where(np.abs(coeffs) > threshold, coeffs, 0.0)
        mag = np.maximum(np.abs(coeffs) - threshold, 0.0)
        return np.sign(coeffs) * mag
    if isinstance(coeffs, (tuple, list)):
        thresholded = [wavelet_threshold(c, threshold, mode) for c in coeffs]
        return type(coeffs)(thresholded)
    return coeffs


def wavelet_denoise(image: np.ndarray, levels: int, threshold: float, mode: ThresholdMode = "soft") -> np.ndarray:
    """
    Denoise image via multi-level Haar thresholding.

    Args:
        image: 2D grayscale image.
        levels: Number of decomposition levels.
        threshold: Coefficient threshold.
        mode: `"soft"` or `"hard"`.

    Returns:
        Denoised image with deterministic behavior.
    """
    ll = np.asarray(image, dtype=np.float64)
    ll_shapes: list[tuple[int, ...]] = []
    decomps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for _ in range(levels):
        ll_shapes.append(ll.shape)
        ll, bands = haar_dwt2(ll)
        decomps.append(bands)

    decomps = [wavelet_threshold(b, threshold, mode) for b in decomps]

    for bands, shape in zip(reversed(decomps), reversed(ll_shapes)):
        ll = haar_idwt2(ll, bands)
        ll = ll[: shape[0], : shape[1]]

    return np.clip(ll, 0.0, 255.0).astype(np.float32)


def stft1(
    x: np.ndarray,
    fs_hz: float,
    frame_len: int,
    hop_len: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for 1D signal using SciPy.

    Returns:
        `(freqs_hz, times_s, Zxx)` where `Zxx` is complex.
    """
    freqs, times, Zxx = _scipy_signal.stft(
        x,
        fs=fs_hz,
        window=window,
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
    )
    return freqs, times, Zxx


def spectrogram_magnitude(Zxx: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Convert STFT matrix to magnitude spectrogram.

    Args:
        Zxx: Complex STFT matrix.
        log_scale: If True, return `log(1 + magnitude)`.

    Returns:
        Non-negative finite magnitude matrix.
    """
    mag = np.abs(Zxx)
    if log_scale:
        mag = np.log1p(mag)
    return mag.astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Min-max normalize an array to `[0,255]` for visualization."""
    arr = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) * (255.0 / (mx - mn)), 0.0, 255.0).astype(np.uint8)


def main() -> int:
    """
    Lab 02 demo (skeleton).

    Expected behavior after implementation:
    - wavelet denoising demo on image from `./imgs/`
    - LL/LH/HL/HH band visualization
    - STFT spectrogram demo on synthetic chirp signal
    - save outputs to `./out/lab02/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 02 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab02", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    # --- Wavelet demo ---
    try:
        rng = np.random.default_rng(0)
        noisy = img.astype(np.float32) + rng.normal(0.0, 20.0, size=img.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)
        den = wavelet_denoise(noisy, levels=2, threshold=20.0, mode="soft")

        ll, (lh, hl, hh) = haar_dwt2(img.astype(np.float32))

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Original", img),
                ("Noisy (Gaussian)", noisy),
                ("Wavelet denoised", den),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_denoise.png")

        plt.figure(figsize=(10, 8))
        for i, (title, band) in enumerate(
            [
                ("LL", ll),
                ("LH", lh),
                ("HL", hl),
                ("HH", hh),
            ],
            start=1,
        ):
            plt.subplot(2, 2, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(band), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_bands.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- STFT bridge demo ---
    try:
        fs = 400.0
        duration_s = 2.0
        t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
        f0, f1 = 15.0, 120.0
        k = (f1 - f0) / duration_s
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.sin(phase)

        freqs, times, zxx = stft1(x, fs_hz=fs, frame_len=128, hop_len=32, window="hann")
        mag = spectrogram_magnitude(zxx, log_scale=True)

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(times, freqs, mag, shading="gouraud")
        plt.title("STFT Spectrogram (log-magnitude)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="log(1 + |Zxx|)")
        save_figure(out_dir / "stft_spectrogram.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 02 demo is incomplete. Implement the TODO functions in labs/lab02_wavelets_stft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
