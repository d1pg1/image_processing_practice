from __future__ import annotations

"""Lab 01 (skeleton): filtering/convolution + FFT tools (spatial & frequency domain)."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

BorderType = Literal["reflect", "constant", "wrap", "replicate"]

_BORDER_PAD_MODE: dict[str, str] = {
    "reflect": "reflect",
    "constant": "constant",
    "wrap": "wrap",
    "replicate": "edge",
}


def conv2d(image: npt.NDArray[np.generic], kernel: npt.NDArray[np.generic], border: BorderType = "reflect") -> np.ndarray:
    """
    2D convolution for grayscale/color images (spatial-domain linear filtering).

    Args:
        image: `(H,W)` or `(H,W,C)` array (any numeric dtype; computed in `float32`).
        kernel: `(kH,kW)` 2D kernel (any numeric dtype).
        border: `"reflect" | "constant" | "wrap" | "replicate"`.

    Returns:
        `float32` array with the same shape as `image`.
    """
    img = np.asarray(image, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad_mode = _BORDER_PAD_MODE[border]

    def _convolve_channel(ch: np.ndarray) -> np.ndarray:
        if pad_mode == "constant":
            padded = np.pad(ch, ((ph, ph), (pw, pw)), mode=pad_mode, constant_values=0)
        else:
            padded = np.pad(ch, ((ph, ph), (pw, pw)), mode=pad_mode)
        return signal.convolve2d(padded, k, mode="valid")

    if img.ndim == 2:
        return _convolve_channel(img).astype(np.float32)
    channels = [_convolve_channel(img[:, :, c]) for c in range(img.shape[2])]
    return np.stack(channels, axis=2).astype(np.float32)


def make_gaussian_kernel(ksize: int, sigma: float) -> npt.NDArray[np.float32]:
    """
    Create a normalized 2D Gaussian kernel (sum ~ 1).

    Args:
        ksize: Positive odd kernel size.
        sigma: Standard deviation in pixels (> 0).

    Returns:
        `(ksize, ksize)` `float32` kernel.
    """
    ax = np.arange(ksize, dtype=np.float32) - ksize // 2
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    g /= g.sum()
    return np.outer(g, g).astype(np.float32)


def _clip_to_dtype_range(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(x, info.min, info.max)
    return x


def apply_gaussian_blur(image: npt.NDArray[np.generic], ksize: int, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing in the spatial domain (via `conv2d`).

    Args:
        image: `(H,W)` or `(H,W,C)` image.
        ksize: Positive odd kernel size.
        sigma: Standard deviation in pixels.

    Returns:
        Same shape/dtype as input.
    """
    orig_dtype = np.asarray(image).dtype
    k = make_gaussian_kernel(ksize, sigma)
    result = conv2d(image, k)
    return _clip_to_dtype_range(result, orig_dtype).astype(orig_dtype)


def apply_box_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    """
    Box/mean blur using a `(ksize x ksize)` uniform kernel (via `conv2d`).

    Args:
        image: `(H,W)` or `(H,W,C)` image.
        ksize: Positive odd window size.

    Returns:
        Same shape/dtype as input.
    """
    orig_dtype = np.asarray(image).dtype
    k = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    result = conv2d(image, k)
    return _clip_to_dtype_range(result, orig_dtype).astype(orig_dtype)


def apply_median_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    """
    Median filter (best for salt-and-pepper noise).

    Args:
        image: `uint8` image (grayscale or color).
        ksize: Positive odd neighborhood size.

    Returns:
        Same shape/dtype as input.
    """
    return cv2.medianBlur(np.asarray(image), ksize)


def add_salt_pepper_noise(
    image: npt.NDArray[np.generic],
    amount: float,
    salt_vs_pepper: float = 0.5,
    *,
    seed: int = 0,
) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise (deterministic by `seed`).

    Args:
        image: Input image (any numeric dtype).
        amount: Fraction of pixels to corrupt in `[0, 1]`.
        salt_vs_pepper: Probability of "salt" among corrupted pixels.
        seed: RNG seed.

    Returns:
        Noised image with the same shape/dtype.
    """
    img = np.asarray(image).copy()
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    num_corrupt = int(amount * H * W)

    ys = rng.integers(0, H, size=num_corrupt)
    xs = rng.integers(0, W, size=num_corrupt)
    is_salt = rng.random(num_corrupt) < salt_vs_pepper

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        max_val: float | int = info.max
        min_val: float | int = info.min
    else:
        max_val, min_val = 1.0, 0.0

    img[ys[is_salt], xs[is_salt]] = max_val
    img[ys[~is_salt], xs[~is_salt]] = min_val
    return img


def add_gaussian_noise(image: npt.NDArray[np.generic], sigma: float, *, seed: int = 0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise (deterministic by `seed`).

    Args:
        image: Input image (any numeric dtype).
        sigma: Standard deviation (>= 0), in image intensity units.
        seed: RNG seed.

    Returns:
        Noised image with the same shape/dtype.
    """
    orig_dtype = np.asarray(image).dtype
    img = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    result = img + noise
    return _clip_to_dtype_range(result, orig_dtype).astype(orig_dtype)


def sobel_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobel gradients and magnitude (edge strength).

    Returns:
        `(gx, gy, magnitude)` as `float32` arrays of shape `(H, W)`.

    Args:
        image: Input image (converted to grayscale internally).
        ksize: Positive odd Sobel size.
    """
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, magnitude


def laplacian_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> np.ndarray:
    """
    Laplacian edge response (absolute value).

    Args:
        image: Input image (converted to grayscale internally).
        ksize: Positive odd aperture size.

    Returns:
        `float32` array `(H, W)` (non-negative).
    """
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=ksize)
    return np.abs(lap)


def fft2_image(image: npt.NDArray[np.generic]) -> np.ndarray:
    """
    Compute the 2D DFT using OpenCV, returning a 2-channel float32 spectrum.

    Returns:
        spectrum: (H, W, 2) float32 array where spectrum[...,0] is Re and spectrum[...,1] is Im.

    Args:
        image: Input image (grayscale or color). Converted to grayscale internally.
    """
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.dft(gray.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)


def fftshift2(spectrum: npt.NDArray[np.floating]) -> np.ndarray:
    """
    Shift the zero-frequency component to the center.

    Args:
        spectrum: A 2D array `(H,W)` or a 3D array `(H,W,2)` (OpenCV DFT format).

    Returns:
        Spectrum with quadrants swapped so that DC is at the center.
    """
    spec = np.asarray(spectrum)
    if spec.ndim == 2:
        return np.fft.fftshift(spec)
    return np.fft.fftshift(spec, axes=(0, 1))


def magnitude_spectrum(spectrum: npt.NDArray[np.floating], log_scale: bool = True) -> np.ndarray:
    """
    Convert a 2-channel OpenCV DFT spectrum into a magnitude image.

    Args:
        spectrum: OpenCV DFT output in shape `(H, W, 2)` with Re/Im channels.
        log_scale: If True, returns `log(1 + magnitude)` which is the standard way to
            visualize FFT spectra with large dynamic range.

    Returns:
        `float32` array of shape `(H, W)` with non-negative values.
    """
    mag = cv2.magnitude(spectrum[:, :, 0], spectrum[:, :, 1])
    if log_scale:
        mag = np.log1p(mag)
    return mag.astype(np.float32)


def ideal_low_pass_filter(shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float) -> np.ndarray:
    """
    Create an ideal (hard) low-pass frequency-domain mask.

    Args:
        shape: Target shape, typically the DFT spectrum shape `(H,W,2)` or `(H,W)`.
        cutoff_radius: Cutoff radius in pixels (in the frequency plane).

    Returns:
        A `float32` mask of shape `(H, W, 2)` suitable for elementwise multiplication
        with an OpenCV DFT spectrum.
    """
    H, W = shape[0], shape[1]
    cy, cx = H // 2, W // 2
    y, x = np.mgrid[0:H, 0:W]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask_2d = (dist <= cutoff_radius).astype(np.float32)
    return np.stack([mask_2d, mask_2d], axis=2)


def ideal_high_pass_filter(shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float) -> np.ndarray:
    """
    Create an ideal (hard) high-pass frequency-domain mask.

    This is defined as `1 - ideal_low_pass_filter(...)`.
    """
    return 1.0 - ideal_low_pass_filter(shape, cutoff_radius)


def apply_frequency_filter(image: npt.NDArray[np.generic], filter_mask: npt.NDArray[np.floating]) -> np.ndarray:
    """
    Filter an image in the frequency domain using an (H,W) or (H,W,2) mask.

    Args:
        image: Input image (grayscale or color). Converted to grayscale internally.
        filter_mask:
            Either:
            - `(H, W)` single-channel mask (will be broadcast to 2 channels), or
            - `(H, W, 2)` OpenCV-compatible 2-channel mask.

    Returns:
        Filtered spatial-domain image as `float32` of shape `(H, W)`.
    """
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    dft = cv2.dft(gray.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=(0, 1))

    mask = np.asarray(filter_mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = np.stack([mask, mask], axis=2)

    filtered_ishift = np.fft.ifftshift(dft_shift * mask, axes=(0, 1))
    result = cv2.idft(filtered_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return result.astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """
    Min-max normalize an array to `[0, 255]` (`uint8`) for visualization.

    Args:
        x: Any numeric array-like.

    Returns:
        2D/3D array (same shape as input) scaled to `uint8`.
    """
    arr = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    y = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(y, 0.0, 255.0).astype(np.uint8)


def main() -> int:
    """
    Lab 01 demo (skeleton).

    Expected behavior after implementation:
    - load 1-2 images from `./imgs/`
    - synthesize salt&pepper + Gaussian noise
    - compare median vs Gaussian vs box blur
    - compute Sobel/Laplacian edges
    - visualize FFT magnitude spectrum and apply ideal LPF/HPF
    - save all outputs into `./out/lab01/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 01 skeleton (implement functions first).")
    parser.add_argument("--img1", type=str, default="lenna.png", help="First image from ./imgs/")
    parser.add_argument("--img2", type=str, default="airplane.bmp", help="Second image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab01", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        arr = np.asarray(x, dtype=np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx <= mn:
            return np.zeros_like(arr, dtype=np.uint8)
        y = (arr - mn) * (255.0 / (mx - mn))
        return np.clip(y, 0.0, 255.0).astype(np.uint8)

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(imgs_dir / args.img1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(imgs_dir / args.img2), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise FileNotFoundError(str(imgs_dir / args.img1))
    if img2 is None:
        raise FileNotFoundError(str(imgs_dir / args.img2))

    missing: list[str] = []

    # --- Noise + denoise comparisons ---
    try:
        sp_noisy = add_salt_pepper_noise(img1, amount=0.08, salt_vs_pepper=0.55, seed=0)
        g_noisy = add_gaussian_noise(img1, sigma=15.0, seed=0)

        median_sp = apply_median_blur(sp_noisy, 5)
        gauss_sp = apply_gaussian_blur(sp_noisy, 5, 1.2)
        box_sp = apply_box_blur(sp_noisy, 5)

        gauss_g = apply_gaussian_blur(g_noisy, 5, 1.2)
        box_g = apply_box_blur(g_noisy, 5)

        plt.figure(figsize=(12, 6))
        for i, (title, im) in enumerate(
            [
                ("Original", img1),
                ("Salt & pepper", sp_noisy),
                ("Median (5x5)", median_sp),
                ("Gaussian (5,σ=1.2)", gauss_sp),
                ("Box (5x5)", box_sp),
                ("Gaussian noise", g_noisy),
            ],
            start=1,
        ):
            plt.subplot(2, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_sp_and_examples.png")

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Gaussian noise", g_noisy),
                ("Gaussian blur", gauss_g),
                ("Box blur", box_g),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_gaussian_noise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- Edge detection ---
    try:
        gx, gy, mag = sobel_edges(img2, ksize=3)
        _ = (gx, gy)
        lap = laplacian_edges(img2, ksize=3)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Sobel magnitude", normalize_to_uint8(mag)),
                ("Laplacian |·|", normalize_to_uint8(lap)),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "edges.png")

        cv2.imwrite(str(out_dir / "sobel_mag.png"), normalize_to_uint8(mag))
        cv2.imwrite(str(out_dir / "laplacian_abs.png"), normalize_to_uint8(lap))
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- FFT + frequency-domain filtering ---
    try:
        spec = fft2_image(img2)
        spec_shift = fftshift2(spec)
        mag = magnitude_spectrum(spec_shift, log_scale=True)

        lp = ideal_low_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        hp = ideal_high_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        lowpassed = apply_frequency_filter(img2, lp)
        highpassed = apply_frequency_filter(img2, hp)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Magnitude spectrum (log)", normalize_to_uint8(mag)),
                ("LPF result", normalize_to_uint8(lowpassed)),
                ("HPF result", normalize_to_uint8(highpassed)),
            ],
            start=1,
        ):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "fft_frequency_filters.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 01 demo is incomplete. Implement the TODO functions in labs/lab01_filtering_convolution_fft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
