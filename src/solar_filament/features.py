"""Versioned two-channel feature recipes shared by training and inference."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np


FEATURE_SPECS = {
    'contrast': {'version': 1, 'channels': 2, 'sigma': 20, 'scale': 32,
                 'background_dtype': 'uint8', 'border': 'reflect101'},
    'bandpass': {'version': 1, 'channels': 2, 'sigmas': [2, 12], 'scale': 32,
                 'padding': 'reflect_4sigma', 'response': 'signed_large_minus_small_gaussian_dft'},
    'gabor': {'version': 1, 'channels': 2, 'sigmas': [2, 4], 'wavelengths': [8, 16],
              'orientations_degrees': [0, 45, 90, 135], 'gamma': 0.5, 'phase': 0,
              'kernel': 'zero_mean_l1_norm_2', 'pool': 'signed_max_absolute',
              'scale': 32, 'border': 'reflect101'},
}


@lru_cache(maxsize=8)
def _transfer(shape: tuple[int, int], small: float, large: float) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    squared = fy * fy + fx * fx
    return (np.exp(-2 * np.pi**2 * large**2 * squared)
            - np.exp(-2 * np.pi**2 * small**2 * squared)).astype(np.float32)


def dft_bandpass(image: np.ndarray, small: float = 2, large: float = 12) -> np.ndarray:
    if not 0 < small < large:
        raise ValueError('Require 0 < small < large')
    padding = int(np.ceil(4 * large))
    padded = np.pad(image.astype(np.float32), padding, mode='reflect')
    response = np.fft.irfft2(np.fft.rfft2(padded) * _transfer(padded.shape, small, large), s=padded.shape).real
    return response[padding:-padding, padding:-padding].astype(np.float32)


@lru_cache(maxsize=1)
def _gabor_kernels() -> tuple[np.ndarray, ...]:
    kernels = []
    for sigma, wavelength in zip((2, 4), (8, 16)):
        radius = int(np.ceil(4 * sigma / 0.5))
        for angle in (0, np.pi/4, np.pi/2, 3*np.pi/4):
            kernel = cv2.getGaborKernel((2*radius+1,)*2, sigma, angle, wavelength, 0.5, 0, ktype=cv2.CV_32F)
            kernel -= kernel.mean()
            kernel *= np.float32(2 / np.abs(kernel).sum())
            kernels.append(kernel)
    return tuple(kernels)


def signed_gabor(image: np.ndarray) -> np.ndarray:
    intensity = image.astype(np.float32)
    pooled = np.zeros(image.shape, np.float32)
    for kernel in _gabor_kernels():
        response = -cv2.filter2D(intensity, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
        pooled = np.where(np.abs(response) > np.abs(pooled), response, pooled)
    return pooled


def make_features(tile: np.ndarray, recipe: str = 'contrast') -> np.ndarray:
    if recipe not in FEATURE_SPECS:
        raise ValueError(f'Unknown feature recipe: {recipe}')
    intensity = tile.astype(np.float32) / 255.0
    if recipe == 'contrast':
        # Preserve the accepted checkpoint's uint8 Gaussian rounding exactly.
        background = cv2.GaussianBlur(tile, (0, 0), 20)
        response = background.astype(np.float32) - tile
    elif recipe == 'bandpass':
        response = dft_bandpass(tile)
    else:
        response = signed_gabor(tile)
    return np.stack((intensity, np.clip(response / 32.0, -1, 1)), axis=0)
