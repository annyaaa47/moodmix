/// <reference lib="webworker" />
import * as Comlink from "comlink";
import { averageColor, colorDescriptor, dHash, extractPalette } from "@/core/pixels";
import type { MediaApi, ProcessedImage } from "./types";

const FULL_MAX_SIDE = 2000;
const THUMB_MAX_SIDE = 400;
const WEBP_QUALITY = 0.85;
const PALETTE_SIDE = 64;

async function process(file: Blob): Promise<ProcessedImage> {
  const buffer = await file.arrayBuffer();
  const contentHash = await sha256(buffer);
  // createImageBitmap decodes JPG/PNG/WebP/AVIF; for GIF it returns the first frame.
  const bitmap = await createImageBitmap(new Blob([buffer], { type: file.type }));
  try {
    const { width, height } = bitmap;
    const full = await encode(bitmap, FULL_MAX_SIDE);
    const thumb = await encode(bitmap, THUMB_MAX_SIDE);
    const phash = dHash(pixels(bitmap, 9, 8));
    const avgColor = averageColor(pixels(bitmap, 16, 16));
    const colorDesc = colorDescriptor(pixels(bitmap, 48, 48));
    const palette = extractPalette(pixels(bitmap, PALETTE_SIDE, PALETTE_SIDE));
    return { full: full.blob, thumb: thumb.blob, width, height, mime: full.blob.type, contentHash, phash, avgColor, colorDesc, palette };
  } finally {
    bitmap.close();
  }
}

/** Colour descriptor for an already stored image (backfill). */
async function describe(blob: Blob): Promise<Float32Array> {
  const bitmap = await createImageBitmap(blob);
  try {
    return colorDescriptor(pixels(bitmap, 48, 48));
  } finally {
    bitmap.close();
  }
}

/** Palette for an already stored image (backfill). */
async function palette(blob: Blob): Promise<string[]> {
  const bitmap = await createImageBitmap(blob);
  try {
    return extractPalette(pixels(bitmap, PALETTE_SIDE, PALETTE_SIDE));
  } finally {
    bitmap.close();
  }
}

async function encode(bitmap: ImageBitmap, maxSide: number): Promise<{ blob: Blob }> {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, w, h);
  let blob = await canvas.convertToBlob({ type: "image/webp", quality: WEBP_QUALITY });
  // Safari cannot encode WebP and silently returns PNG; JPEG is much smaller then.
  if (blob.type !== "image/webp") blob = await canvas.convertToBlob({ type: "image/jpeg", quality: WEBP_QUALITY });
  return { blob };
}

function pixels(bitmap: ImageBitmap, w: number, h: number): Uint8ClampedArray {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Content hash of any file (videos: duplicates are found by the file itself). */
async function hash(file: Blob): Promise<string> {
  return sha256(await file.arrayBuffer());
}

const api: MediaApi = { process, describe, palette, hash };
Comlink.expose(api);
