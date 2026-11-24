/// <reference lib="webworker" />
import * as Comlink from "comlink";
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  env,
  RawImage,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type Processor,
  type Tensor,
} from "@huggingface/transformers";
import { mean, normalize } from "@/core/vector";
import { imageViews, type View } from "@/core/views";
import type { ClipWorkerApi } from "./clipWorkerTypes";
import { CLIP_MODEL_REPO, CLIP_RUNTIMES } from "./clipWorkerTypes";
import type { ModelInfo, ModelLoadProgress } from "./types";

env.allowLocalModels = false;
env.useBrowserCache = true;

interface Loaded {
  device: string;
  processor: Processor;
  tokenizer: PreTrainedTokenizer;
  vision: PreTrainedModel;
  text: PreTrainedModel;
}

let loading: Promise<Loaded> | null = null;

function load(onProgress?: (p: ModelLoadProgress) => void): Promise<Loaded> {
  loading ??= (async () => {
    // Prefer the library's aggregate event; per-file sums jump to 100 % before later files start.
    const files = new Map<string, { loaded: number; total: number }>();
    let hasAggregate = false;
    const progress_callback = (info: { status: string; file?: string; loaded?: number; total?: number }) => {
      if (!onProgress) return;
      if (info.status === "progress_total") {
        hasAggregate = true;
        onProgress({ loadedBytes: info.loaded ?? 0, totalBytes: info.total ?? 0 });
        return;
      }
      if (info.status !== "progress" || !info.file || hasAggregate) return;
      files.set(info.file, { loaded: info.loaded ?? 0, total: info.total ?? 0 });
      let loadedBytes = 0;
      let totalBytes = 0;
      for (const f of files.values()) {
        loadedBytes += f.loaded;
        totalBytes += f.total;
      }
      onProgress({ loadedBytes, totalBytes });
    };

    const [processor, tokenizer] = await Promise.all([
      AutoProcessor.from_pretrained(CLIP_MODEL_REPO, { progress_callback }),
      AutoTokenizer.from_pretrained(CLIP_MODEL_REPO, { progress_callback }),
    ]);

    let lastError: unknown = null;
    for (const runtime of CLIP_RUNTIMES) {
      if (runtime.device === "webgpu" && !("gpu" in navigator)) continue;
      try {
        const opts = { device: runtime.device, dtype: runtime.dtype, progress_callback };
        const [vision, text] = await Promise.all([
          CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_REPO, opts),
          CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_REPO, opts),
        ]);
        return { device: runtime.device, processor, tokenizer, vision, text };
      } catch (err) {
        lastError = err;
        console.warn(`CLIP: runtime ${runtime.device} failed, trying next`, err);
      }
    }
    throw lastError ?? new Error("No CLIP runtime available");
  })();
  loading.catch(() => {
    loading = null; // allow a retry after a failed download
  });
  return loading;
}

function rows(tensor: Tensor): Float32Array[] {
  const [n, dim] = tensor.dims as [number, number];
  const data = tensor.data as Float32Array;
  return Array.from({ length: n }, (_, i) => normalize(data.subarray(i * dim, (i + 1) * dim)));
}

const VIEW_SIZE = 224;

/** Draws one view as a 224 px square; letterboxed views are padded with the image's mean colour. */
function renderView(bitmap: ImageBitmap, v: View): RawImage {
  const canvas = new OffscreenCanvas(VIEW_SIZE, VIEW_SIZE);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  if (v.letterbox) {
    ctx.drawImage(bitmap, 0, 0, 1, 1);
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, 0, VIEW_SIZE, VIEW_SIZE);
    const scale = VIEW_SIZE / Math.max(v.sw, v.sh);
    const w = v.sw * scale;
    const h = v.sh * scale;
    ctx.drawImage(bitmap, v.sx, v.sy, v.sw, v.sh, (VIEW_SIZE - w) / 2, (VIEW_SIZE - h) / 2, w, h);
  } else {
    ctx.drawImage(bitmap, v.sx, v.sy, v.sw, v.sh, 0, 0, VIEW_SIZE, VIEW_SIZE);
  }
  return RawImage.fromCanvas(canvas);
}

const api: ClipWorkerApi = {
  async load(onProgress): Promise<ModelInfo> {
    const m = await load(onProgress);
    return { device: m.device };
  },

  async embedImages(blobs) {
    const m = await load();
    // Several views per image (see core/views), embedded in one batch and averaged.
    const perImage: RawImage[][] = [];
    for (const blob of blobs) {
      const bitmap = await createImageBitmap(blob);
      try {
        perImage.push(imageViews(bitmap.width, bitmap.height).map((v) => renderView(bitmap, v)));
      } finally {
        bitmap.close();
      }
    }
    const inputs = await m.processor(perImage.flat());
    const { image_embeds } = await m.vision(inputs);
    const all = rows(image_embeds as Tensor);
    let offset = 0;
    return perImage.map((views) => {
      const out = normalize(mean(all.slice(offset, offset + views.length)));
      offset += views.length;
      return out;
    });
  },

  async embedTexts(texts) {
    const m = await load();
    const inputs = m.tokenizer(texts, { padding: true, truncation: true });
    const { text_embeds } = await m.text(inputs);
    return rows(text_embeds as Tensor);
  },
};

Comlink.expose(api);
