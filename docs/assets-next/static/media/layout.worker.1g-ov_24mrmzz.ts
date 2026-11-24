/// <reference lib="webworker" />
import * as Comlink from "comlink";
import { overviewPositions } from "@/core/lookLayout";
import type { LayoutApi } from "./layoutClient";

const api: LayoutApi = { overview: overviewPositions };

Comlink.expose(api);
