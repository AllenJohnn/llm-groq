// LLM ShardX performance and scaling analytics.
// Room throughput is measured from the single sampled output stream; device cards show shard participation.

const DEVICE_COLORS = [
  "#2a45e0", // Host / You
  "#159a78",
  "#e39a2d",
  "#8664d8",
  "#d66b58",
  "#218eaa",
  "#b6568d",
  "#718096",
];
const PERF_CHART_TEXT = "#5e616b";
const PERF_CHART_GRID = "rgba(70, 78, 98, 0.12)";
const PERF_SIDEBAR_WIDTH_KEY = "swarm_perf_sidebar_width";
const PERF_SIDEBAR_MIN_WIDTH = 310;
const PERF_SIDEBAR_MAX_WIDTH = 560;
const PERF_SIDEBAR_DEFAULT_WIDTH = 330;

export class PerfSidebar {
  constructor() {
    this.container = null;
    this.liveCanvas = null;
    this.scalingCanvas = null;
    this.sessionCanvas = null;
    this.liveCtx = null;
    this.scalingCtx = null;
    this.sessionCtx = null;
    this.deviceCanvas = null;
    this.comparisonCanvas = null;
    this.deviceShareChart = null;
    this.deviceComparisonChart = null;
    
    // Live stream state
    this.isStreaming = false;
    this.genStartTime = 0;
    this.lastTokenTime = 0;
    this.tokenCount = 0;
    this.firstTokenTime = null;
    this.peakTps = 0;
    this.streamPoints = []; // { t: elapsedSec, tps: instantTps, total: count }
    this.recentTokenTimes = []; // [timestamp, timestamp, ...] for sliding window
    this.animFrameId = null;

    // Devices & Grid state
    this.devices = [
      {
        id: "self",
        name: "you",
        self: true,
        color: DEVICE_COLORS[0],
        layers: "",
        stage: "Solo Engine",
        stageIndex: 0,
        totalStages: 1,
        meta: {},
        rtt: null,
        bw: null,
        tokens: 0,
        tps: 0,
        peakTps: 0,
        streamPoints: [],
        recentTokenTimes: [],
        visible: true,
        pipelineDelayMs: 0
      }
    ];
    this.deviceSessionTotals = new Map(); // id -> totalTokensInSession
    this.activeHighlightId = null; // null = all visible, or specific device id
    this.showClusterCurve = true;

    // Cluster & Session state
    this.clusterSize = 1;
    this.sessionTokens = 0;
    this.promptCount = 0;
    this.totalGenerationTime = 0;
    this.currentModel = "";
    this.currentDeviceCount = 1;
    this.backend = "local";
    this.lastInstantTps = 0;
    this.flowSignature = "";
    this.sessionHistory = [];
    this.sessionStorageKey = "webslice_perf_history_v1";
    this.devicePerformanceStorageKey = "webslice_device_performance_v1";
    this.devicePerformance = [];
    try {
      const saved = sessionStorage.getItem(this.sessionStorageKey);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) this.sessionHistory = parsed;
      }
    } catch {}
    try {
      const savedComparisons = localStorage.getItem(this.devicePerformanceStorageKey);
      if (savedComparisons) {
        const parsed = JSON.parse(savedComparisons);
        if (Array.isArray(parsed)) this.devicePerformance = parsed.filter(item => item && typeof item.model === "string");
      }
    } catch {}
    this.sessionTokens = this.sessionHistory.reduce((sum, item) => sum + (Number(item.tokens) || 0), 0);
    this.promptCount = this.sessionHistory.length;
    this.totalGenerationTime = this.sessionHistory.reduce((sum, item) => sum + (Number(item.seconds) || 0), 0);

    // Scalability benchmark curve (Nodes vs Throughput Multiplier / Speed)
    // Strictly in increasing order to demonstrate multi-system efficiency
    this.observedScaling = new Map(); // clusterSize -> peakTps
  }

  init() {
    if (typeof document === "undefined") return;
    if (document.getElementById("perf-sidebar")) return;

    this.injectStyles();
    this.mountDOM();
    this.setupListeners();
    this.setupCanvases();
    this.updateDeviceListUI();
    this.renderDevicePerformance();
    this.renderScalingChart();
    this.renderLiveChart();
    this.renderSessionChart();
  }

  setDevices(newDevices) {
    const oldMap = new Map(this.devices.map(d => [d.id, d]));
    this.devices = newDevices.map((nd, idx) => {
      const old = oldMap.get(nd.id) || {};
      return {
        ...old,
        ...nd,
        tps: old.tps || 0,
        tokens: old.tokens || 0,
        workerRole: nd.workerRole || old.workerRole || "Idle",
        visible: old.visible !== undefined ? old.visible : true,
        streamPoints: old.streamPoints || [],
        color: DEVICE_COLORS[idx % DEVICE_COLORS.length]
      };
    });
    this.clusterSize = this.devices.length;
    this.updateDeviceListUI();
    this.renderDevicePerformance();
    this.renderScalingChart();
    this.renderLiveChart();
    this.renderSessionChart();
  }

  setBackend(backend) {
    this.backend = backend === "cloud" ? "cloud" : "local";
    this.renderShardFlow(this.lastInstantTps);
    this.renderDeviceShareChart();
    this.renderScalingChart();
  }

  injectStyles() {
    if (document.getElementById("perf-sidebar-styles")) return;
    const style = document.createElement("style");
    style.id = "perf-sidebar-styles";
    style.textContent = `
      /* ---- LLM ShardX Performance & Scaling Sidebar ---- */
      #perf-sidebar {
        flex: 1 1 360px;
        max-width: 500px;
        min-width: 310px;
        width: auto !important;
        display: flex;
        flex-direction: column;
        border: 1px solid var(--border);
        border-radius: 24px;
        background: color-mix(in srgb, var(--panel) 85%, transparent);
        overflow-y: auto;
        overflow-x: hidden;
        min-height: 0;
        transition: transform 0.28s cubic-bezier(0.2, 0.8, 0.2, 1),
                    opacity 0.2s ease;
        z-index: 25;
        box-shadow: 0 4px 20px rgba(20, 21, 26, 0.04);
      }
      #perf-sidebar.collapsed {
        display: none;
      }
      @media (max-width: 1200px) {
        #perf-sidebar {
          position: fixed;
          top: 61px;
          right: 0;
          bottom: 0;
          width: 360px !important;
          max-width: 90vw;
          border-radius: 0;
          box-shadow: -8px 0 32px rgba(20, 21, 26, 0.14);
          z-index: 90;
        }
        #perf-sidebar.collapsed {
          display: flex;
          transform: translateX(100%);
          margin-right: 0;
          width: 360px !important;
        }
      }
      @media (max-width: 640px) {
        #perf-sidebar {
          width: 100vw !important;
          max-width: 100vw;
        }
        #perf-sidebar.collapsed {
          transform: translateX(100%);
          width: 100vw !important;
        }
      }

      /* Topbar Toggle Chip */
      .topbar-perf-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 100px;
        padding: 6px 14px 6px 12px;
        font-family: var(--mono);
        font-size: 13.5px;
        color: var(--text);
        cursor: pointer;
        user-select: none;
        transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1);
        white-space: nowrap;
        margin-right: 4px;
      }
      .topbar-perf-chip:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(43, 78, 255, 0.1);
      }
      .topbar-perf-chip.active {
        border-color: color-mix(in srgb, var(--accent) 60%, var(--border));
        background: color-mix(in srgb, var(--accent) 8%, var(--panel));
        color: var(--accent);
        font-weight: 600;
      }
      .topbar-perf-chip.streaming {
        border-color: var(--accent);
        background: color-mix(in srgb, var(--accent) 12%, var(--panel));
        color: var(--accent);
        animation: perfChipPulse 1.6s infinite ease-in-out;
      }
      @keyframes perfChipPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(43, 78, 255, 0.3); }
        50% { box-shadow: 0 0 0 4px rgba(43, 78, 255, 0); }
      }
      .topbar-perf-icon {
        color: var(--accent);
        flex: none;
      }

      /* Sidebar Internal Layout */
      .perf-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 16px 12px;
        border-bottom: 1px solid var(--border);
        background: color-mix(in srgb, var(--panel-2) 60%, var(--panel));
        flex: none;
      }
      .perf-title-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .perf-icon-badge {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: color-mix(in srgb, var(--accent) 15%, transparent);
        color: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        flex: none;
      }
      .perf-title {
        font-family: var(--mono);
        font-size: 13.5px;
        letter-spacing: 0.14em;
        font-weight: 700;
        color: var(--text);
      }
      .perf-header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .perf-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-family: var(--mono);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        padding: 3px 8px;
        border-radius: 100px;
        background: var(--panel-2);
        color: var(--muted);
        border: 1px solid var(--border);
        transition: all 0.2s ease;
      }
      .perf-status-pill.streaming {
        background: color-mix(in srgb, var(--accent) 14%, transparent);
        color: var(--accent);
        border-color: color-mix(in srgb, var(--accent) 40%, transparent);
      }
      .perf-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--muted);
      }
      .perf-status-pill.streaming .perf-status-dot {
        background: var(--accent);
        animation: perfDotBlink 1s infinite alternate;
      }
      @keyframes perfDotBlink {
        from { opacity: 0.4; transform: scale(0.85); }
        to { opacity: 1; transform: scale(1.15); }
      }
      .perf-close-btn {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 4px;
        cursor: pointer;
        color: var(--muted);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.18s ease;
      }
      .perf-close-btn:hover {
        color: var(--text);
        border-color: var(--border);
        background: var(--panel-2);
      }

      /* Sections */
      .perf-section {
        padding: 14px 16px;
        border-bottom: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .perf-sec-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 13px;
        letter-spacing: 0.16em;
        color: var(--muted);
        font-weight: 600;
        text-transform: uppercase;
      }
      .perf-unit {
        font-size: 12px;
        font-weight: 500;
        color: var(--text);
        text-transform: none;
        letter-spacing: 0;
      }
      .perf-badge-pill {
        font-size: 12px;
        font-weight: 600;
        color: var(--accent);
        background: color-mix(in srgb, var(--accent) 12%, transparent);
        padding: 3px 8px;
        border-radius: 100px;
        letter-spacing: 0;
      }

      /* Hero Speed Display */
      .perf-hero-stat {
        display: flex;
        flex-direction: column;
        gap: 2px;
        margin: 2px 0 4px;
      }
      .perf-hero-main-row {
        display: flex;
        align-items: baseline;
        gap: 6px;
      }
      .perf-hero-val {
        font-family: var(--sans);
        font-size: 38px;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.03em;
        color: var(--text);
      }
      .perf-hero-val.streaming {
        color: var(--accent);
      }
      .perf-hero-unit {
        font-family: var(--mono);
        font-size: 13.5px;
        font-weight: 600;
        color: var(--muted);
        letter-spacing: 0.08em;
      }
      .perf-hero-sub {
        font-family: var(--mono);
        font-size: 13px;
        color: var(--muted);
        margin-top: 2px;
      }

      /* Multi-Device Legend Chips above Canvas */
      .perf-device-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin: 2px 0 6px;
      }
      .perf-legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-family: var(--mono);
        font-size: 10px;
        color: var(--text);
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 100px;
        padding: 3px 8px;
        cursor: pointer;
        user-select: none;
        transition: all 0.18s ease;
      }
      .perf-legend-chip:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
      }
      .perf-legend-chip.active {
        border-color: color-mix(in srgb, var(--accent) 50%, var(--border));
        background: color-mix(in srgb, var(--accent) 8%, var(--panel));
      }
      .perf-legend-chip.dimmed {
        opacity: 0.45;
        border-style: dashed;
      }
      .perf-chip-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex: none;
      }
      .perf-chip-speed {
        font-weight: 600;
        color: var(--muted);
      }
      .perf-legend-chip.active .perf-chip-speed {
        color: var(--text);
      }

      /* Canvas Wrap */
      .perf-canvas-wrap {
        position: relative;
        width: 100%;
        height: 124px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
      }
      .perf-canvas-wrap canvas {
        width: 100%;
        height: 100%;
        display: block;
      }
      .perf-canvas-empty {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        padding: 16px;
        pointer-events: none;
        line-height: 1.4;
      }
      .perf-canvas-wrap.active .perf-canvas-empty {
        display: none;
      }

      .perf-viz-card {
        grid-column: 1 / -1;
        padding: 16px 18px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--panel);
        box-shadow: 0 1px 2px rgba(20, 22, 29, .03);
      }
      .perf-viz-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 8px;
      }
      .perf-viz-title { color: var(--text); font: 600 14px/1.35 var(--sans); }
      .perf-viz-subtitle { margin-top: 3px; color: var(--muted); font: 12px/1.4 var(--sans); }
      .perf-viz-badge {
        flex: none; padding: 5px 9px; border: 1px solid var(--border); border-radius: 999px;
        background: var(--panel-2); color: var(--muted); font: 600 10px/1 var(--sans);
      }
      .perf-device-canvas-wrap, .perf-compare-canvas-wrap {
        position: relative; width: 100%; height: 210px; min-height: 160px;
      }
      .perf-compare-canvas-wrap { height: 190px; min-height: 150px; margin: 10px 0 14px; }
      .perf-device-canvas-wrap canvas, .perf-compare-canvas-wrap canvas { display: block; width: 100%; height: 100%; }
      .perf-viz-empty {
        position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
        padding: 18px; border-radius: 12px; background: color-mix(in srgb, var(--panel) 88%, transparent);
        color: var(--muted); text-align: center; font: 12px/1.5 var(--sans); pointer-events: none;
      }
      .perf-viz-empty[hidden] { display: none; }
      .perf-device-status-list {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; margin-top: 10px;
      }
      .perf-device-status {
        display: flex; align-items: flex-start; gap: 9px; min-width: 0; padding: 9px 10px;
        border: 1px solid var(--border-2); border-radius: 11px; background: var(--panel-2);
      }
      .perf-device-status-dot { width: 8px; height: 8px; flex: none; margin-top: 4px; border-radius: 50%; }
      .perf-device-status-copy { min-width: 0; }
      .perf-device-status-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); font: 600 12px/1.3 var(--sans); }
      .perf-device-status-meta { margin-top: 3px; color: var(--muted); font: 11px/1.35 var(--sans); }
      .perf-viz-legend { display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 8px; }
      .perf-viz-legend-item { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font: 11px/1.3 var(--sans); }
      .perf-viz-legend-item i { width: 8px; height: 8px; border-radius: 3px; }

      /* Animated model-shard flow: the stream rate is shared, each node shows its assigned work. */
      .perf-flow {
        grid-column: 1 / -1;
        padding: 14px 16px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 5%, white), white 68%);
        overflow: hidden;
      }
      .perf-flow-head, .perf-flow-readouts {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
      }
      .perf-flow-title { font: 600 13px/1.3 var(--sans); color: var(--text); }
      .perf-flow-state { font: 11px/1.3 var(--sans); color: var(--muted); }
      .perf-flow-track {
        display: flex;
        align-items: center;
        gap: 0;
        margin: 16px 0 12px;
        overflow-x: auto;
        scrollbar-width: thin;
        padding: 3px 2px 8px;
      }
      .perf-flow-node {
        position: relative;
        z-index: 1;
        flex: 0 0 auto;
        min-width: 112px;
        max-width: 180px;
        padding: 9px 11px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(255,255,255,.94);
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
      }
      .perf-flow-node-top { display: flex; align-items: center; gap: 7px; min-width: 0; }
      .perf-flow-node-dot {
        width: 8px; height: 8px; border-radius: 50%; flex: none;
        background: var(--device-color, var(--accent));
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--device-color, var(--accent)) 13%, transparent);
      }
      .perf-flow-node-name {
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        font: 600 12px/1.2 var(--sans); color: var(--text);
      }
      .perf-flow-node-range {
        display: block; margin: 6px 0 0 15px;
        font: 10.5px/1.25 var(--sans); color: var(--muted);
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      }
      .perf-flow-node--prompt, .perf-flow-node--answer {
        border-color: color-mix(in srgb, var(--accent) 24%, var(--border));
        background: color-mix(in srgb, var(--accent) 5%, white);
      }
      .perf-flow-node-step {
        display: block; margin: 6px 0 0 15px;
        font: 10.5px/1.25 var(--sans); color: var(--muted);
      }
      .perf-chart-title { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 7px 12px; margin: 4px 0 8px; font: 500 12px/1.4 var(--sans); color: var(--muted); }
      .perf-live-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; color: var(--muted); font: 11px/1.3 var(--sans); }
      .perf-live-legend-item { display: inline-flex; align-items: center; gap: 6px; }
      .perf-live-legend-swatch { width: 16px; height: 2px; border-radius: 2px; background: #2a45e0; }
      .perf-live-legend-swatch.raw { height: 1px; background: #8a9afa; }
      .perf-flow-link {
        position: relative; flex: 1 0 30px; height: 2px; min-width: 30px;
        background: color-mix(in srgb, var(--accent) 20%, var(--border));
      }
      .perf-flow-pulse {
        position: absolute; top: 50%; left: 0; width: 7px; height: 7px;
        border-radius: 50%; background: var(--accent); transform: translate(-50%, -50%);
        opacity: 0; box-shadow: 0 0 10px color-mix(in srgb, var(--accent) 70%, transparent);
      }
      .perf-flow.streaming .perf-flow-node {
        border-color: color-mix(in srgb, var(--device-color, var(--accent)) 38%, var(--border));
        animation: perfShardPulse 1.25s ease-in-out infinite alternate;
      }
      .perf-flow.streaming .perf-flow-node:nth-of-type(4n + 1) { animation-delay: -.2s; }
      .perf-flow.streaming .perf-flow-node:nth-of-type(4n + 2) { animation-delay: -.45s; }
      .perf-flow.streaming .perf-flow-node:nth-of-type(4n + 3) { animation-delay: -.7s; }
      .perf-flow.streaming .perf-flow-pulse { opacity: 1; animation: perfFlowTravel var(--perf-flow-duration, 1s) ease-in-out infinite; }
      .perf-flow.streaming .perf-flow-link:nth-child(4n + 2) .perf-flow-pulse { animation-delay: -.35s; }
      .perf-flow.streaming .perf-flow-link:nth-child(4n + 4) .perf-flow-pulse { animation-delay: -.65s; }
      @keyframes perfShardPulse {
        from { box-shadow: 0 0 0 rgba(0,122,255,0); transform: translateY(0); }
        to { box-shadow: 0 5px 18px rgba(0,122,255,.09); transform: translateY(-2px); }
      }
      @keyframes perfFlowTravel {
        from { left: 0; }
        to { left: 100%; }
      }
      .perf-flow-readouts {
        justify-content: flex-start; flex-wrap: wrap;
        padding-top: 10px; border-top: 1px solid color-mix(in srgb, var(--border) 75%, transparent);
      }
      .perf-flow-metric { font: 11px/1.35 var(--sans); color: var(--muted); }
      .perf-flow-metric b { margin-left: 4px; color: var(--text); font-weight: 600; }
      .perf-flow-note { display: block; margin-top: 8px; font: 11px/1.45 var(--sans); color: var(--muted); }
      .perf-flow-empty { padding: 12px 2px; color: var(--muted); font: 12px/1.45 var(--sans); }
      @media (prefers-reduced-motion: reduce) {
        .perf-flow.streaming .perf-flow-node, .perf-flow.streaming .perf-flow-pulse { animation: none; }
        .perf-flow.streaming .perf-flow-pulse { opacity: .9; left: 50%; }
      }

      /* Grid Devices Live Processing Breakdown */
      .perf-devices-breakdown-wrap {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 6px;
      }
      .perf-dev-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 9px 11px;
        display: flex;
        flex-direction: column;
        gap: 6px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }
      .perf-dev-card.streaming {
        border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
        box-shadow: 0 2px 10px rgba(43, 78, 255, 0.06);
      }
      .perf-dev-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .perf-dev-title-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
        min-width: 0;
      }
      .perf-dev-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        flex: none;
      }
      .perf-dev-name {
        font-family: var(--mono);
        font-size: 13.5px;
        font-weight: 600;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .perf-dev-stage-badge {
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 500;
        color: var(--muted);
        background: var(--panel-2);
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid var(--border);
      }
      .perf-dev-rates {
        display: flex;
        align-items: baseline;
        gap: 6px;
        flex: none;
      }
      .perf-dev-tps {
        font-family: var(--mono);
        font-size: 14.5px;
        font-weight: 700;
        color: var(--text);
      }
      .perf-dev-toks {
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }
      .perf-dev-meter-track {
        height: 5px;
        background: color-mix(in srgb, var(--border) 70%, transparent);
        border-radius: 100px;
        overflow: hidden;
        width: 100%;
      }
      .perf-dev-meter-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 0.18s cubic-bezier(0.2, 0.8, 0.2, 1);
      }
      .perf-dev-foot {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }

      /* Metric Readouts Grid */
      .perf-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 4px;
      }
      .perf-stat-item {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 9px 12px;
        display: flex;
        flex-direction: column;
        gap: 3px;
      }
      .perf-stat-k {
        font-family: var(--mono);
        font-size: 11.5px;
        letter-spacing: 0.12em;
        color: var(--muted);
        font-weight: 600;
      }
      .perf-stat-v {
        font-family: var(--mono);
        font-size: 16px;
        font-weight: 600;
        color: var(--text);
      }
      .perf-compare {
        margin-top: 14px;
        padding: 14px;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: var(--panel);
      }
      .perf-compare-title { font: 600 13px/1.3 var(--sans); color: var(--text); }
      .perf-compare-context { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 10px; margin-top: 8px; color: var(--muted); font: 11px/1.4 var(--sans); }
      .perf-compare-current { display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border: 1px solid color-mix(in srgb, var(--accent) 24%, var(--border)); border-radius: 999px; background: color-mix(in srgb, var(--accent) 6%, var(--panel)); color: var(--text); font-weight: 600; }
      .perf-compare-current::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #159a78; }
      .perf-compare-model { margin-top: 4px; font: 11px/1.4 var(--sans); color: var(--muted); white-space: normal; }
      .perf-compare-grid { display: grid; grid-template-columns: minmax(94px, 1.35fr) repeat(3, minmax(48px, .8fr)); gap: 6px; align-items: center; }
      .perf-compare-head { margin-top: 13px; padding: 0 6px 6px; color: var(--muted); font: 9px/1.2 var(--sans); }
      .perf-compare-row { min-height: 48px; padding: 7px 6px; border-top: 1px solid color-mix(in srgb, var(--border) 72%, transparent); color: var(--text); font: 11px/1.2 var(--sans); }
      .perf-compare-count { font-weight: 600; }
      .perf-compare-detail { display: block; margin-top: 3px; color: var(--muted); font-size: 9px; font-weight: 400; }
      .perf-compare-value { font-variant-numeric: tabular-nums; white-space: nowrap; }
      .perf-compare-empty { margin-top: 12px; color: var(--muted); font: 11px/1.45 var(--sans); }

      /* Scalability section */
      .perf-desc {
        font-size: 13.5px;
        color: var(--muted);
        line-height: 1.5;
        margin-bottom: 2px;
      }
      .perf-scaling-legend {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-top: 6px;
      }
      .perf-scaling-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 13px;
        padding: 5px 10px;
        border-radius: 6px;
        background: var(--panel);
        border: 1px solid transparent;
        transition: all 0.2s ease;
      }
      .perf-scaling-row.active {
        border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
        background: color-mix(in srgb, var(--accent) 8%, var(--panel));
        font-weight: 600;
      }
      .perf-scaling-row.active .scaling-node-label {
        color: var(--accent);
      }
      .perf-scaling-row.active .scaling-node-mult {
        color: var(--accent);
      }
      .scaling-node-label {
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .scaling-active-tag {
        font-size: 10.5px;
        background: var(--accent);
        color: #fff;
        padding: 1px 5px;
        border-radius: 3px;
        font-weight: 700;
        letter-spacing: 0.05em;
      }
      .scaling-node-mult {
        font-weight: 600;
        color: var(--muted);
      }

      .perf-session-history {
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-top: 7px;
      }
      .perf-history-row {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2px 8px;
        padding: 7px 10px;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--panel);
      }
      .perf-history-main, .perf-history-sub {
        display: flex;
        align-items: center;
        gap: 5px;
      }
      .perf-history-main { color: var(--text); font-size: 12.5px; }
      .perf-history-sub { grid-column: 1 / -1; color: var(--muted); font-size: 11.5px; }
      .perf-history-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex: none;
      }
      .perf-history-empty {
        color: var(--muted);
        font-size: 12.5px;
        line-height: 1.4;
        padding: 4px 0;
      }

      /* Summary List */
      .perf-summary-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .perf-summary-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 13.5px;
        color: var(--muted);
        padding: 2px 0;
      }
      .perf-summary-row b {
        color: var(--text);
        font-weight: 600;
      }
      .perf-dev-sess-list {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px dashed var(--border);
      }
      .perf-dev-sess-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 12.5px;
        color: var(--muted);
      }
      .perf-dev-sess-row span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
      }
    `;
    document.head.appendChild(style);
  }

  mountDOM() {
    // 1. Mount Topbar Toggle Button inside <header>
    const header = document.querySelector("header");
    if (header && !document.getElementById("topbar-perf-btn")) {
      const topbarPeers = document.getElementById("topbar-peers");
      const btn = document.createElement("button");
      btn.id = "topbar-perf-btn";
      btn.className = "topbar-perf-chip active";
      btn.type = "button";
      btn.title = "Toggle Speed & LLM ShardX Scaling Analytics";
      btn.innerHTML = `
        <svg class="topbar-perf-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        <span id="topbar-perf-rate">0.0 text pieces/s</span>
      `;
      btn.onclick = () => this.toggleSidebar();
      if (topbarPeers) {
        header.insertBefore(btn, topbarPeers);
      } else {
        header.appendChild(btn);
      }
    }

    // 2. Mount Sidebar into #room-screen
    const roomScreen = document.getElementById("room-screen");
    if (!roomScreen) return;

    const aside = document.createElement("aside");
    aside.id = "perf-sidebar";
    aside.className = "perf-sidebar";
    aside.innerHTML = `
      <div class="perf-resize-handle" id="perf-resize-handle" role="separator" tabindex="0"
        aria-orientation="vertical" aria-label="Resize Performance sidebar" aria-valuemin="310" aria-valuemax="560" aria-valuenow="330"
        title="Drag to resize. Use the arrow keys to adjust width."></div>
      <!-- Header -->
      <div class="perf-header">
        <div class="perf-title-row">
          <div class="perf-icon-badge">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
          </div>
          <span class="perf-title">Performance</span>
        </div>
        <div class="perf-header-actions">
          <span class="perf-status-pill idle" id="perf-status-pill">
            <span class="perf-status-dot"></span>
            <span id="perf-status-label">Ready</span>
          </span>
          <button type="button" class="perf-close-btn" id="perf-close-btn" title="Collapse analytics sidebar" aria-label="Collapse analytics sidebar">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
      </div>

      <!-- Simple room speed and request flow -->
      <div class="perf-section">
        <div class="perf-sec-label">
          <span>LIVE ANSWER SPEED</span>
          <span class="perf-unit" id="perf-cur-model">—</span>
        </div>
        
        <div class="perf-hero-stat">
          <div class="perf-hero-main-row">
            <div class="perf-hero-val" id="perf-hero-tps">0.0</div>
          <div class="perf-hero-unit">text pieces / sec</div>
          </div>
          <div class="perf-hero-sub" id="perf-hero-sub">Start a local model to begin</div>
        </div>

        <div class="perf-chart-title">
          <span>Answer speed over time</span>
          <span class="perf-live-legend" aria-label="Chart series">
            <span class="perf-live-legend-item"><i class="perf-live-legend-swatch raw" aria-hidden="true"></i>Measured</span>
            <span class="perf-live-legend-item"><i class="perf-live-legend-swatch" aria-hidden="true"></i>5-sample average</span>
          </span>
        </div>
        <div class="perf-canvas-wrap" id="perf-live-canvas-wrap">
          <canvas id="perf-live-canvas" width="298" height="124" role="img" aria-label="Live answer speed over time, showing measured speed and a rolling five-sample average"></canvas>
          <div class="perf-canvas-empty" id="perf-live-empty">Your answer speed will appear here while it is being written.</div>
        </div>

        <section class="perf-viz-card" aria-labelledby="perf-device-viz-title">
          <div class="perf-viz-head">
            <div>
              <div class="perf-viz-title" id="perf-device-viz-title">Device contribution</div>
              <div class="perf-viz-subtitle" id="perf-device-viz-subtitle">GPU memory pledged by each device in this room</div>
            </div>
            <span class="perf-viz-badge" id="perf-device-viz-badge">1 DEVICE</span>
          </div>
          <div class="perf-device-canvas-wrap" id="perf-device-canvas-wrap">
            <canvas id="perf-device-canvas" aria-label="A chart of device contributions to model inference"></canvas>
            <div class="perf-viz-empty" id="perf-device-viz-empty" hidden></div>
          </div>
          <div class="perf-device-status-list" id="perf-device-status-list"></div>
        </section>

        <div class="perf-flow" id="perf-shard-flow">
          <div class="perf-flow-head">
            <span class="perf-flow-title">How your room answers</span>
            <span class="perf-flow-state" id="perf-flow-state">Waiting for local model</span>
          </div>
          <div class="perf-flow-track" id="perf-flow-track" aria-label="Your prompt passes through the devices helping to create an answer"></div>
          <div class="perf-flow-empty" id="perf-flow-empty">Start a local model. Compatible devices in the room can share the work.</div>
          <div class="perf-flow-note" id="perf-flow-note">Your prompt is handled by the model, then the answer is written and shown here.</div>
          <div class="perf-flow-readouts">
            <span class="perf-flow-metric">Answer speed<b id="perf-flow-speed">0.0 text pieces/s</b></span>
            <span class="perf-flow-metric">Devices helping<b id="perf-flow-devices">0</b></span>
            <span class="perf-flow-metric">Average if shared evenly<b id="perf-flow-efficiency">—</b></span>
          </div>
        </div>

        <div class="perf-grid">
          <div class="perf-stat-item">
            <span class="perf-stat-k">BEST SPEED</span>
            <span class="perf-stat-v" id="perf-stat-peak">0.0 text pieces/s</span>
          </div>
          <div class="perf-stat-item">
            <span class="perf-stat-k">PIECES OF TEXT</span>
            <span class="perf-stat-v" id="perf-stat-tokens">0</span>
          </div>
          <div class="perf-stat-item">
            <span class="perf-stat-k">FIRST RESPONSE</span>
            <span class="perf-stat-v" id="perf-stat-ttft">—</span>
          </div>
          <div class="perf-stat-item">
            <span class="perf-stat-k">TOTAL TIME</span>
            <span class="perf-stat-v" id="perf-stat-time">0.0s</span>
          </div>
        </div>

        <div class="perf-compare" aria-live="polite">
          <div class="perf-compare-title">Past speed by device setup</div>
          <div class="perf-compare-context">
            <span class="perf-compare-current" id="perf-compare-current-devices" role="status" aria-live="polite">Connected now: 1 device</span>
            <span>Chart and rows below show saved answers. They can include setups that are no longer online.</span>
          </div>
          <div class="perf-compare-model" id="perf-compare-model">Complete a local answer to start comparing.</div>
          <div class="perf-compare-canvas-wrap" id="perf-compare-canvas-wrap">
            <canvas id="perf-compare-canvas" role="img" aria-label="Historical average throughput and best answer peak by device setup"></canvas>
            <div class="perf-viz-empty" id="perf-compare-chart-empty">Complete a local answer with each device setup to build a speed comparison.</div>
          </div>
          <div class="perf-viz-legend" aria-hidden="true">
            <span class="perf-viz-legend-item"><i style="background:#2a45e0"></i>Average speed</span>
            <span class="perf-viz-legend-item"><i style="background:#159a78"></i>Best answer peak</span>
          </div>
          <div class="perf-compare-grid perf-compare-head" aria-hidden="true">
            <span>Setup</span><span>Avg. pieces/s</span><span>Best peak</span><span>First reply</span>
          </div>
          <div id="perf-compare-rows"></div>
          <div class="perf-compare-empty" id="perf-compare-empty">Your completed local answers will be saved on this browser.</div>
        </div>
      </div>
    `;

    roomScreen.appendChild(aside);
    this.container = aside;
    this.roomScreen = roomScreen;
    this.applySidebarWidth(this.getSavedSidebarWidth());

    // Check stored collapse state
    try {
      const stored = localStorage.getItem("swarm_perf_sidebar");
      this.setCollapsed(stored !== "open");
    } catch {}
  }

  setupListeners() {
    const closeBtn = document.getElementById("perf-close-btn");
    if (closeBtn) {
      closeBtn.onclick = () => this.toggleSidebar();
    }

    const chatTab = document.getElementById("room-chat-tab");
    const performanceTab = document.getElementById("room-performance-tab");
    if (chatTab) chatTab.onclick = () => this.setCollapsed(true);
    if (performanceTab) performanceTab.onclick = () => this.setCollapsed(false);

    this.setupSidebarResizing();

    window.addEventListener("resize", () => {
      this.resizeCanvases();
      this.renderScalingChart();
      this.renderLiveChart();
      this.renderDeviceShareChart();
      this.renderDeviceComparisonChart();
      this.renderSessionChart();
    });

    if (typeof ResizeObserver !== "undefined" && this.container) {
      const ro = new ResizeObserver(() => {
        this.resizeCanvases();
        this.renderScalingChart();
        this.renderLiveChart();
        this.renderDeviceShareChart();
        this.renderDeviceComparisonChart();
        this.renderSessionChart();
      });
      ro.observe(this.container);
    }
  }

  getSavedSidebarWidth() {
    try {
      const saved = Number(localStorage.getItem(PERF_SIDEBAR_WIDTH_KEY));
      if (Number.isFinite(saved) && saved > 0) return saved;
    } catch {}
    return PERF_SIDEBAR_DEFAULT_WIDTH;
  }

  clampSidebarWidth(width) {
    const maxWidth = Math.min(PERF_SIDEBAR_MAX_WIDTH, Math.floor(window.innerWidth * 0.48));
    return Math.round(Math.max(Math.min(PERF_SIDEBAR_MIN_WIDTH, maxWidth), Math.min(maxWidth, width)));
  }

  applySidebarWidth(width) {
    if (!this.roomScreen) return PERF_SIDEBAR_DEFAULT_WIDTH;
    const clamped = this.clampSidebarWidth(width);
    this.roomScreen.style.setProperty("--perf-sidebar-width", `${clamped}px`);
    const handle = document.getElementById("perf-resize-handle");
    if (handle) {
      handle.setAttribute("aria-valuenow", String(clamped));
      handle.setAttribute("aria-valuemax", String(Math.min(PERF_SIDEBAR_MAX_WIDTH, Math.floor(window.innerWidth * 0.48))));
    }
    return clamped;
  }

  saveSidebarWidth(width) {
    const clamped = this.applySidebarWidth(width);
    try { localStorage.setItem(PERF_SIDEBAR_WIDTH_KEY, String(clamped)); } catch {}
  }

  setupSidebarResizing() {
    const handle = document.getElementById("perf-resize-handle");
    if (!handle || !this.roomScreen) return;

    let drag = null;
    const finishDrag = (event) => {
      if (!drag || (event && event.pointerId !== drag.pointerId)) return;
      this.saveSidebarWidth(drag.width);
      drag = null;
      document.body.classList.remove("perf-sidebar-resizing");
      window.removeEventListener("pointermove", moveDrag);
      window.removeEventListener("pointerup", finishDrag);
      window.removeEventListener("pointercancel", finishDrag);
    };
    const moveDrag = (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      drag.width = this.applySidebarWidth(drag.startWidth - (event.clientX - drag.startX));
    };

    handle.addEventListener("pointerdown", (event) => {
      if (window.innerWidth <= 1200 || event.button !== 0) return;
      event.preventDefault();
      drag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startWidth: this.getSavedSidebarWidth(),
        width: this.getSavedSidebarWidth()
      };
      document.body.classList.add("perf-sidebar-resizing");
      try { handle.setPointerCapture(event.pointerId); } catch {}
      window.addEventListener("pointermove", moveDrag);
      window.addEventListener("pointerup", finishDrag);
      window.addEventListener("pointercancel", finishDrag);
    });

    handle.addEventListener("keydown", (event) => {
      if (window.innerWidth <= 1200) return;
      let width = this.getSavedSidebarWidth();
      if (event.key === "ArrowLeft") width += 16;
      else if (event.key === "ArrowRight") width -= 16;
      else if (event.key === "Home") width = PERF_SIDEBAR_MIN_WIDTH;
      else if (event.key === "End") width = PERF_SIDEBAR_MAX_WIDTH;
      else return;
      event.preventDefault();
      this.saveSidebarWidth(width);
    });
  }

  toggleSidebar() {
    if (!this.container) return;
    const isClosed = this.container.classList.contains("collapsed");
    this.setCollapsed(!isClosed);
  }

  setCollapsed(collapsed) {
    if (!this.container) return;
    this.roomScreen?.classList.toggle("room-performance-view", !collapsed);
    const btn = document.getElementById("topbar-perf-btn");
    const chatTab = document.getElementById("room-chat-tab");
    const performanceTab = document.getElementById("room-performance-tab");
    if (chatTab) {
      chatTab.classList.toggle("active", collapsed);
      chatTab.setAttribute("aria-pressed", String(collapsed));
    }
    if (performanceTab) {
      performanceTab.classList.toggle("active", !collapsed);
      performanceTab.setAttribute("aria-pressed", String(!collapsed));
    }
    if (collapsed) {
      this.container.classList.add("collapsed");
      if (btn) btn.classList.remove("active");
      try { localStorage.setItem("swarm_perf_sidebar", "closed"); } catch {}
    } else {
      this.container.classList.remove("collapsed");
      if (btn) btn.classList.add("active");
      try { localStorage.setItem("swarm_perf_sidebar", "open"); } catch {}
      // Force canvas refresh on expand
      setTimeout(() => {
        this.resizeCanvases();
        this.renderScalingChart();
        this.renderLiveChart();
        this.renderDeviceShareChart();
        this.renderDeviceComparisonChart();
        this.renderSessionChart();
      }, 150);
    }
  }

  setupCanvases() {
    this.liveCanvas = document.getElementById("perf-live-canvas");
    this.scalingCanvas = document.getElementById("perf-scaling-canvas");
    this.sessionCanvas = document.getElementById("perf-session-canvas");
    this.deviceCanvas = document.getElementById("perf-device-canvas");
    this.comparisonCanvas = document.getElementById("perf-compare-canvas");
    if (this.liveCanvas) this.liveCtx = this.liveCanvas.getContext("2d");
    if (this.scalingCanvas) this.scalingCtx = this.scalingCanvas.getContext("2d");
    if (this.sessionCanvas) this.sessionCtx = this.sessionCanvas.getContext("2d");
    this.resizeCanvases();
  }

  
  resizeCanvases() {
    if (this.liveChartInstance) this.liveChartInstance.resize();
    if (this.scalingChartInstance) this.scalingChartInstance.resize();
    if (this.sessionChartInstance) this.sessionChartInstance.resize();
    if (this.deviceShareChart) this.deviceShareChart.resize();
    if (this.deviceComparisonChart) this.deviceComparisonChart.resize();
  }

  updateDeviceListUI() {
    if (typeof document === "undefined") return;

    // 0. Update Hero Sub and Grid Badge
    const activeNodes = this.devices.filter(d => d.workerRole && d.workerRole !== "Idle").length;
    const isDistributed = activeNodes > 1;
    const heroSubEl = document.getElementById("perf-hero-sub");
    if (heroSubEl) {
      heroSubEl.textContent = this.backend === "cloud"
        ? "An online AI service is writing your answer"
        : isDistributed ? "Your devices are sharing the work" : activeNodes === 1 ? "This device is running the model" : "Start a local model to begin";
    }
    const gridShareBadgeEl = document.getElementById("perf-grid-share-badge");
    if (gridShareBadgeEl) {
      gridShareBadgeEl.textContent = activeNodes === 1 ? "1 ACTIVE" : `${activeNodes} ACTIVE`;
    }
    const activeBadgeEl = document.getElementById("perf-active-nodes-badge");
    if (activeBadgeEl) activeBadgeEl.textContent = activeNodes ? `${activeNodes} COMPUTING` : "NO LOCAL MODEL";

    // 1. The answer stream is sampled once for the whole room, so show one room-rate series.
    const legendEl = document.getElementById("perf-device-legend");
    if (legendEl) {
      if (this.devices.length > 1) {
        legendEl.innerHTML = `
          <span class="perf-legend-chip active" aria-label="Room output speed">
            <span class="perf-chip-dot" style="background:var(--accent);"></span>
            <span>Room output</span>
            <span class="perf-chip-speed" id="legend-speed-cluster">${this.peakTps.toFixed(1)} tok/s</span>
          </span>
        `;
      } else {
        legendEl.innerHTML = "";
      }
    }

    // 2. Grid Devices Processing Breakdown Cards
    const listEl = document.getElementById("perf-devices-list");
    if (listEl) {
      listEl.innerHTML = this.devices.map(d => {
        const gpuMeta = d.meta?.gpu || (d.meta?.webgpu ? "WebGPU" : "Mesh Node");
        const latMeta = d.rtt !== null ? `RTT: ${d.rtt}ms` : "Local Host";

        const workerText = d.workerRole === "Worker" ? "Model shard" : (d.workerRole === "Host" ? "Host · output sampler" : (d.stage || "Idle"));
        return `
          <div class="perf-dev-card ${this.isStreaming ? 'streaming' : ''}" id="dev-card-${escapeHtml(d.id)}">
            <div class="perf-dev-top">
              <div class="perf-dev-title-wrap">
                <span class="perf-dev-dot" style="background:${d.color};"></span>
                <span class="perf-dev-name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>
                <span class="perf-dev-stage-badge">${escapeHtml(workerText)}</span>
              </div>
              <div class="perf-dev-rates">
                <span class="perf-dev-toks" id="dev-role-${escapeHtml(d.id)}">${escapeHtml(d.layers || (d.workerRole === "Host" ? "Samples the shared output" : "Waiting for a model shard"))}</span>
              </div>
            </div>
            <div class="perf-dev-foot">
              <span>${escapeHtml(gpuMeta)}</span>
              <span>${escapeHtml(latMeta)}</span>
            </div>
          </div>
        `;
      }).join("");
    }
    this.renderShardFlow(this.lastInstantTps);
    this.renderDeviceShareChart();
  }

  renderShardFlow(speed = this.lastInstantTps) {
    if (typeof document === "undefined") return;
    const flow = document.getElementById("perf-shard-flow");
    const track = document.getElementById("perf-flow-track");
    const empty = document.getElementById("perf-flow-empty");
    const state = document.getElementById("perf-flow-state");
    const speedEl = document.getElementById("perf-flow-speed");
    const devicesEl = document.getElementById("perf-flow-devices");
    const efficiencyEl = document.getElementById("perf-flow-efficiency");
    const noteEl = document.getElementById("perf-flow-note");
    if (!flow || !track) return;

    const activeDevices = this.backend === "cloud"
      ? []
      : this.devices.filter(device => device.workerRole && device.workerRole !== "Idle").sort((a, b) => {
          const startLayer = device => {
            const match = String(device.layers || "").match(/layers?\s+(\d+)/i);
            return match ? Number(match[1]) : device.self ? -1 : Number.MAX_SAFE_INTEGER;
          };
          return startLayer(a) - startLayer(b);
        });
    const flowDevices = this.backend === "cloud"
      ? [{ name: "Online AI service", step: "Runs the model", color: "#8e8e93" }]
      : activeDevices.map((device, index) => ({
          ...device,
          step: `Part ${index + 1} of ${activeDevices.length}`
        }));
    const stages = flowDevices.length
      ? [{ type: "prompt", name: "Your prompt", step: "The question you asked" }, ...flowDevices.map(device => ({ type: "device", ...device })), { type: "answer", name: "Your answer", step: "Ready to read" }]
      : [];

    const signature = `${this.backend}|${stages.map(stage => `${stage.type}:${stage.id || stage.name}:${stage.step}`).join("|")}`;
    if (signature !== this.flowSignature) {
      track.innerHTML = stages.map((stage, index) => {
        const node = `<div class="perf-flow-node perf-flow-node--${stage.type}" style="--device-color:${escapeHtml(stage.color || DEVICE_COLORS[index % DEVICE_COLORS.length])}">
          <div class="perf-flow-node-top"><span class="perf-flow-node-dot"></span><span class="perf-flow-node-name">${escapeHtml(stage.name || "Device")}</span></div>
          <span class="perf-flow-node-step">${escapeHtml(stage.step || "Helping create your answer")}</span>
        </div>`;
        const link = index < stages.length - 1
          ? '<span class="perf-flow-link" aria-hidden="true"><i class="perf-flow-pulse"></i></span>'
          : "";
        return node + link;
      }).join("");
      this.flowSignature = signature;
    }

    const activeCount = activeDevices.length;
    if (empty) {
      empty.style.display = stages.length ? "none" : "block";
      empty.textContent = "Start a local model. Compatible devices in the room can share the work.";
    }
    if (state) {
      state.textContent = this.backend === "cloud"
        ? this.isStreaming ? "Online service is writing your answer…" : this.tokenCount > 0 ? "Your answer is ready" : "Online service is ready"
        : activeCount
          ? this.isStreaming
            ? "Working on your answer…"
            : this.tokenCount > 0 ? "Your answer is ready" : `${activeCount} ${activeCount === 1 ? "device is" : "devices are"} ready to help`
          : "Waiting for local model";
    }
    if (speedEl) speedEl.textContent = `${Math.max(0, Number(speed) || 0).toFixed(1)} text pieces/s`;
    if (devicesEl) devicesEl.textContent = String(activeCount);
    if (efficiencyEl) efficiencyEl.textContent = activeCount ? `${(Math.max(0, Number(speed) || 0) / activeCount).toFixed(1)} text pieces/s` : "—";
    if (noteEl) {
      noteEl.textContent = this.backend === "cloud"
        ? "This answer is being created online. Devices in your room are not helping with the calculation."
        : activeCount
          ? "Each device handles a different part of the AI model. The answer is shown here when the work is done. The last number is an estimate assuming the work is shared evenly."
          : "Choose a local model to let compatible devices in your room share the work.";
    }
    const animationSeconds = Math.min(1.8, Math.max(0.38, 2.4 / Math.max(1, Number(speed) || 1)));
    flow.style.setProperty("--perf-flow-duration", `${animationSeconds.toFixed(2)}s`);
    flow.classList.toggle("streaming", this.isStreaming && (activeCount > 0 || this.backend === "cloud"));
    flow.classList.toggle("cloud", this.backend === "cloud");
    if (this.backend === "cloud") {
      flow.setAttribute("aria-label", "Your prompt goes to an online AI service and the answer returns here");
    } else {
      flow.setAttribute("aria-label", activeCount
        ? `Your prompt passes through ${activeCount} ${activeCount === 1 ? "device" : "devices"} helping create the answer`
        : "Start a local model to see how devices share the work");
    }
  }

  /**
   * Fast dynamic update for live token numbers during streaming
   */
  updateDeviceReadouts(instantTps) {
    if (typeof document === "undefined") return;

    const clusterSpeedEl = document.getElementById("legend-speed-cluster");
    if (clusterSpeedEl) clusterSpeedEl.textContent = `${instantTps.toFixed(1)} pieces/s`;
  }

  updateSessionDevicesList() {
    if (typeof document === "undefined") return;
    const sessList = document.getElementById("perf-dev-sess-list");
    if (!sessList) return;

    if (this.devices.length > 1) {
      sessList.style.display = "flex";
      sessList.innerHTML = this.devices.map(d => {
        const tot = this.deviceSessionTotals.get(d.id) || 0;
        return `
          <div class="perf-dev-sess-row">
            <span>
              <span class="perf-dev-dot" style="background:${d.color};"></span>
              ${escapeHtml(d.name)}
            </span>
            <b>${tot.toLocaleString()} tok</b>
          </div>
        `;
      }).join("");
    } else {
      sessList.style.display = "none";
    }
  }

  /**
   * Render Live Stream Velocity Chart
   * Displays instantaneous tok/s curve over time with leading pulse ring,
   * showing per-device velocity traces when multiple devices are in the grid.
   */

  renderDeviceShareChart() {
    if (!this.deviceCanvas || typeof document === "undefined") return;
    const titleEl = document.getElementById("perf-device-viz-title");
    const subtitleEl = document.getElementById("perf-device-viz-subtitle");
    const badgeEl = document.getElementById("perf-device-viz-badge");
    const emptyEl = document.getElementById("perf-device-viz-empty");
    const statusList = document.getElementById("perf-device-status-list");
    const localDevices = this.devices || [];
    const assignedLayers = this.backend === "local" && localDevices.some(device => layerCount(device.layers) > 0);
    const metricLabel = assignedLayers ? "Model layers" : "GPU memory pledged";
    const metricUnit = assignedLayers ? "layers" : "GB";
    const values = localDevices.map(device => this.backend === "cloud" ? 0 : assignedLayers
      ? layerCount(device.layers)
      : Math.max(0, Number(device.meta?.contribGB ?? device.meta?.budgetGB ?? device.meta?.maxBufGB) || 0));
    const hasValues = values.some(value => value > 0);

    if (titleEl) titleEl.textContent = assignedLayers ? "Model layers by device" : "GPU memory by device";
    if (subtitleEl) subtitleEl.textContent = this.backend === "cloud"
      ? "Online inference · local devices are not used for this answer"
      : assignedLayers
      ? `${this.currentModel || "Current model"} · each bar shows the layers assigned to that device`
      : "Room capacity · start a local model to see how its layers are split";
    if (badgeEl) badgeEl.textContent = `${localDevices.length} ${localDevices.length === 1 ? "DEVICE" : "DEVICES"}`;
    if (emptyEl) {
      emptyEl.hidden = hasValues || this.backend === "cloud";
      emptyEl.textContent = "No GPU memory has been pledged yet. Compatible devices will appear here.";
      if (this.backend === "cloud") {
        emptyEl.hidden = false;
        emptyEl.textContent = "Online models run outside this room, so local devices do not contribute to this answer.";
      }
    }

    if (statusList) {
      statusList.innerHTML = localDevices.map(device => {
        const memory = Number(device.meta?.contribGB ?? device.meta?.budgetGB ?? device.meta?.maxBufGB) || 0;
        const network = device.rtt !== null && device.rtt !== undefined && Number.isFinite(Number(device.rtt))
          ? `${Math.round(Number(device.rtt))} ms response`
          : device.self ? "This device" : "Checking connection";
        const role = this.backend === "cloud"
          ? "Not used by online inference"
          : device.workerRole && device.workerRole !== "Idle"
            ? `${device.workerRole}${device.layers ? ` · ${device.layers}` : ""}`
            : "Connected · waiting for model";
        const meta = [role, memory > 0 ? `${memory.toFixed(memory % 1 ? 1 : 0)} GB pledged` : "No GPU pledge", network].join(" · ");
        return `<div class="perf-device-status">
          <i class="perf-device-status-dot" style="background:${device.color}"></i>
          <div class="perf-device-status-copy">
            <div class="perf-device-status-name">${escapeHtml(device.name || "Device")}${device.self ? " (you)" : ""}</div>
            <div class="perf-device-status-meta">${escapeHtml(meta)}</div>
          </div>
        </div>`;
      }).join("");
    }

    if (!this.deviceShareChart) {
      this.deviceShareChart = new Chart(this.deviceCanvas, {
        type: "bar",
        data: { labels: [], datasets: [] },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 220 },
          layout: { padding: { left: 4, right: 12, top: 4, bottom: 2 } },
          scales: {
            x: {
              beginAtZero: true,
              grace: "12%",
              grid: { color: PERF_CHART_GRID },
              border: { display: false },
              title: { display: true, text: metricUnit, color: PERF_CHART_TEXT, font: { size: 11 } },
              ticks: { color: PERF_CHART_TEXT, precision: 0, maxTicksLimit: 7, font: { size: 10 } },
            },
            y: {
              grid: { display: false },
              border: { display: false },
              ticks: { color: PERF_CHART_TEXT, font: { size: 11 }, padding: 8 },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              displayColors: false,
              callbacks: {
                label: context => {
                  const unit = context.chart.$metricUnit || "GB";
                  return `${Number(context.raw || 0).toFixed(unit === "GB" ? 1 : 0)} ${unit}`;
                },
              },
            },
          },
        },
        plugins: [{
          id: "shardxDeviceValueLabels",
          afterDatasetsDraw: chart => {
            const context = chart.ctx;
            const bars = chart.getDatasetMeta(0).data || [];
            const values = chart.data.datasets[0]?.data || [];
            const unit = chart.options.scales.x.title.text || "";
            context.save();
            context.font = "600 10px -apple-system, BlinkMacSystemFont, sans-serif";
            bars.forEach((bar, index) => {
              const value = Number(values[index]) || 0;
              if (!value) return;
              const label = `${value.toFixed(unit === "GB" ? 1 : 0)} ${unit}`;
              const inside = bar.width > context.measureText(label).width + 18;
              context.textAlign = inside ? "right" : "left";
              context.fillStyle = inside ? "#ffffff" : "#343846";
              context.fillText(label, inside ? bar.x - 8 : bar.x + 8, bar.y + 3.5);
            });
            context.restore();
          },
        }],
      });
    }

    const chart = this.deviceShareChart;
    chart.data.labels = localDevices.map(device => `${device.name || "Device"}${device.self ? " (you)" : ""}`);
    chart.data.datasets = [{
      label: metricLabel,
      data: values,
      backgroundColor: localDevices.map(device => `${device.color}C8`),
      borderColor: localDevices.map(device => device.color),
      borderWidth: 1,
      borderRadius: 7,
      borderSkipped: false,
      maxBarThickness: 28,
    }];
    chart.$metricUnit = metricUnit;
    chart.options.scales.x.title.text = metricUnit;
    chart.update();
  }

  renderDeviceComparisonChart(records = null) {
    if (!this.comparisonCanvas) return;
    const chartEmpty = document.getElementById("perf-compare-chart-empty");
    const allRecords = records || this.devicePerformance
      .filter(item => !this.currentModel || item.model === this.currentModel)
      .sort((a, b) => Number(a.deviceCount) - Number(b.deviceCount));
    if (!this.deviceComparisonChart) {
      this.deviceComparisonChart = new Chart(this.comparisonCanvas, {
        type: "bar",
        data: { labels: [], datasets: [] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 220 },
          interaction: { mode: "index", intersect: false },
          scales: {
            x: { grid: { display: false }, border: { display: false }, ticks: { color: PERF_CHART_TEXT, font: { size: 10 } } },
          y: {
              beginAtZero: true,
              grid: { color: PERF_CHART_GRID },
              border: { display: false },
              title: { display: true, text: "Text pieces / sec", color: PERF_CHART_TEXT, font: { size: 10 } },
              ticks: { color: PERF_CHART_TEXT, maxTicksLimit: 5, font: { size: 10 } },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              displayColors: true,
              callbacks: {
                title: contexts => {
                  const context = contexts[0];
                  const runs = context?.chart?.$runCounts?.[context.dataIndex] || 1;
                  return `${context?.label || "Device setup"} · ${runs} completed ${runs === 1 ? "answer" : "answers"}`;
                },
                label: context => `${context.dataset.label}: ${Number(context.raw || 0).toFixed(1)} pieces/s`,
              },
            },
          },
        },
      });
    }
    const counts = allRecords.map(item => Math.max(1, Math.floor(Number(item.deviceCount) || 1)));
    this.deviceComparisonChart.$runCounts = allRecords.map(item => Math.max(1, Math.floor(Number(item.runs) || 1)));
    const averageSpeeds = allRecords.map(item => Number(item.totalTokens) / Math.max(.001, Number(item.totalSeconds)));
    const bestSpeeds = allRecords.map(item => Number(item.bestSpeed) || 0);
    this.deviceComparisonChart.data.labels = counts.map(count => `${count}-device setup`);
    this.deviceComparisonChart.data.datasets = [
      { label: "Average speed", data: averageSpeeds, backgroundColor: "rgba(42,69,224,.76)", borderColor: "#2a45e0", borderWidth: 1, borderRadius: 6, maxBarThickness: 34 },
      { label: "Best speed", data: bestSpeeds, backgroundColor: "rgba(21,154,120,.72)", borderColor: "#159a78", borderWidth: 1, borderRadius: 6, maxBarThickness: 34 },
    ];
    this.deviceComparisonChart.update();
    if (chartEmpty) chartEmpty.hidden = allRecords.length > 0;
  }

  renderLiveChart() {
    if (!this.liveCtx || !this.liveCanvas) return;
    
    if (!this.liveChartInstance) {
      this.liveChartInstance = new Chart(this.liveCanvas, {
        type: 'line',
        data: {
          datasets: []
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          scales: {
            x: {
              type: 'linear', display: true,
              title: { display: true, text: 'Time (seconds)', color: PERF_CHART_TEXT, font: { size: 11 } },
              ticks: { color: PERF_CHART_TEXT, font: { size: 10 }, maxTicksLimit: 5 },
              grid: { display: false }
            },
          y: {
              beginAtZero: false,
              grace: "18%",
              title: { display: true, text: 'Text pieces per second', color: PERF_CHART_TEXT, font: { size: 11 } },
              ticks: { color: PERF_CHART_TEXT, font: { size: 10 }, maxTicksLimit: 4 },
              grid: { color: PERF_CHART_GRID }
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              mode: "index",
              intersect: false,
              callbacks: {
                title: contexts => `${Number(contexts[0]?.parsed.x || 0).toFixed(1)} seconds`,
                label: context => `${context.dataset.label}: ${Number(context.parsed.y || 0).toFixed(1)} pieces/s`,
                afterBody: contexts => {
                  const point = this.streamPoints[contexts[0]?.dataIndex];
                  return point ? `${Number(point.total || 0).toLocaleString()} pieces generated` : "";
                },
              },
            },
          },
          interaction: { mode: "index", intersect: false },
        }
      });
    }

    const gradient = this.liveCtx.createLinearGradient(0, 0, 0, this.liveCanvas.height || 220);
    gradient.addColorStop(0, "rgba(42, 69, 224, 0.22)");
    gradient.addColorStop(1, "rgba(42, 69, 224, 0.015)");
    const datasets = this.streamPoints.length >= 2 ? [
      {
        label: "Measured speed",
        data: this.streamPoints.map(p => ({ x: p.t, y: p.tps })),
        borderColor: "rgba(42, 69, 224, 0.38)",
        borderWidth: 1.25,
        fill: false,
        tension: 0.12,
        pointRadius: 0,
        pointHitRadius: 8,
        pointHoverRadius: 3,
      },
      {
        label: "5-sample average",
        data: this.streamPoints.map((point, index, points) => {
          if (index < 4) return { x: point.t, y: null };
          const start = Math.max(0, index - 4);
          const window = points.slice(start, index + 1);
          const average = window.reduce((sum, sample) => sum + sample.tps, 0) / window.length;
          return { x: point.t, y: average };
        }),
        borderColor: "#2a45e0",
        borderWidth: 2.5,
        fill: true,
        backgroundColor: gradient,
        tension: 0.28,
        pointRadius: 0,
        pointHitRadius: 10,
        pointHoverRadius: 4,
        pointHoverBackgroundColor: "#ffffff",
        pointHoverBorderColor: "#2a45e0",
        pointHoverBorderWidth: 2,
      },
    ] : [];

    this.liveChartInstance.data.datasets = datasets;
    this.liveChartInstance.update();

    if (typeof document !== 'undefined') {
      const emptyEl = document.getElementById('perf-live-empty');
      if (emptyEl) {
        emptyEl.style.display = datasets.length > 0 ? 'none' : 'flex';
      }
    }
  }

  renderScalingChart() {
    if (!this.scalingCtx || !this.scalingCanvas) return;
    
    if (!this.scalingChartInstance) {
      this.scalingChartInstance = new Chart(this.scalingCanvas, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          scales: {
            x: { grid: { display: false } },
            y: { beginAtZero: true, display: false }
          },
          plugins: { legend: { display: false } }
        }
      });
    }

    
      const labels = ["1 device", "2 devices", "3 devices", "4 devices", "5+ devices"];
      const speeds = [1, 2, 3, 4, 5].map(n => this.observedScaling.get(n) || 0);

      const activeNodes = this.devices.filter(d => d.workerRole && d.workerRole !== "Idle").length;
      const currentNodes = this.backend === "cloud" ? 0 : activeNodes;
      const bgColors = [1, 2, 3, 4, 5].map(n => {
        const isActive = (n === currentNodes) || (n === 5 && currentNodes >= 5);
        return isActive ? "rgba(10, 132, 255, 0.66)" : "rgba(165, 169, 177, 0.2)";
      });

      const borderColors = [1, 2, 3, 4, 5].map(n => {
        const isActive = (n === currentNodes) || (n === 5 && currentNodes >= 5);
        return isActive ? "#0a84ff" : "transparent";
      });

      if (this.scalingChartInstance && this.scalingChartInstance.data) {
        this.scalingChartInstance.data.labels = labels;
        this.scalingChartInstance.data.datasets = [
          {
            type: 'bar',
            label: 'Peak Speed (tok/s)',
            data: speeds,
            backgroundColor: bgColors,
            borderColor: borderColors,
            borderWidth: 1.5,
            borderRadius: 4
          }
        ];
        try { this.scalingChartInstance.update(); } catch (e) { console.warn("Chart update failed", e); }
      }
      this.updateScalingLegend(currentNodes);

  }
  updateScalingLegend(currentNodes = this.backend === "cloud" ? 0 : this.devices.filter(d => d.workerRole && d.workerRole !== "Idle").length) {
    if (typeof document === "undefined") return;
    const legend = document.getElementById("perf-scaling-legend");
    if (!legend) return;

    
    legend.innerHTML = [1, 2, 3, 4, 5].map(n => {
      const isActive = (n === currentNodes) || (n === 5 && currentNodes >= 5);
      const label = n === 5 ? "5+ devices" : n + " device" + (n > 1 ? "s" : "");
      const speed = this.observedScaling.get(n);
      const speedText = speed ? speed.toFixed(1) + " tok/s" : "— waiting for benchmark";
      return `
        <div class="perf-scaling-row ${isActive ? 'active' : ''}">
          <span class="scaling-node-label">
            ${label}
            ${isActive ? '<span class="scaling-active-tag">CURRENT</span>' : ''}
          </span>
          <span class="scaling-node-mult">${speedText}</span>
        </div>
      `;
    }).join("");

  }

  renderSessionChart() {
    if (!this.sessionCanvas) return;
    const history = this.sessionHistory.filter((item) => Number(item.deviceCount) > 0);
    const grouped = new Map();
    history.forEach((item) => {
      const count = Math.max(1, Math.floor(Number(item.deviceCount) || 1));
      if (!grouped.has(count)) grouped.set(count, []);
      grouped.get(count).push(item);
    });

    const metrics = [
      { label: "Avg tok/s", value: (items) => items.reduce((sum, item) => sum + (Number(item.tokensPerSecond) || 0), 0) / items.length },
      { label: "Peak tok/s", value: (items) => items.reduce((sum, item) => sum + (Number(item.peakTokensPerSecond) || 0), 0) / items.length },
      { label: "Avg tokens", value: (items) => items.reduce((sum, item) => sum + (Number(item.tokens) || 0), 0) / items.length },
      { label: "Chats", value: (items) => items.length },
    ];
    const summaries = [...grouped.entries()].sort(([a], [b]) => a - b).map(([count, items]) => ({
      count,
      items,
      values: metrics.map((metric) => metric.value(items)),
    }));
    const maxima = metrics.map((_, index) => Math.max(0.001, ...summaries.map((summary) => summary.values[index])));

    if (!this.sessionChartInstance) {
      this.sessionChartInstance = new Chart(this.sessionCanvas, {
        type: "radar",
        data: { labels: metrics.map((metric) => metric.label), datasets: [] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 180 },
          scales: {
            r: {
              beginAtZero: true,
              min: 0,
              max: 100,
              ticks: { display: false, stepSize: 25 },
              grid: { color: PERF_CHART_GRID },
              angleLines: { color: PERF_CHART_GRID },
              pointLabels: { color: PERF_CHART_TEXT, font: { size: 10 } },
            },
          },
          plugins: {
            legend: { display: true, position: "bottom", labels: { color: PERF_CHART_TEXT, boxWidth: 9, padding: 8, font: { size: 10 } } },
            tooltip: {
              callbacks: {
                label: (context) => {
                  const value = context.dataset.metricValues?.[context.dataIndex];
                  return `${context.dataset.label}: ${Number(value || 0).toFixed(1)} ${metrics[context.dataIndex].label}`;
                },
              },
            },
          },
        },
      });
    }

    const datasets = summaries.map((summary) => {
      const color = DEVICE_COLORS[(summary.count - 1) % DEVICE_COLORS.length];
      return {
        label: `${summary.count} ${summary.count === 1 ? "device" : "devices"}`,
        data: summary.values.map((value, index) => (value / maxima[index]) * 100),
        metricValues: summary.values,
        borderColor: color,
        backgroundColor: `${color}22`,
        pointBackgroundColor: color,
        pointBorderColor: color,
        pointRadius: 2.5,
        borderWidth: 2,
      };
    });
    this.sessionChartInstance.data.labels = metrics.map((metric) => metric.label);
    this.sessionChartInstance.data.datasets = datasets;
    this.sessionChartInstance.update();

    const empty = document.getElementById("perf-session-empty");
    if (empty) empty.style.display = history.length ? "none" : "flex";
    const sessionTokens = document.getElementById("perf-sess-tokens");
    if (sessionTokens) sessionTokens.innerText = `${this.sessionTokens.toLocaleString()} tok`;
    const sessionSpeed = document.getElementById("perf-sess-avg-tps");
    if (sessionSpeed) sessionSpeed.innerText = this.totalGenerationTime > 0
      ? `${(this.sessionTokens / this.totalGenerationTime).toFixed(1)} tok/s`
      : "— tok/s";
    const list = document.getElementById("perf-session-history");
    if (list) {
      const recent = [...history].slice(-5).reverse();
      list.innerHTML = recent.length ? recent.map((item) => {
        const count = Math.max(1, Math.floor(Number(item.deviceCount) || 1));
        const color = DEVICE_COLORS[(count - 1) % DEVICE_COLORS.length];
        const time = new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return `<div class="perf-history-row">
          <div class="perf-history-main"><i class="perf-history-dot" style="background:${color}"></i><b>${count} ${count === 1 ? "device" : "devices"}</b><span>· ${Number(item.tokens) || 0} tokens</span></div>
          <span>${time}</span>
          <div class="perf-history-sub"><span>${(Number(item.tokensPerSecond) || 0).toFixed(1)} tok/s</span><span>·</span><span>${(Number(item.tokensPerMinute) || 0).toFixed(0)} tok/min</span><span>·</span><span>${escapeHtml(item.model || "Model")}</span></div>
        </div>`;
      }).join("") : '<div class="perf-history-empty">Completed chats will appear here with speed and device count.</div>';
    }
  }

  renderDevicePerformance() {
    if (typeof document === "undefined") return;
    const rowsEl = document.getElementById("perf-compare-rows");
    const emptyEl = document.getElementById("perf-compare-empty");
    const modelEl = document.getElementById("perf-compare-model");
    const currentDevicesEl = document.getElementById("perf-compare-current-devices");
    if (!rowsEl || !emptyEl || !modelEl) return;

    const onlineCount = this.devices.length;
    if (currentDevicesEl) {
      currentDevicesEl.textContent = onlineCount
        ? `Connected now: ${onlineCount} ${onlineCount === 1 ? "device" : "devices"}`
        : "Connected now: no devices";
    }

    const newest = [...this.devicePerformance].sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")))[0];
    const model = this.currentModel || newest?.model || "";
    modelEl.textContent = model
      ? `${model} · completed local answers; prompts and device load can affect speed.`
      : "Complete a local answer to start comparing.";
    const records = this.devicePerformance
      .filter(item => item.model === model)
      .sort((a, b) => Number(a.deviceCount) - Number(b.deviceCount));
    this.renderDeviceComparisonChart(records);
    const baseline = records.find(item => Number(item.deviceCount) === 1);

    rowsEl.innerHTML = records.map(item => {
      const count = Math.max(1, Math.floor(Number(item.deviceCount) || 1));
      const runs = Math.max(1, Math.floor(Number(item.runs) || 1));
      const speed = Number(item.totalTokens) / Math.max(0.001, Number(item.totalSeconds));
      const peak = Number(item.bestSpeed) || 0;
      const firstReply = Number(item.firstReplyCount) > 0
        ? `${(Number(item.firstReplyTotal) / Number(item.firstReplyCount)).toFixed(1)}s`
        : "—";
      let comparison = "1-device baseline";
      if (count !== 1 && baseline) {
        const baseSpeed = Number(baseline.totalTokens) / Math.max(0.001, Number(baseline.totalSeconds));
        if (baseSpeed > 0) {
          const change = ((speed / baseSpeed) - 1) * 100;
          comparison = `${change > 0 ? "+" : ""}${change.toFixed(0)}% vs 1`;
        }
      } else if (count !== 1) {
        comparison = "run once on 1 device";
      }
      return `<div class="perf-compare-grid perf-compare-row">
        <span class="perf-compare-count">${count}-device setup<small class="perf-compare-detail">${runs} ${runs === 1 ? "saved answer" : "saved answers"} · ${escapeHtml(comparison)}</small></span>
        <span class="perf-compare-value">${Number.isFinite(speed) ? speed.toFixed(1) : "0.0"}/s</span>
        <span class="perf-compare-value">${peak.toFixed(1)}/s</span>
        <span class="perf-compare-value">${firstReply}</span>
      </div>`;
    }).join("");
    emptyEl.style.display = records.length ? "none" : "block";
    if (!records.length) emptyEl.textContent = "Your completed local answers will be saved on this browser.";
  }

  recordDevicePerformance({ totalTokens, totalSecs }) {
    if (this.backend !== "local" || !this.currentModel) return;
    const tokens = Math.max(0, Number(totalTokens) || 0);
    const seconds = Math.max(0, Number(totalSecs) || 0);
    const deviceCount = Math.max(1, Math.floor(Number(this.currentDeviceCount) || 1));
    if (!tokens || !seconds) return;

    let record = this.devicePerformance.find(item => item.model === this.currentModel && Number(item.deviceCount) === deviceCount);
    if (!record) {
      record = { model: this.currentModel, deviceCount, runs: 0, totalTokens: 0, totalSeconds: 0, bestSpeed: 0, firstReplyTotal: 0, firstReplyCount: 0 };
      this.devicePerformance.push(record);
    }
    record.runs = (Number(record.runs) || 0) + 1;
    record.totalTokens = (Number(record.totalTokens) || 0) + tokens;
    record.totalSeconds = (Number(record.totalSeconds) || 0) + seconds;
    record.bestSpeed = Math.max(Number(record.bestSpeed) || 0, Number(this.peakTps) || 0);
    if (this.firstTokenTime !== null && this.genStartTime) {
      record.firstReplyTotal = (Number(record.firstReplyTotal) || 0) + Math.max(0, (this.firstTokenTime - this.genStartTime) / 1000);
      record.firstReplyCount = (Number(record.firstReplyCount) || 0) + 1;
    }
    record.updatedAt = new Date().toISOString();
    this.devicePerformance = this.devicePerformance
      .sort((a, b) => String(a.updatedAt || "").localeCompare(String(b.updatedAt || "")))
      .slice(-200);
    try { localStorage.setItem(this.devicePerformanceStorageKey, JSON.stringify(this.devicePerformance)); } catch {}
    this.renderDevicePerformance();
  }

  saveSessionRecord(opts = {}) {
    const totalTokens = Math.max(0, Number(opts.totalTokens ?? this.tokenCount) || 0);
    const totalSecs = Math.max(0, Number(opts.totalSecs ?? ((performance.now() - this.genStartTime) / 1000)) || 0);
    const failed = String(opts.stats || "").toLowerCase().startsWith("failed:");
    if (failed) return;
    this.recordDevicePerformance({ totalTokens, totalSecs });
    const tokensPerSecond = totalSecs > 0 ? totalTokens / totalSecs : 0;
    this.sessionHistory.push({
      timestamp: new Date().toISOString(),
      model: this.currentModel,
      deviceCount: this.currentDeviceCount || this.clusterSize || 1,
      tokens: totalTokens,
      seconds: totalSecs,
      tokensPerSecond,
      tokensPerMinute: tokensPerSecond * 60,
      peakTokensPerSecond: this.peakTps,
    });
    this.sessionHistory = this.sessionHistory.slice(-60);
    try { sessionStorage.setItem(this.sessionStorageKey, JSON.stringify(this.sessionHistory)); } catch {}
    this.renderSessionChart();
  }

  onGenStart(opts) {
    try {
      this.isStreaming = true;
      const nextModel = opts.model || '';
      if (this.currentModel && nextModel && nextModel !== this.currentModel) this.observedScaling.clear();
      this.currentModel = nextModel;
      this.renderDevicePerformance();
      this.backend = opts.backend || 'local';
      this.renderDeviceShareChart();
      this.currentDeviceCount = Math.max(1, Number(opts.deviceCount) || this.clusterSize || this.devices.length || 1);
      const statusPill = typeof document !== 'undefined' ? document.getElementById('perf-status-pill') : null;
      const statusLabel = typeof document !== 'undefined' ? document.getElementById('perf-status-label') : null;
      if (statusPill) statusPill.classList.add('streaming');
      if (statusLabel) statusLabel.textContent = 'Writing';
      const heroSub = typeof document !== 'undefined' ? document.getElementById('perf-hero-sub') : null;
      if (heroSub) {
        heroSub.textContent = this.backend === 'cloud'
          ? 'An online AI service is writing your answer'
          : this.currentDeviceCount > 1
            ? 'Your devices are sharing the work'
            : 'This device is running the model';
      }
      this.genStartTime = performance.now();
      this.firstTokenTime = null;
      this.tokenCount = 0;
      this.peakTps = 0;
      this.streamPoints = [];
      this.recentTokenTimes = [];
      this.lastInstantTps = 0;
      this.renderShardFlow(0);
      
      if (typeof document !== 'undefined') {
        const topBtn = document.getElementById('topbar-perf-btn');
        if (topBtn) topBtn.classList.add('streaming');
        
        const curMod = document.getElementById('perf-cur-model');
        if (curMod) {
          curMod.innerText = this.currentModel;
        }
        
        const ttftEl = document.getElementById('perf-stat-ttft');
        if (ttftEl) ttftEl.innerText = '—';
        const tokEl = document.getElementById('perf-stat-tokens');
        if (tokEl) tokEl.innerText = '0';
        const timeEl = document.getElementById('perf-stat-time');
        if (timeEl) timeEl.innerText = '0.0s';
        const heroTps = document.getElementById('perf-hero-tps');
        if (heroTps) heroTps.innerText = '0.0';
        const peakEl = document.getElementById('perf-stat-peak');
        if (peakEl) peakEl.innerText = '0.0 text pieces/s';
      }
      this.renderScalingChart();
    } catch (e) {
      console.warn("perfSidebar.onGenStart failed:", e);
    }
  }

  onToken(text, totalCount) {
    try {
      if (!this.isStreaming) return;
      const now = performance.now();
      
      if (this.firstTokenTime === null) {
        this.firstTokenTime = now;
        const ttft = now - this.genStartTime;
        if (typeof document !== 'undefined') {
          const ttftEl = document.getElementById('perf-stat-ttft');
          if (ttftEl) ttftEl.innerText = `${(ttft / 1000).toFixed(1)}s`;
        }
      }
      
      this.tokenCount = totalCount || (this.tokenCount + 1);
      this.recentTokenTimes.push(now);
      
      while (this.recentTokenTimes.length > 0 && now - this.recentTokenTimes[0] > 1000) {
        this.recentTokenTimes.shift();
      }
      
      const elapsedSec = (now - this.genStartTime) / 1000;
      
      let instantTps = 0;
      if (this.recentTokenTimes.length > 1) {
        const windowStart = this.recentTokenTimes[0];
        const windowEnd = this.recentTokenTimes[this.recentTokenTimes.length - 1];
        const windowSpanSec = (windowEnd - windowStart) / 1000;
        instantTps = (this.recentTokenTimes.length - 1) / Math.max(windowSpanSec, 0.05);
      }
      
      if (instantTps > this.peakTps) {
        this.peakTps = instantTps;
        if (typeof document !== 'undefined') {
          const peakEl = document.getElementById('perf-stat-peak');
          if (peakEl) peakEl.innerText = this.peakTps.toFixed(1) + ' text pieces/s';
        }
        
        const bucket = Math.min(5, Math.max(1, this.currentDeviceCount || 1));
        const currentRecord = this.observedScaling.get(bucket) || 0;
        if (this.backend === "local" && this.peakTps > currentRecord) {
          this.observedScaling.set(bucket, this.peakTps);
          this.renderScalingChart();
        }
      }
      
      if (instantTps > 0) this.streamPoints.push({ t: elapsedSec, tps: instantTps, total: this.tokenCount });
      this.lastInstantTps = instantTps;
      
      if (typeof document !== 'undefined') {
        const topRate = document.getElementById('topbar-perf-rate');
        if (topRate) topRate.innerText = instantTps.toFixed(1) + ' text pieces/s';
        const heroTps = document.getElementById('perf-hero-tps');
        if (heroTps) heroTps.innerText = instantTps.toFixed(1);
        
        const tokEl = document.getElementById('perf-stat-tokens');
        if (tokEl) tokEl.innerText = this.tokenCount.toLocaleString();
        
        const timeEl = document.getElementById('perf-stat-time');
        if (timeEl) timeEl.innerText = elapsedSec.toFixed(1) + 's';
      }
      
      this.updateDeviceReadouts(instantTps);
      this.renderShardFlow(instantTps);
      this.renderLiveChart();
    } catch (e) {
      console.warn("perfSidebar.onToken failed:", e);
    }
  }

  onGenDone(opts) {
    try {
      this.isStreaming = false;
      this.saveSessionRecord(opts || {});
      if (typeof document !== 'undefined') {
        const topBtn = document.getElementById('topbar-perf-btn');
        if (topBtn) topBtn.classList.remove('streaming');
        const statusPill = document.getElementById('perf-status-pill');
        const statusLabel = document.getElementById('perf-status-label');
        if (statusPill) statusPill.classList.remove('streaming');
        if (statusLabel) statusLabel.textContent = 'Ready';
        
        if (opts && opts.totalTokens !== undefined) {
          this.tokenCount = opts.totalTokens;
          const tokEl = document.getElementById('perf-stat-tokens');
          if (tokEl) tokEl.innerText = this.tokenCount.toLocaleString();
        }
        if (opts && opts.totalSecs !== undefined) {
          const timeEl = document.getElementById('perf-stat-time');
          if (timeEl) timeEl.innerText = opts.totalSecs.toFixed(1) + 's';
        }
      }
      this.sessionTokens += this.tokenCount;
      this.promptCount++;
      if (opts && opts.totalSecs) {
        this.totalGenerationTime += opts.totalSecs;
      }
      if (typeof document !== 'undefined') {
        const sessTokensEl = document.getElementById('perf-sess-tokens');
        if (sessTokensEl) sessTokensEl.innerText = this.sessionTokens.toLocaleString() + ' tok';
        const sessAvgEl = document.getElementById('perf-sess-avg-tps');
        if (sessAvgEl && this.totalGenerationTime > 0) {
          const avg = this.sessionTokens / this.totalGenerationTime;
          sessAvgEl.innerText = avg.toFixed(1) + ' tok/s';
        }
        const sessDevicesEl = document.getElementById('perf-sess-devices');
        if (sessDevicesEl) {
          const activeNodes = this.devices.filter(d => d.workerRole && d.workerRole !== "Idle").length;
          const displayCount = activeNodes > 0 ? activeNodes : (this.currentDeviceCount || 1);
          sessDevicesEl.innerText = displayCount === 1 ? '1 Device' : `${displayCount} Devices`;
        }
      }
      this.renderLiveChart();
      this.renderShardFlow(this.lastInstantTps);
    } catch (e) {
      console.warn("perfSidebar.onGenDone failed:", e);
    }
  }
}

function layerCount(rangeLabel) {
  const text = String(rangeLabel || "");
  if (/\b0\s+layers?\b/i.test(text) || /embed\/head only/i.test(text)) return 0;
  const match = text.match(/\blayers?\s+(\d+)(?:\s*[–—-]\s*(\d+))?/i);
  if (!match) return 0;
  const start = Number(match[1]);
  const end = match[2] ? Number(match[2]) : start;
  return Math.max(0, end - start + 1);
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Global instance export
export const perfSidebar = new PerfSidebar();



