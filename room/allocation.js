// webslice layer allocation and memory pledge planner
// Handles capacity calculations, layer slicing, zero-layer prevention, and layer range string formatting.

export function pledgeOf(m) {
  if (!m) return 0;
  const gb = m.contribGB ?? (m.maxBufGB ? m.maxBufGB * 0.5 : 0.5);
  return Math.max(0, gb || 0) * (2 ** 30);
}

export function calculateClusterPledge(hostMeta, peerMetas = []) {
  const all = [hostMeta, ...peerMetas].filter(Boolean);
  return all.reduce((acc, m) => acc + Math.max(0, m.contribGB || 0), 0);
}

export function formatLayerRange(range, isHost = false) {
  if (!range || range.length < 2) return "";
  const [lo, hi] = range;
  if (lo >= hi) return isHost ? "embed/head only" : "0 layers";
  if (hi - lo === 1) return `layer ${lo}`;
  return `layers ${lo}–${hi - 1}`;
}

export function allocateLayers(L, layerBytes, embedBytes, hostMeta, peerMetas = []) {
  const hostPledge = pledgeOf(hostMeta);
  const parts = [
    { cap: Math.max(hostPledge - embedBytes, layerBytes / 2) },
    ...peerMetas.map((m) => ({ cap: Math.max(pledgeOf(m), layerBytes / 2) })),
  ];
  const totalCap = parts.reduce((s, p) => s + p.cap, 0);
  const assigned = parts.map((p) => Math.floor((L * p.cap) / totalCap));
  const fracs = parts.map((p, i) => ({ i, f: (L * p.cap) / totalCap - assigned[i] })).sort((a, b) => b.f - a.f);
  let rem = L - assigned.reduce((a, b) => a + b, 0);
  for (let k = 0; k < rem; k++) assigned[fracs[k % fracs.length].i]++;

  // Ensure every participant (host and peers) gets >= 1 layer when L >= assigned.length
  if (L >= assigned.length) {
    for (let i = 0; i < assigned.length; i++) {
      if (assigned[i] === 0) {
        let maxVal = -1;
        let maxIdx = -1;
        for (let j = 0; j < assigned.length; j++) {
          if (assigned[j] > maxVal && assigned[j] > 1) {
            maxVal = assigned[j];
            maxIdx = j;
          }
        }
        if (maxIdx >= 0) {
          assigned[maxIdx]--;
          assigned[i]++;
        }
      }
    }
  }

  const ranges = [];
  let acc = 0;
  for (const a of assigned) {
    ranges.push([acc, acc + a]);
    acc += a;
  }

  const needGB = (L * layerBytes + embedBytes) / (2 ** 30);
  const haveGB = parts.reduce((s, p) => s + p.cap, embedBytes) / (2 ** 30);

  return { assigned, ranges, needGB, haveGB };
}
