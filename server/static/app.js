/* Control room.
 *
 * Polls the API on a cursor and paints two views of the same data: the map on
 * top, the database table below. Polling rather than SSE on purpose -- it is
 * cheap, debuggable, and survives the server restarting mid-demo, which SSE
 * does not without reconnect logic. It looks identical to the audience.
 */

const POLL_MS = 2000;
const CHENNAI = [13.03, 80.24];

const state = {
  layer: "defects",        // which view the map and table are showing
  taxonomy: {},            // key -> {label, hex, severity}
  defects: new Map(),      // id -> row
  events: new Map(),
  lastEventId: 0,          // cursor into /api/events
  markers: new Map(),      // "kind:id" -> leaflet layer
  selected: null,
  follow: true,
  heat: false,
  filters: { type: "", status: "" },
  firstLoad: true,
};

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) => (n === null || n === undefined ? "—" : Number(n).toFixed(d));

/* ------------------------------------------------------------------ map */

// SVG rendering, not canvas: the arrival ping animates a CSS class on the
// marker's `_path` element, which canvas-rendered markers do not have. At
// prototype scale (hundreds of defects) SVG costs nothing.
const map = L.map("map", { zoomControl: true, preferCanvas: false }).setView(CHENNAI, 13);

// Plain OSM tiles, darkened in CSS (see .leaflet-tile-pane). CARTO's dark
// basemap now watermarks "API KEY REQUIRED" across every tile, and a demo
// should not depend on a key. The filter only touches the tile pane, so
// marker and route colours stay true.
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 19,
}).addTo(map);

const routeLayer = L.layerGroup().addTo(map);
const markerLayer = L.layerGroup().addTo(map);
const heatLayer = L.layerGroup();

/* ---------------------------------------------------------------- helpers */

function colourOf(key) {
  return (state.taxonomy[key] && state.taxonomy[key].hex) || "#8b98a9";
}
function labelOf(key) {
  return (state.taxonomy[key] && state.taxonomy[key].label) || key;
}

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

function setConn(ok, note) {
  const el = $("conn");
  el.textContent = note;
  el.className = "pill " + (ok ? "pill-live" : "pill-down");
}

/* ----------------------------------------------------------------- markers */

function radiusFor(row) {
  // Severity drives size, sightings nudge it -- a pothole four buses have
  // reported should read louder than a one-off crack.
  const sev = row.severity || 0.3;
  const seen = state.layer === "defects" ? Math.min(row.sightings || 1, 6) : 1;
  return 5 + sev * 6 + seen * 0.9;
}

function styleFor(row) {
  const c = colourOf(row.damage_type);
  const unconfirmed = state.layer === "defects" && row.status === "unconfirmed";
  const repaired = row.status === "repaired";
  return {
    radius: radiusFor(row),
    color: repaired ? "#3fb950" : unconfirmed ? "#6b7686" : c,
    weight: repaired ? 1.5 : unconfirmed ? 1.5 : 2,
    opacity: repaired ? 0.65 : 1,
    fillColor: repaired ? "transparent" : unconfirmed ? "#39424f" : c,
    fillOpacity: repaired ? 0 : unconfirmed ? 0.35 : 0.62,
    dashArray: repaired ? "3,3" : null,
  };
}

function upsertMarker(row, kind, isNew) {
  if (row.lat == null || row.lon == null) return;
  const key = `${kind}:${row.id}`;
  let m = state.markers.get(key);
  if (m) {
    m.setStyle(styleFor(row));
    m.setLatLng([row.lat, row.lon]);
  } else {
    m = L.circleMarker([row.lat, row.lon], styleFor(row));
    m.on("click", () => select(row, kind));
    m.addTo(markerLayer);
    state.markers.set(key, m);
  }
  m.bindTooltip(
    `${labelOf(row.damage_type)} · ${fmt((row.max_confidence ?? row.confidence) * 100)}%` +
      (kind === "defects" ? ` · ${row.sightings || 0} sighting(s)` : ` · ${row.bus_id}`),
    { direction: "top", offset: [0, -4] }
  );

  if (isNew && !state.firstLoad) {
    // A brief expanding ring so the eye catches what just arrived.
    const ring = L.circleMarker([row.lat, row.lon], {
      radius: radiusFor(row),
      color: colourOf(row.damage_type),
      fillOpacity: 0,
      weight: 2,
    }).addTo(markerLayer);
    if (ring._path) ring._path.classList.add("ping");
    setTimeout(() => markerLayer.removeLayer(ring), 1500);
    if (state.follow) map.panTo([row.lat, row.lon], { animate: true, duration: 0.6 });
  }
}

function rebuildMarkers() {
  markerLayer.clearLayers();
  state.markers.clear();
  for (const row of visibleRows()) upsertMarker(row, state.layer, false);
  drawHeat();
}

function drawHeat() {
  heatLayer.clearLayers();
  if (!state.heat) { map.removeLayer(heatLayer); return; }
  for (const row of visibleRows()) {
    if (row.lat == null) continue;
    // Severity-weighted blur. Not a true kernel-density heatmap, but honest
    // about what it shows: where the weighted damage is.
    L.circle([row.lat, row.lon], {
      radius: 40 + (row.severity || 0.3) * 90,
      stroke: false,
      fillColor: colourOf(row.damage_type),
      fillOpacity: 0.16,
    }).addTo(heatLayer);
  }
  heatLayer.addTo(map);
}

/* ------------------------------------------------------------------ table */

const COLUMNS = {
  defects: [
    ["id", "id", "num"],
    ["damage_type", "damage", "tag"],
    ["status", "status", "status"],
    ["sightings", "sightings", "num"],
    ["distinct_buses", "buses", "num"],
    ["max_confidence", "conf", "pct"],
    ["peak_area_pct", "size %", "num2"],
    ["lat", "lat", "coord"],
    ["lon", "lon", "coord"],
    ["route_id", "route", ""],
    ["first_seen", "first seen", "time"],
    ["last_seen", "last seen", "time"],
  ],
  events: [
    ["id", "id", "num"],
    ["damage_type", "damage", "tag"],
    ["bus_id", "bus", ""],
    ["route_id", "route", ""],
    ["confidence", "conf", "pct"],
    ["area_pct_frame", "size %", "num2"],
    ["lat", "lat", "coord"],
    ["lon", "lon", "coord"],
    ["frames_seen", "frames", "num"],
    ["captured_at", "captured", "time"],
    ["defect_id", "defect", "num"],
    ["source_clip", "clip", ""],
  ],
};

function cell(row, key, kind) {
  const v = row[key];
  if (v === null || v === undefined || v === "") return '<td class="num">—</td>';
  switch (kind) {
    case "tag":
      return `<td><span class="tag"><i style="background:${colourOf(v)}"></i>${labelOf(v)}</span></td>`;
    case "status":
      return `<td><span class="tag st-${v}">${v}</span></td>`;
    case "pct":
      return `<td class="num">${fmt(v * 100)}%</td>`;
    case "num":
      return `<td class="num">${v}</td>`;
    case "num2":
      return `<td class="num">${fmt(v, 2)}</td>`;
    case "coord":
      return `<td class="num">${fmt(v, 5)}</td>`;
    case "time":
      return `<td>${String(v).replace("T", " ").slice(0, 19)}</td>`;
    default:
      return `<td>${v}</td>`;
  }
}

function emptyMessage() {
  const filtered = state.filters.type || state.filters.status;
  if (filtered) return "No rows match the current filters.";
  if (state.layer === "defects" && state.events.size)
    return "No clustered defects yet — showing nothing here until sightings are grouped. Switch to Sightings for the raw log.";
  return "Nothing reported yet. Start a pass in the onboard window.";
}

function visibleRows() {
  const src = state.layer === "defects" ? state.defects : state.events;
  let rows = [...src.values()];
  if (state.filters.type) rows = rows.filter((r) => r.damage_type === state.filters.type);
  if (state.filters.status && state.layer === "defects")
    rows = rows.filter((r) => r.status === state.filters.status);
  return rows.sort((a, b) => b.id - a.id);
}

function renderTable(newIds = new Set()) {
  const cols = COLUMNS[state.layer];
  $("thead").innerHTML = "<tr>" + cols.map((c) => `<th>${c[1]}</th>`).join("") + "</tr>";
  const rows = visibleRows();
  $("tbody").innerHTML = rows.length
    ? rows
        .map(
          (r) =>
            `<tr data-id="${r.id}" class="${newIds.has(r.id) ? "new" : ""}${
              state.selected === r.id ? " sel" : ""
            }">` + cols.map((c) => cell(r, c[0], c[2])).join("") + "</tr>"
        )
        .join("")
    : `<tr class="empty"><td colspan="${cols.length}">${emptyMessage()}</td></tr>`;
  $("rowcount").textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
  $("table-title").textContent = `Database — ${state.layer}`;

  for (const tr of $("tbody").querySelectorAll("tr")) {
    tr.onclick = () => {
      const id = Number(tr.dataset.id);
      const row = (state.layer === "defects" ? state.defects : state.events).get(id);
      if (row) select(row, state.layer);
    };
  }
}

/* ----------------------------------------------------------------- detail */

async function select(row, kind) {
  state.selected = row.id;
  renderTable();
  if (row.lat != null) map.panTo([row.lat, row.lon], { animate: true });

  $("d-title").textContent =
    kind === "defects" ? `Defect #${row.id}` : `Sighting #${row.id}`;

  const crop = row.best_crop || row.crop_path;
  const kv = [];
  const push = (k, v) => kv.push(`<span>${k}</span><span>${v}</span>`);
  push("type", labelOf(row.damage_type));
  push("severity", fmt(row.severity, 2));
  if (kind === "defects") {
    push("status", row.status);
    push("sightings", row.sightings);
    push("buses", row.distinct_buses);
    push("max conf", fmt(row.max_confidence * 100) + "%");
    push("first seen", String(row.first_seen).replace("T", " ").slice(0, 19));
    push("last seen", String(row.last_seen).replace("T", " ").slice(0, 19));
    const days = daysBetween(row.first_seen, row.last_seen);
    if (days >= 1) push("open for", `${days} day${days === 1 ? "" : "s"}`);
  } else {
    push("bus", row.bus_id);
    push("route", row.route_id || "—");
    push("conf", fmt(row.confidence * 100) + "%");
    push("frames", row.frames_seen);
    push("captured", String(row.captured_at).replace("T", " ").slice(0, 19));
    push("clip", row.source_clip || "—");
    push("model", row.model_id || "—");
  }
  if (row.lat != null) push("position", `${fmt(row.lat, 5)}, ${fmt(row.lon, 5)}`);

  let html = "";
  if (crop) html += `<img src="${crop}" alt="defect crop">`;
  html += `<div class="kv">${kv.join("")}</div>`;

  // Sighting history is the whole point of clustering -- show who saw it.
  if (kind === "defects") {
    try {
      const hist = await api(`/api/events/by-defect/${row.id}`);
      if (hist.length) {
        html +=
          `<div class="hist"><h4>Sighting history</h4>` +
          hist
            .map(
              (e) =>
                `<div>${String(e.captured_at).replace("T", " ").slice(0, 16)} · ` +
                `${e.bus_id} · ${fmt(e.confidence * 100)}%</div>`
            )
            .join("") +
          `</div>`;
      }
    } catch (_) { /* history is a nicety; never break the panel over it */ }
  }

  $("d-body").innerHTML = html;
  $("detail").classList.remove("hidden");
}

function daysBetween(a, b) {
  if (!a || !b) return 0;
  return Math.floor((new Date(b) - new Date(a)) / 86400000);
}

/* ----------------------------------------------------------------- alerts */

// A severity threshold, evaluated client-side. Real SMS/WhatsApp delivery is
// an integration cost a prototype cannot justify, so this shows the rule
// firing rather than pretending a message was sent -- see V1-Plan.md.
const ALERT_SEVERITY = 0.75;
const ALERT_CONF = 0.55;

function maybeAlert(ev) {
  if ((ev.severity || 0) < ALERT_SEVERITY) return;
  if ((ev.confidence || 0) < ALERT_CONF) return;

  const el = document.createElement("div");
  el.className = "alert";
  el.style.borderLeftColor = colourOf(ev.damage_type);
  el.innerHTML =
    `<h5><span style="color:${colourOf(ev.damage_type)}">⚠</span> ` +
    `${labelOf(ev.damage_type)} · ${fmt(ev.confidence * 100)}%</h5>` +
    `<p>${ev.bus_id} · ${ev.route_id || "—"}<br>` +
    `${ev.lat != null ? fmt(ev.lat, 5) + ", " + fmt(ev.lon, 5) : "no fix"}</p>`;
  el.onclick = () => select(ev, "events");

  const box = $("alerts");
  box.prepend(el);
  while (box.children.length > 4) box.lastChild.remove();
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 400);
  }, 9000);
}

/* ------------------------------------------------------------------- kpis */

function renderStats(s) {
  $("k-defects").textContent = s.defects;
  $("k-events").textContent = s.events;
  $("k-buses").textContent = s.buses_reporting;
  $("k-routes").textContent = `${s.routes} route${s.routes === 1 ? "" : "s"}`;
  $("k-score").textContent = fmt(s.damage_score, 1);

  const open = s.by_status && s.by_status.open ? s.by_status.open : 0;
  const unc = s.by_status && s.by_status.unconfirmed ? s.by_status.unconfirmed : 0;
  $("k-defects-sub").textContent =
    s.defects > 0 ? `${open} open · ${unc} unconfirmed` : "clustering pending";

  $("k-types").innerHTML = Object.entries(s.by_type || {})
    .map(
      ([k, v]) =>
        `<span class="chip"><i style="background:${colourOf(k)}"></i>${labelOf(k)} ${v}</span>`
    )
    .join("") || '<span class="muted">no events yet</span>';
}

/* ------------------------------------------------------------------ fetch */

async function loadStatic() {
  const tax = await api("/api/taxonomy");
  for (const t of tax) state.taxonomy[t.key] = t;

  $("legend").innerHTML = tax
    .filter((t) => t.key !== "unknown")
    .map((t) => `<div><i style="background:${t.hex}"></i>${t.label}</div>`)
    .join("");

  const sel = $("f-type");
  for (const t of tax) {
    const o = document.createElement("option");
    o.value = t.key;
    o.textContent = t.label;
    sel.appendChild(o);
  }

  const routes = await api("/api/routes.geojson");
  L.geoJSON(routes, {
    style: { color: "#3fb6ff", weight: 3, opacity: 0.32 },
    onEachFeature: (f, l) =>
      l.bindTooltip(f.properties.name || f.properties.route_id, { sticky: true }),
  }).addTo(routeLayer);

  if (routes.features && routes.features.length) {
    map.fitBounds(L.geoJSON(routes).getBounds(), { padding: [40, 40] });
  }
}

function fitToData() {
  // Routes span the whole city but a clip covers ~85 m, so fitting to the
  // routes leaves the actual findings as unreadable specks. Once there is
  // data, frame that instead.
  const pts = visibleRows()
    .filter((r) => r.lat != null)
    .map((r) => [r.lat, r.lon]);
  if (!pts.length) return;
  map.fitBounds(L.latLngBounds(pts).pad(0.35), { maxZoom: 17 });
}

async function poll() {
  try {
    const stats = await api("/api/stats");

    // The server was reset (or pointed at a different database) while this
    // page stayed open. Event ids restart from 1, so a cursor left at the old
    // high-water mark would silently skip every new event. Detect it by the
    // id going backwards and start over.
    if (stats.latest_event_id < state.lastEventId) {
      state.lastEventId = 0;
      state.events.clear();
      state.defects.clear();
      state.selected = null;
      state.firstLoad = true;
      $("detail").classList.add("hidden");
      $("alerts").innerHTML = "";
      markerLayer.clearLayers();
      state.markers.clear();
    }

    renderStats(stats);

    // Events come in on a cursor; defects are re-read whole because clustering
    // mutates existing rows (sightings, status) rather than only appending.
    const fresh = await api(`/api/events?since=${state.lastEventId}&limit=500`);
    const newIds = new Set();
    for (const e of fresh) {
      state.events.set(e.id, e);
      state.lastEventId = Math.max(state.lastEventId, e.id);
      newIds.add(e.id);
      if (!state.firstLoad) maybeAlert(e);  // don't replay history as alerts
    }

    const defects = await api("/api/defects?limit=5000");
    const seenBefore = new Set(state.defects.keys());
    const newDefectIds = new Set();
    state.defects.clear();
    for (const d of defects) {
      state.defects.set(d.id, d);
      if (!seenBefore.has(d.id)) newDefectIds.add(d.id);
    }

    if (state.layer === "defects") {
      rebuildMarkers();
      for (const id of newDefectIds) {
        const d = state.defects.get(id);
        if (d) upsertMarker(d, "defects", true);
      }
      renderTable(newDefectIds);
    } else {
      for (const e of fresh) upsertMarker(e, "events", true);
      renderTable(newIds);
    }

    setConn(true, `live · ${stats.events} events`);
    if (state.firstLoad) {
      // Opening on an empty Defects panel reads as a broken dashboard. If
      // nothing has been clustered yet but sightings exist, show those.
      if (!state.defects.size && state.events.size) setLayer("events");
      fitToData();
    }
    state.firstLoad = false;
  } catch (err) {
    setConn(false, "server unreachable");
  } finally {
    $("clock").textContent = new Date().toLocaleTimeString();
  }
}

/* ------------------------------------------------------------------ wire */

function setLayer(layer) {
  state.layer = layer;
  for (const el of $("layer-toggle").children)
    el.classList.toggle("on", el.dataset.layer === layer);
  $("f-status").disabled = layer !== "defects";
  rebuildMarkers();
  renderTable();
}

$("layer-toggle").onclick = (e) => {
  const b = e.target.closest("button");
  if (b) setLayer(b.dataset.layer);
};

$("fit").onclick = fitToData;
$("heat").onchange = (e) => { state.heat = e.target.checked; drawHeat(); };
$("follow").onchange = (e) => { state.follow = e.target.checked; };
$("f-type").onchange = (e) => { state.filters.type = e.target.value; rebuildMarkers(); renderTable(); };
$("f-status").onchange = (e) => { state.filters.status = e.target.value; rebuildMarkers(); renderTable(); };
$("d-close").onclick = () => {
  $("detail").classList.add("hidden");
  state.selected = null;
  renderTable();
};

(async function start() {
  try {
    await loadStatic();
  } catch (err) {
    setConn(false, "cannot reach API");
    return;
  }
  await poll();
  setInterval(poll, POLL_MS);
})();
