// Dashboard client: polls camera JPEGs, listens to the Server-Sent Events stream.
"use strict";

const APPROACH_NAMES = { N: "North", E: "East", S: "South", W: "West" };
const $ = (id) => document.getElementById(id);
const feedsBuilt = new Set();
let evCount = 0;

// ---------------------------------------------------------------- camera tiles
function buildFeed(name) {
  if (feedsBuilt.has(name)) return;
  feedsBuilt.add(name);
  if (name === "overview") {
    $("overview-card").hidden = false;
    pollImage($("overview"), name);
    return;
  }
  const el = document.createElement("div");
  el.className = "feed";
  el.id = `feed-${name}`;
  el.innerHTML = `
    <div class="feed-h">
      <span class="name"><span class="dot" id="dot-${name}"></span>${APPROACH_NAMES[name] || name} approach</span>
      <span class="nums" id="nums-${name}"></span>
    </div>
    <img id="img-${name}" alt="${APPROACH_NAMES[name] || name} approach camera">`;
  // keep N, E, S, W order
  const order = ["N", "E", "S", "W"];
  const after = [...$("feeds").children].find((c) => order.indexOf(c.id.slice(5)) > order.indexOf(name));
  $("feeds").insertBefore(el, after || null);
  pollImage($(`img-${name}`), name);
}

// Load the next frame only after the previous one arrived (no request pile-up).
function pollImage(img, name) {
  const next = () => setTimeout(load, 90);
  img.onload = next;
  img.onerror = () => setTimeout(load, 800);
  function load() { img.src = `/frame/${name}.jpg?t=${Date.now()}`; }
  load();
}

// ---------------------------------------------------------------- state
function renderState(s) {
  if (!s.ready) return;
  $("junction").textContent = s.junction;
  $("mode").textContent = s.mode === "sim" ? "simulation" : "live video";
  $("detector").textContent = s.detector;
  $("clock").textContent = fmtClock(s.t);
  $("fps").textContent = `${s.processing_fps} fps max`;
  s.feeds.forEach(buildFeed);

  // signal widget
  const row = $("signal-row");
  row.innerHTML = Object.entries(s.signal).map(([a, st]) =>
    `<div class="sig ${st}"><div class="lamp"></div><div class="a">${a}</div><div class="hint">${st}</div></div>`).join("");
  const c = s.controller;
  $("ctl-name").textContent = `${c.controller} timing`;
  $("ctl-current").textContent = APPROACH_NAMES[c.current] || c.current;
  $("ctl-stage").textContent = c.stage.replace("_", " ");
  $("ctl-remaining").textContent = `${c.remaining_s.toFixed(1)} s`;

  renderEmergency(s);
  renderEnforcement(s);
  renderIncidents(s);
  $("demo-card").hidden = s.mode !== "sim";

  // per approach
  const rows = [];
  for (const [a, v] of Object.entries(s.approaches)) {
    const dot = $(`dot-${a}`);
    if (dot) dot.className = `dot ${v.signal}`;
    const feed = $(`feed-${a}`);
    if (feed) feed.classList.toggle("green", v.signal === "GREEN");
    const nums = $(`nums-${a}`);
    if (nums) nums.textContent = `${v.count} in lane - ${v.queue} queued`;
    const occ = Math.round(v.occupancy * 100);
    rows.push(`<tr><td><span class="dot ${v.signal}"></span>${APPROACH_NAMES[a] || a}</td><td>${v.count}</td>
      <td>${v.queue}</td><td><span class="bar"><i style="width:${occ}%"></i></span>${occ}%</td>
      <td>${v.flow_per_min}</td></tr>`);
  }
  $("stats-body").innerHTML = rows.join("");

  if (s.sim) {
    const box = $("simstats");
    box.hidden = false;
    box.innerHTML = `
      <div><span class="label">Avg wait</span><b>${s.sim.avg_wait_s}s</b></div>
      <div><span class="label">Vehicles served</span><b>${s.sim.completed}</b></div>
      <div><span class="label">Waiting to enter</span><b>${s.sim.pending_entry}</b></div>`;
  }
}

// ---------------------------------------------------------------- emergency
let emgButtonsFor = "";
function renderEmergency(s) {
  const e = s.emergency;
  if (!e) return;
  $("emg-card").classList.toggle("active", !!e.active);
  $("emg-status").textContent = !e.enabled ? "Disabled in config"
    : e.active ? `Priority green for ${APPROACH_NAMES[e.active.approach] || e.active.approach} (${e.active.source})`
    : "No emergency vehicle";
  $("emg-served").textContent = `${e.served} served`;
  // Sim: send a real ambulance into traffic. Video: manual request (GPS / RFID / operator).
  const sim = s.mode === "sim";
  const key = `${sim}:${Object.keys(s.approaches).join("")}`;
  if (key === emgButtonsFor) return;
  emgButtonsFor = key;
  $("emg-label").textContent = sim ? "Send ambulance" : "Manual priority";
  $("emg-btns").innerHTML = Object.keys(s.approaches).map((a) =>
    `<button class="btn" data-a="${a}" title="${APPROACH_NAMES[a] || a}">${a}</button>`).join("");
  $("emg-btns").querySelectorAll("button").forEach((b) => {
    b.onclick = async () => {
      b.disabled = true;
      const url = sim ? `/api/sim/ambulance/${b.dataset.a}` : `/api/emergency/${b.dataset.a}`;
      try { await fetch(url, { method: "POST" }); } finally { setTimeout(() => { b.disabled = false; }, 800); }
    };
  });
}

// ---------------------------------------------------------------- enforcement
const VIO = { red_light: "Red light", wrong_way: "Wrong way", no_helmet: "No helmet", speeding: "Speeding" };
const evLink = (p, text) => p ? `<a href="/evidence/${encodeURI(p)}" target="_blank" rel="noopener">${text}</a>` : "";

function renderEnforcement(s) {
  const v = s.violations;
  if (!v) return;
  $("enf-total").textContent = v.enabled ? `${v.total} violations - Rs. ${v.fines_rs.toLocaleString("en-IN")}` : "disabled";
  $("vio-grid").innerHTML = Object.entries(VIO).map(([k, name]) =>
    `<div><span class="label">${name}</span><b>${v.counts[k] || 0}</b></div>`).join("");
  // Video mode: say which rules cannot run on which camera (and why).
  const off = [];
  for (const [a, r] of Object.entries(v.rules || {})) {
    const missing = Object.entries(r).filter(([, on]) => !on).map(([k]) => VIO[k]);
    if (missing.length) off.push(`${a}: ${missing.join(", ")}`);
  }
  $("enf-rules").textContent = off.length && s.mode !== "sim"
    ? `Not checked - ${off.join("; ")} (needs light_roi / calib in zones, or a helmet model)` : "";
  const list = $("challans");
  if (!v.recent.length) return;
  list.innerHTML = v.recent.map((c) => `
    <li><span class="plate">${escapeHtml(c.plate || "UNREAD")}</span>
      <span>${VIO[c.violation] || escapeHtml(c.violation)} - ${escapeHtml(c.approach)}<br>
        <span class="hint">${escapeHtml(c.challan_no)} - Rs. ${c.amount}${c.status === "REVIEW" ? " - manual review" : ""}</span></span>
      <span class="links">${evLink(c.pdf, "PDF")} ${evLink(c.snapshot, "photo")}</span></li>`).join("");
}

function renderIncidents(s) {
  const c = s.collision;
  if (!c) return;
  $("inc-alerts").textContent = c.enabled ? `alerts: ${c.alerts}` : "disabled";
  $("inc-card").classList.toggle("active", c.recent.some((i) => s.t - i.t < 60));
  if (!c.recent.length) return;
  $("incidents").innerHTML = c.recent.map((i) => {
    const alert = i.alert == null ? "calling..." : i.delivered ? `alert sent (${i.alert})`
      : i.alert === "dry-run" ? "alert recorded (dry run)" : "alert FAILED";
    return `<li><span><b>${escapeHtml(i.id)}</b> - ${escapeHtml(i.approach)} approach, ${fmtClock(i.t)}<br>
      <span class="hint">${i.vehicles.map(escapeHtml).join(" + ")} - ${alert}${i.views > 1 ? ` - ${i.views} cameras` : ""}</span></span>
      <span class="links">${evLink(i.snapshot, "photo")}</span></li>`;
  }).join("");
}

$("demo-btns").querySelectorAll("button").forEach((b) => {
  b.onclick = async () => {
    b.disabled = true;
    try {
      const r = await (await fetch(`/api/sim/inject/${b.dataset.k}`, { method: "POST" })).json();
      $("demo-msg").textContent = r.ok ? `${b.textContent} sent into the ${APPROACH_NAMES[r.approach] || r.approach} approach`
        : (r.error || "failed");
    } catch (e) { $("demo-msg").textContent = "request failed"; }
    finally { setTimeout(() => { b.disabled = false; }, 800); }
  };
});

function fmtClock(t) {
  const m = Math.floor(t / 60), sec = Math.floor(t % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

// ---------------------------------------------------------------- events
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function addEvent(ev, isNew) {
  if (ev.type === "SIGNAL_CHANGE" && !(ev.data && ev.data.log)) return;  // too chatty
  const li = document.createElement("li");
  li.className = `${ev.severity || "info"}${isNew ? " new" : ""}`;
  const time = new Date(ev.ts * 1000).toLocaleTimeString([], { hour12: false });
  const links = (ev.evidence || []).map((p) =>
    ` <a href="/evidence/${encodeURI(p)}" target="_blank" rel="noopener">evidence</a>`).join("");
  li.innerHTML = `<span class="ts">${time}</span>
    <span><span class="type">${escapeHtml(ev.type)}</span>${escapeHtml(ev.message)}${links}</span>`;
  const log = $("log");
  log.insertBefore(li, log.firstChild);
  while (log.children.length > 200) log.removeChild(log.lastChild);
  $("ev-count").textContent = `${++evCount}`;
}

async function loadHistory() {
  try {
    const rows = await (await fetch("/api/events?limit=60")).json();
    rows.reverse().forEach((r) => addEvent(r, false));
  } catch (e) { /* server not ready yet */ }
}

function connect() {
  const es = new EventSource("/events/stream");
  es.addEventListener("state", (m) => { renderState(JSON.parse(m.data)); });
  es.addEventListener("ev", (m) => addEvent(JSON.parse(m.data), true));
  es.onopen = () => { $("conn").textContent = "live"; $("conn").classList.add("live"); };
  es.onerror = () => { $("conn").textContent = "reconnecting"; $("conn").classList.remove("live"); };
}

loadHistory().then(connect);
