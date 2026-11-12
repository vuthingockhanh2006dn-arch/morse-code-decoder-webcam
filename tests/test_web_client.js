// Run with node tests/test_web_client.js. Exercises real presentation/input code.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const elements = new Map(), timers = [], calls = [];
function element() {
  return {children: [], dataset: {}, style: {}, options: [], value: "auto", checked: false,
    classList: {toggle() {}}, handlers: {},
    append(...children) { this.children.push(...children); },
    appendChild(child) { this.children.push(child); },
    addEventListener(name, fn) { this.handlers[name] = fn; }, setPointerCapture() {},
    getContext() { return {}; }};
}
function get(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }
const snapshot = {state: "IDLE", morse_buffer: "", decoded_text: "", preview: null, events: []};
const context = {console, performance: {now: () => 0}, AbortController, AbortSignal, Blob,
  setTimeout, clearTimeout, setInterval: fn => timers.push(fn),
  location: {protocol: "http:"}, navigator: {clipboard: {writeText: async () => {}}},
  document: {getElementById: get, createElement: element, createTextNode: text => ({text})},
  window: {addEventListener() {}},
  fetch: async (url, options) => {
    if (options?.body) calls.push({url, body: JSON.parse(options.body)});
    return {ok: true, json: async () => url === "/api/status" ?
      {...snapshot, config: {touch_threshold: .25}, loaded_models: [], morse_table: {".": "E", "-": "T"}} : snapshot};
  }};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8"), context);
const settle = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  await settle();
  assert.equal(get("morse-table").children.length, 2, "table comes from Python");
  const tick = timers[0];
  get("pad-thumb").handlers.pointerdown({preventDefault() {}, pointerId: 1});
  await settle();
  assert.equal(calls.at(-1).body.pose, "PINCH_THUMB");
  get("pad-thumb").handlers.pointerup();
  for (let i = 0; i < 5; i++) { tick(); await settle(); }
  assert.ok(calls.slice(-5).every(call => call.body.pose === "IDLE"), "release continues without camera");
  vm.runInContext("isStreaming = true", context);
  const before = calls.length;
  get("pad-ring").handlers.pointerdown({preventDefault() {}, pointerId: 2});
  tick(); await settle();
  assert.equal(calls.length, before, "simulator cannot mix commands into camera frames");
  await get("clear-btn").handlers.click();
  assert.equal(calls.at(-1).url, "/api/reset", "clear uses the same Python buffer");
  vm.runInContext('render({state:"IDLE", morse_buffer:".", decoded_text:"AN", preview:"E", events:[]})', context);
  assert.equal(get("decoded-text-view").textContent, "AN");
  assert.equal(get("preview-char").textContent, "Ký tự tiếp theo: E");
  assert.equal(vm.runInContext("typeof ClientMorseFSM", context), "undefined");
  assert.equal(vm.runInContext("typeof ClientMorseBuffer", context), "undefined");
  console.log("Web client checks passed: ordered API, continuous release, single decoder, rendering.");
})().catch(error => { console.error(error); process.exitCode = 1; });
