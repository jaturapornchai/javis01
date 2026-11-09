// Real footage for the javis video: one call to the customer line (127.0.0.1:7861) in Playwright's Chromium.
// The caller's lines are ElevenLabs takes played into a fake mic; javis answers for real.
//   node video/capture.mjs t1   -> build/cap/t1/: f*.jpg screencast frames, call.webm (caller + javis), take.json (timings, texts, boxes)
// The ElevenLabs key is read from a file and never printed.
import fs from "node:fs";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL(".", import.meta.url)).replace(/\\/g, "/").replace(/\/$/, "");
const take = process.argv[2] ?? "t1", DIR = `${ROOT}/build/cap/${take}`;
const URL_ = "http://127.0.0.1:7861/", W = 1920, H = 1080;
const VOICE = "H9eYlNHpm1vah6PPgYAk" /* Ekkarat, Thai male, conversational */, MODEL = "eleven_v4";
// barge: played while javis is still answering the line before (shows the caller can cut in)
const LINES = [
  // no greeting or comma pauses: a 0.5 s pause (config.py PAUSE_SILENCE_S) ends the caller's turn; "เฉลี่ย" not "ถัวเฉลี่ย" (STT heard ถั่ว)
  { id: "l1", text: "ผมเปิดร้านวัสดุก่อสร้างครับ อยากรู้ว่าโปรแกรมคิดต้นทุนเฉลี่ยยังไงครับ" },
  { id: "l2", text: "ช่วยทำตัวอย่างเป็นตารางให้ดูหน่อยครับ" },
  { id: "l3", text: "ขอคิวรี่ดูต้นทุนเฉลี่ยของสินค้าแต่ละตัวหน่อยครับ" },
  { id: "l4", text: "เดี๋ยวครับ ขอถามเรื่องระบบเงินเดือนหน่อยครับ", barge: true },
  { id: "l5", text: "ไม่เปลี่ยนครับ วันนี้พอแค่นี้ ขอบคุณมากครับ" },
];
const BOXES = ["hud", "pill", "team", "teamlist", "tr", "panel", "pbody", "cards", "subs", "user", "bot", "chat", "callBtn"];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = (...a) => console.log(...a);

function tighten(pcm, rate) { // pauses inside a take over 0.3 s -> 0.25 s, so the server never ends the turn mid-line
  const n = pcm.length >> 1, quiet = i => Math.abs(pcm.readInt16LE(i * 2)) < 400, keep = [], max = rate * 0.3, cut = rate * 0.25;
  for (let i = 0; i < n;) { let j = i; while (j < n && quiet(j)) j++; if (j - i > max) { keep.push([i, i + cut]); } else if (j > i) keep.push([i, j]); i = j; while (j < n && !quiet(j)) j++; keep.push([i, j]); i = j; }
  return Buffer.concat(keep.filter(([a, b]) => b > a).map(([a, b]) => pcm.subarray(a * 2, b * 2)));
}
function wav(pcm, rate) { // 16-bit mono PCM -> WAV bytes, so the page can decodeAudioData it
  const h = Buffer.alloc(44); h.write("RIFF", 0); h.writeUInt32LE(36 + pcm.length, 4); h.write("WAVEfmt ", 8); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20);
  h.writeUInt16LE(1, 22); h.writeUInt32LE(rate, 24); h.writeUInt32LE(rate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34); h.write("data", 36); h.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([h, pcm]);
}
async function callerTake(text) { // cached per text: re-runs pay nothing
  const f = `${ROOT}/build/caller/${createHash("sha1").update([MODEL, VOICE, text].join("|")).digest("hex").slice(0, 16)}.wav`;
  if (!fs.existsSync(f)) {
    const key = fs.readFileSync(`${process.env.LOCALAPPDATA}/BotPress/elevenlabs.key`, "utf8").trim();
    const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${VOICE}?output_format=pcm_16000`, { method: "POST",
      headers: { "xi-api-key": key, "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ text, model_id: MODEL, language_code: "th", voice_settings: { stability: 0.5, similarity_boost: 0.75 } }) });
    if (!r.ok) throw new Error(`ElevenLabs ${r.status}: ${(await r.text()).slice(0, 200)}`);
    fs.mkdirSync(`${ROOT}/build/caller`, { recursive: true });
    fs.writeFileSync(f, wav(tighten(Buffer.from(await r.arrayBuffer()), 16000), 16000));
    log(`caller tts: ${text.length} ตัวอักษร`);
  }
  return fs.readFileSync(f).toString("base64");
}

// call.webm starts at audioStart, not recStart: the recorder gets no samples while nothing feeds it (t2: 4.24 s later)
// runs in the page before its own script: a fake mic the caller lines play into, and a recorder of mic + javis
function hooks() {
  const ac = new AudioContext(), mic = ac.createMediaStreamDestination(), rec = ac.createMediaStreamDestination();
  navigator.mediaDevices.getUserMedia = async () => mic.stream;
  // a real mic always hisses; dead digital silence between lines sent no audio at all, so the server never saw the pause
  const hiss = ac.createBuffer(1, ac.sampleRate, ac.sampleRate), d0 = hiss.getChannelData(0); for (let i = 0; i < d0.length; i++) d0[i] = (Math.random() * 2 - 1) * 0.0008;
  const hs = ac.createBufferSource(); hs.buffer = hiss; hs.loop = true; hs.connect(mic); hs.start();
  window.__say = async b64 => {
    const buf = await ac.decodeAudioData(Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer);
    const s = ac.createBufferSource(); s.buffer = buf; s.connect(mic); s.connect(rec); s.start();
    return new Promise(r => { s.onended = r; });
  };
  const d = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "srcObject");
  Object.defineProperty(HTMLMediaElement.prototype, "srcObject", { get() { return d.get.call(this); }, set(v) {
    if (v && this.id === "remote") { const s = ac.createMediaStreamSource(v), an = ac.createAnalyser(); s.connect(rec); s.connect(an); window.__an = an; window.__audioStart ??= Date.now(); }
    d.set.call(this, v);
  } });
  window.__level = () => { if (!window.__an) return 0; const a = new Float32Array(2048); window.__an.getFloatTimeDomainData(a); let m = 0; for (const x of a) m = Math.max(m, Math.abs(x)); return m; };
  window.__rec = () => { window.__parts = []; const mr = new MediaRecorder(rec.stream, { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 192000 });
    mr.ondataavailable = e => window.__parts.push(e.data); mr.start(1000); window.__mr = mr; return Date.now(); };
  window.__stop = () => new Promise(r => { window.__mr.onstop = async () => {
    const u = new Uint8Array(await new Blob(window.__parts).arrayBuffer()); let s = "";
    for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000));
    r(btoa(s)); }; window.__mr.stop(); });
}

const pw = createRequire(import.meta.url)(`${process.env.APPDATA}/npm/node_modules/@playwright/mcp/node_modules/playwright`);
fs.rmSync(DIR, { recursive: true, force: true }); fs.mkdirSync(DIR, { recursive: true });
const says = {}; for (const l of LINES) says[l.id] = await callerTake(l.text);
const browser = await pw.chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, permissions: ["microphone"] });
await ctx.addInitScript(hooks);
const page = await ctx.newPage();
const errs = []; page.on("pageerror", e => errs.push(e.message));
const cdp = await ctx.newCDPSession(page), frames = [];
cdp.on("Page.screencastFrame", f => {
  const n = frames.length; fs.writeFileSync(`${DIR}/f${String(n).padStart(5, "0")}.jpg`, Buffer.from(f.data, "base64"));
  frames.push(f.metadata.timestamp * 1000); cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
});
const events = [], mark = (what, extra = {}) => { events.push({ t: Date.now(), what, ...extra }); log(what, extra.id ?? ""); };
const state = () => page.evaluate(() => [document.getElementById("hud").dataset.s, window.__level()]);
const boxes = () => page.evaluate(ids => Object.fromEntries(ids.map(id => { const r = document.getElementById(id)?.getBoundingClientRect();
  return [id, r && r.width ? [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)] : null]; })), BOXES);
async function waitQuiet(limit = 120000) { // the turn is over and javis has been silent for 1.5 s
  const t0 = Date.now(); let quiet = 0;
  while (Date.now() - t0 < limit) { await sleep(250); const [s, lv] = await state(); quiet = s === "idle" && lv < 0.01 ? quiet + 250 : 0; if (quiet >= 1500) return true; }
  return false;
}
const turns = () => page.evaluate(() => document.querySelectorAll("#trlist li.u").length);
async function waitTurn(n0) { const t0 = Date.now(); while (Date.now() - t0 < 20000 && (await turns()) <= n0) await sleep(200); } // the server took the line as a turn
async function waitSpeaking(ms) { const t0 = Date.now(); while (Date.now() - t0 < 60000) { await sleep(200); if ((await state())[1] > 0.02) break; } await sleep(ms); }

await page.goto(URL_);
await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: W, maxHeight: H, everyNthFrame: 1 });
const recStart = await page.evaluate(() => window.__rec()); mark("rec");
await page.waitForFunction(() => document.getElementById("callBtn").textContent === "วางสาย", null, { timeout: 30000 }); mark("connected");
await page.waitForFunction(() => document.getElementById("bot").textContent.length > 0, null, { timeout: 30000 }); mark("hello");
await waitQuiet(); mark("greeted", { boxes: await boxes() });
for (const l of LINES) {
  if (l.barge) await waitSpeaking(3500); else await waitQuiet();
  const n0 = await turns(); mark("say", { id: l.id, text: l.text }); await page.evaluate(b => window.__say(b), says[l.id]); mark("said", { id: l.id });
  if (!LINES[LINES.indexOf(l) + 1]?.barge) { await waitTurn(n0); await waitQuiet(); mark("answered", { id: l.id, boxes: await boxes() }); }
}
await sleep(1500);
const transcript = await page.evaluate(() => [...document.querySelectorAll("#trlist li")].map(li => li.textContent));
const audioStart = await page.evaluate(() => window.__audioStart);
const b64 = await page.evaluate(() => window.__stop()); fs.writeFileSync(`${DIR}/call.webm`, Buffer.from(b64, "base64"));
await cdp.send("Page.stopScreencast"); mark("end");
await page.evaluate(() => document.getElementById("callBtn").click()).catch(() => {});
await sleep(500); await browser.close();
fs.writeFileSync(`${DIR}/take.json`, JSON.stringify({ recStart, audioStart, frames, events, transcript, errs }, null, 1));
log(`${take}: ${frames.length} เฟรม ${((frames.at(-1) - frames[0]) / 1000).toFixed(1)} วินาที, page errors ${errs.length}`);
