// javis YouTube video: ElevenLabs narration per scene + HTML motion scenes (some over real call footage from capture.mjs) captured frame by
// frame in Chromium + Thai captions (libass) + the calls' own audio + synth music bed. Pipeline copied from D:/domecard/video/make.mjs.
//   node video/make.mjs tts              pay ElevenLabs only for new/edited scene text (cached per text in build/tts)
//   node video/make.mjs render [s01 ..]  capture scenes/<id>.html -> build/v/<id>.mp4 (all scenes when none given)
//   node video/make.mjs mix              scenes + narration + call audio + bed + captions -> out/javis-youtube.mp4
//   node video/make.mjs all
// Needs ffmpeg on PATH and the Playwright bundled with the global @playwright/mcp. The ElevenLabs key is read from a file and never printed.
import fs from "node:fs";
import { createHash } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { SCENES, SAY_AS } from "./scenes.mjs";
import { bedGraph } from "file:///D:/botstery/moods.mjs"; // ponytail: borrow the licence-free synth bed from botstery instead of copying it

const ROOT = fileURLToPath(new URL(".", import.meta.url)).replace(/\\/g, "/").replace(/\/$/, "");
const B = `${ROOT}/build`, OUT = `${ROOT}/out`;
const FPS = 30, W = 1920, H = 1080, HEAD = 0.5, TAIL = 0.9, BPS = 96000, JOBS = 4;
const MODEL = "eleven_v4", VOICE = "nealpwJT5tCyJFKDzEq8" /* Aom, Thai female (advertisement) */, VOICE_SETTINGS = { stability: 0.3, similarity_boost: 0.75 };
const MAXLINE = 42; // caption visible chars per line at 1080p
const log = (...a) => console.log(...a), die = m => { console.error(m); process.exit(1); };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const COMB = /[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]/g, vlen = t => t.replace(COMB, "").length;

// ---- narration markup -> { tts, runs: [{a, b, cap}] (caption pieces on the tts string), anchors: [tts index] }
export function parse(text) {
  let tts = ""; const runs = [], anchors = [];
  const plain = s => { // split brand words out of plain text so the caption can show their written form
    while (s) {
      let hit = null;
      for (const [cap, say] of SAY_AS) { const i = s.indexOf(say); if (i >= 0 && (!hit || i < hit.i || (i === hit.i && say.length > hit.say.length))) hit = { i, cap, say }; }
      const head = hit ? s.slice(0, hit.i) : s;
      if (head) { runs.push({ a: tts.length, b: tts.length + head.length }); tts += head; }
      if (!hit) break;
      runs.push({ a: tts.length, b: tts.length + hit.say.length, cap: hit.cap }); tts += hit.say;
      s = s.slice(hit.i + hit.say.length);
    }
  };
  for (const m of text.matchAll(/\[[^\]]*\]\s*|\^|\{([^|}]*)\|([^}]*)\}|[^[^{]+/g)) {
    const t = m[0];
    if (t === "^") anchors.push(tts.length);
    else if (t.startsWith("[")) tts += t; // delivery tag: spoken text for TTS, not part of any caption run
    else if (m[1] !== undefined) { runs.push({ a: tts.length, b: tts.length + m[2].length, cap: m[1] }); tts += m[2]; }
    else plain(t);
  }
  return { tts, runs, anchors };
}

// ---- ElevenLabs (same request shape as D:/botstery/make.mjs, proven with eleven_v4 + Thai)
const elevenKey = () => fs.readFileSync(`${process.env.LOCALAPPDATA}/BotPress/elevenlabs.key`, "utf8").trim();
async function speak(key, text, prev) {
  for (let i = 0; ; i++) {
    const body = { text, model_id: MODEL, language_code: "th", voice_settings: VOICE_SETTINGS, ...(prev.length ? { previous_request_ids: prev.slice(-3) } : {}) };
    const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${VOICE}/with-timestamps?output_format=pcm_48000`, { method: "POST",
      headers: { "xi-api-key": key, "Content-Type": "application/json; charset=utf-8" }, body: JSON.stringify(body) });
    if ((r.status === 429 || r.status >= 500) && i < 4) { await sleep(2000 * 2 ** i); continue; }
    if (r.status === 400 && prev.length) { prev = []; continue; } // stitching ids expire: retry as a standalone take
    if (!r.ok) throw new Error(`ElevenLabs ${r.status}: ${(await r.text()).slice(0, 300)}`);
    const j = await r.json();
    return { pcm: Buffer.from(j.audio_base64, "base64"), al: j.alignment, id: r.headers.get("request-id") };
  }
}
const cacheOf = tts => `${B}/tts/${createHash("sha1").update([MODEL, VOICE, JSON.stringify(VOICE_SETTINGS), "th", tts].join("|")).digest("hex").slice(0, 16)}`;
async function tts() {
  fs.mkdirSync(`${B}/tts`, { recursive: true });
  let key = null, paid = 0; const ids = [];
  for (const s of SCENES) {
    const p = parse(s.text), c = cacheOf(p.tts);
    if (!fs.existsSync(`${c}.json`)) {
      key ??= elevenKey();
      const g = await speak(key, p.tts, ids);
      fs.writeFileSync(`${c}.pcm`, g.pcm); fs.writeFileSync(`${c}.json`, JSON.stringify({ id: g.id, al: g.al }));
      paid += p.tts.length; log(`tts ${s.id}: ${p.tts.length} ตัวอักษร ${(g.pcm.length / BPS).toFixed(1)} วินาที`);
    }
    ids.push(JSON.parse(fs.readFileSync(`${c}.json`, "utf8")).id);
  }
  log(`tts: จ่ายใหม่ ${paid} ตัวอักษร`);
}

// ---- timeline: scene length = HEAD + voice + TAIL (+ hold); --p1 = voice start, --pK = anchor K-1
function timeline() {
  let t0 = 0;
  return SCENES.map(s => {
    const p = parse(s.text), c = cacheOf(p.tts);
    if (!fs.existsSync(`${c}.json`)) die(`ยังไม่มีเสียงของ ${s.id}: node video/make.mjs tts`);
    const { al } = JSON.parse(fs.readFileSync(`${c}.json`, "utf8")), voice = fs.statSync(`${c}.pcm`).size / BPS;
    if (al.characters.join("") !== p.tts) die(`${s.id}: alignment ไม่ตรงกับบท`);
    const lead = s.lead ?? 0, at0 = HEAD + lead; // narration starts after the lead clip
    const startAt = i => { while (i < p.tts.length && /\s/.test(p.tts[i])) i++; return at0 + al.character_start_times_seconds[Math.min(i, p.tts.length - 1)]; };
    const dur = Math.ceil((at0 + voice + TAIL + (s.hold ?? 0)) * FPS) / FPS, h = lead ? HEAD : at0 + voice + 0.2; // h: clip-only part starts
    const cap = s.cap && { take: s.cap.take, base: s.cap.from ?? s.cap.at - h, to: s.cap.to ?? 1e9, h }; // take second = min(base + t, to)
    const sc = { id: s.id, t0, dur, voice, at0, cap, pcm: `${c}.pcm`, p: [at0, ...p.anchors.map(startAt)].map(v => Number(v.toFixed(3))), al, parsed: p };
    t0 += dur; return sc;
  });
}

// ---- render: seek every CSS animation (WAAPI) to f/FPS, screenshot, pipe JPEG frames into ffmpeg
const playwright = () => createRequire(import.meta.url)(`${process.env.APPDATA}/npm/node_modules/@playwright/mcp/node_modules/playwright`);
async function renderScene(browser, sc) {
  const html = `${ROOT}/scenes/${sc.id}.html`;
  if (!fs.existsSync(html)) { log(`render ${sc.id}: ไม่มี ${html} ข้าม`); return; }
  const take = sc.cap && JSON.parse(fs.readFileSync(`${B}/cap/${sc.cap.take}/take.json`, "utf8"));
  const cap = take && { ...sc.cap, dir: `../build/cap/${sc.cap.take}`, f: take.frames.map(ms => Math.round(ms - take.recStart) / 1000) };
  const out = `${B}/v/${sc.id}.mp4`, stamp = createHash("sha1").update(JSON.stringify({ dur: sc.dur, p: sc.p, cap }) + ["common.css", "common.js", `${sc.id}.html`].map(f => fs.readFileSync(`${ROOT}/scenes/${f}`, "utf8")).join("")).digest("hex");
  if (fs.existsSync(out) && fs.existsSync(`${out}.stamp`) && fs.readFileSync(`${out}.stamp`, "utf8") === stamp) { log(`render ${sc.id}: ไม่เปลี่ยน`); return; }
  const page = await browser.newPage({ viewport: { width: W, height: H } });
  const errs = []; page.on("pageerror", e => errs.push(e.message));
  await page.addInitScript(s => { window.SCENE = s; }, { dur: sc.dur, p: sc.p, cap });
  await page.goto(pathToFileURL(html).href, { waitUntil: "load" });
  await page.evaluate(async () => { await document.fonts.ready; await Promise.all([...document.images].map(i => i.decode().catch(() => {}))); });
  const n = Math.round(sc.dur * FPS);
  const ff = spawn("ffmpeg", ["-y", "-v", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", String(FPS), "-frames:v", String(n), `${out}.tmp.mp4`], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((res, rej) => ff.on("close", c => c ? rej(new Error(`ffmpeg ${sc.id} exit ${c}`)) : res()));
  for (let f = 0; f < n; f++) {
    await page.evaluate(t => window.__seek(t), f / FPS);
    const jpg = await page.screenshot({ type: "jpeg", quality: 92 });
    if (!ff.stdin.write(jpg)) await new Promise(r => ff.stdin.once("drain", r));
  }
  ff.stdin.end(); await done; await page.close();
  if (errs.length) log(`render ${sc.id}: page error ${errs.join(" | ").slice(0, 300)}`);
  fs.renameSync(`${out}.tmp.mp4`, out); fs.writeFileSync(`${out}.stamp`, stamp);
  log(`render ${sc.id}: ${n} เฟรม ${sc.dur.toFixed(2)} วินาที`);
}
async function render(ids) {
  fs.mkdirSync(`${B}/v`, { recursive: true });
  const tl = timeline().filter(s => !ids.length || ids.includes(s.id)), browser = await playwright().chromium.launch();
  const queue = [...tl];
  await Promise.all(Array.from({ length: JOBS }, async () => { for (let sc; (sc = queue.shift());) await renderScene(browser, sc); }));
  await browser.close();
}

// ---- captions: word-wrapped cues timed by the alignment; tags are never shown, {cap|say} pieces show cap
const assTime = t => { const cs = Math.max(0, Math.round(t * 100)); return `${Math.floor(cs / 360000)}:${String(Math.floor(cs / 6000) % 60).padStart(2, "0")}:${String(Math.floor(cs / 100) % 60).padStart(2, "0")}.${String(cs % 100).padStart(2, "0")}`; };
const esc = t => t.replace(/[\\{}]/g, m => "\\" + m);
function cues(sc) {
  const seg = new Intl.Segmenter("th", { granularity: "word" }), out = [], st = sc.al.character_start_times_seconds, en = sc.al.character_end_times_seconds, off = sc.t0 + sc.at0;
  let cur = null;
  const flush = () => { if (cur?.text.trim()) out.push(cur); cur = null; };
  const add = (w, a, b) => {
    const latin = /^[A-Za-z0-9]/.test(w), sep = cur && !cur.text.endsWith(" ") && (latin || cur.latin) ? " " : "";
    if (cur && vlen(cur.text + sep + w) > MAXLINE) flush();
    cur ??= { text: "", s: off + st[a] };
    cur.text += (cur.text ? sep : "") + w; cur.e = off + en[b - 1]; cur.latin = latin;
  };
  for (const r of sc.parsed.runs) {
    if (r.cap) { add(r.cap, r.a, r.b); continue; }
    for (const w of seg.segment(sc.parsed.tts.slice(r.a, r.b))) {
      const i = r.a + w.index;
      // ไม้ยมก belongs to the word before it, so never break a line in front of "ๆ"
      if (/^\s+$/.test(w.segment)) { if (cur && vlen(cur.text) >= 24 && sc.parsed.tts[i + w.segment.length] !== "ๆ") flush(); else if (cur && !cur.text.endsWith(" ")) cur.text += " "; continue; }
      add(w.segment, i, i + w.segment.length);
    }
  }
  flush();
  out.forEach((c, k) => { c.text = c.text.trim(); const nx = out[k + 1]?.s ?? Infinity; c.end = Math.min(nx - 0.02, Math.max(c.e + 0.25, c.s + 0.8)); });
  return out;
}
const ASS = `[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,Leelawadee UI,58,&H00FFFFFF,&H00FFFFFF,&H00100A05,&H96000000,1,0,0,0,100,100,0,0,1,4,2,2,120,120,52,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;

// ---- mix: concat scene clips, place each take at its scene start, duck the bed under the voice, burn captions, YouTube encode
function wav48(pcm) { // 48 kHz mono 16-bit
  const h = Buffer.alloc(44); h.write("RIFF", 0); h.writeUInt32LE(36 + pcm.length, 4); h.write("WAVEfmt ", 8); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(48000, 24); h.writeUInt32LE(BPS, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34); h.write("data", 36); h.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([h, pcm]);
}
// each footage scene's slice of its take's call.webm (caller + javis), at the same take seconds its frames show, 0.25 s fades at the cuts
function callTrack(tl, T) {
  const out = Buffer.alloc(Math.round(T * 48000) * 2), raw = {};
  for (const s of tl.filter(s => s.cap)) {
    const dir = `${B}/cap/${s.cap.take}`, take = JSON.parse(fs.readFileSync(`${dir}/take.json`, "utf8"));
    if (!fs.existsSync(`${dir}/call.pcm`)) spawnSync("ffmpeg", ["-y", "-v", "error", "-i", `${dir}/call.webm`, "-ac", "1", "-ar", "48000", "-f", "s16le", `${dir}/call.pcm`], { stdio: "inherit" });
    const src = raw[s.cap.take] ??= fs.readFileSync(`${dir}/call.pcm`), off = (take.audioStart - take.recStart) / 1000; // file second 0 = take second off
    const t1 = Math.max(0, off - s.cap.base), t2 = Math.min(s.dur, s.cap.to - s.cap.base, src.length / BPS + off - s.cap.base), fade = 0.25 * 48000;
    const n = Math.round((t2 - t1) * 48000), a = Math.round((s.cap.base + t1 - off) * 48000), o = Math.round((s.t0 + t1) * 48000);
    for (let i = 0; i < n; i++) out.writeInt16LE(Math.round(src.readInt16LE((a + i) * 2) * Math.min(1, i / fade, (n - i) / fade)), (o + i) * 2);
  }
  return out;
}
function ff(args) { const r = spawnSync("ffmpeg", ["-y", "-v", "error", ...args], { cwd: B, stdio: "inherit" }); if (r.status) die(`ffmpeg exit ${r.status}`); }
function mix() {
  const tl = timeline(), T = tl.at(-1).t0 + tl.at(-1).dur;
  for (const s of tl) if (!fs.existsSync(`${B}/v/${s.id}.mp4`)) die(`ยังไม่ได้ render ${s.id}`);
  const pcm = Buffer.alloc(Math.round(T * 48000) * 2);
  for (const s of tl) fs.readFileSync(s.pcm).copy(pcm, Math.round((s.t0 + s.at0) * 48000) * 2);
  fs.writeFileSync(`${B}/narration.wav`, wav48(pcm));
  fs.writeFileSync(`${B}/calls.wav`, wav48(callTrack(tl, T)));
  fs.writeFileSync(`${B}/captions.ass`, ASS + tl.flatMap(cues).map(c => `Dialogue: 0,${assTime(c.s)},${assTime(c.end)},Sub,,0,0,0,,${esc(c.text)}\n`).join(""));
  fs.writeFileSync(`${B}/list.txt`, tl.map(s => `file 'v/${s.id}.mp4'`).join("\n") + "\n");
  fs.writeFileSync(`${B}/mix.txt`, [bedGraph([{ a: 0, b: T, mood: "fun" }]),
    "[0:v]scale=out_range=tv,ass=captions.ass:fontsdir='C\\:/Windows/Fonts':shaping=complex,format=yuv420p[vout]",
    "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=3[n1][n2][n3]", "[2:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit[c1][c2]",
    "[c1][n2]sidechaincompress=threshold=0.01:ratio=20:attack=20:release=400[cd]", // the call sits far under the narration, full level alone
    "[n3][c2]amix=inputs=2:normalize=0[key]", "[bed]volume=-4dB[bq]", "[bq][key]sidechaincompress=threshold=0.02:ratio=6:attack=30:release=600[duck]",
    `[n1][cd][duck]amix=inputs=3:duration=longest:normalize=0,afade=t=in:d=0.5,afade=t=out:st=${(T - 1.5).toFixed(3)}:d=1.5,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]`].join(";\n"));
  fs.mkdirSync(OUT, { recursive: true });
  ff(["-f", "concat", "-safe", "0", "-i", "list.txt", "-i", "narration.wav", "-i", "calls.wav", "-/filter_complex", "mix.txt", "-map", "[vout]", "-map", "[aout]", "-t", T.toFixed(3), "-r", String(FPS),
    "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "18", "-maxrate", "16M", "-bufsize", "32M", "-g", "15", "-bf", "2", "-flags", "+cgop", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "384k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", `${OUT}/javis-youtube.tmp.mp4`]);
  fs.renameSync(`${OUT}/javis-youtube.tmp.mp4`, `${OUT}/javis-youtube.mp4`);
  log(`mix: ${OUT}/javis-youtube.mp4 ยาว ${Math.floor(T / 60)}:${String(Math.round(T % 60)).padStart(2, "0")} นาที`);
}

// ---- self-check of the markup parser (no network): node video/make.mjs check
function check() {
  const p = parse("[playful] ก่อน ^ใช้ {100 แต้ม|หนึ่งร้อยแต้ม} ที่จาร์วิส");
  const want = "[playful] ก่อน ใช้ หนึ่งร้อยแต้ม ที่จาร์วิส";
  if (p.tts !== want) die(`parse tts: ${p.tts}`);
  if (p.anchors.length !== 1 || p.tts.slice(p.anchors[0], p.anchors[0] + 3) !== "ใช้") die("parse anchor");
  if (!p.runs.some(r => r.cap === "100 แต้ม") || !p.runs.some(r => r.cap === "javis" && p.tts.slice(r.a, r.b) === "จาร์วิส")) die("parse runs");
  if (p.runs.some(r => p.tts.slice(r.a, r.b).includes("["))) die("tag leaked into a caption run");
  log("check ok");
}

const main = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href; // importing parse() must not start a job
const [cmd = "all", ...rest] = process.argv.slice(2);
if (!main);
else if (cmd === "check") check();
else if (cmd === "tts") await tts();
else if (cmd === "render") await render(rest);
else if (cmd === "mix") mix();
else if (cmd === "times") for (const s of timeline()) log(s.id, s.t0.toFixed(1), s.dur.toFixed(1), s.p.join(" "));
else if (cmd === "all") { await tts(); await render([]); mix(); }
else die(`ไม่รู้จักคำสั่ง ${cmd}`);
