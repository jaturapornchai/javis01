// javis video scene runtime (from D:/domecard/video/scenes/common.js). make.mjs injects window.SCENE = { dur, p: [paragraph starts, s], cap } before
// load and awaits __seek(t) once per frame; a scene opened by hand in a browser previews in real time with the fallback timing below (no footage).
(() => {
  const S = window.SCENE ?? { dur: 20, p: [0.5, 5, 10, 15], preview: true };
  const root = document.documentElement;
  S.p.forEach((v, i) => root.style.setProperty(`--p${i + 1}`, v));
  root.style.setProperty("--dur", S.dur);
  root.style.setProperty("--h", S.cap?.h ?? S.dur);
  // "p2+0.5" / "h-1" -> seconds (data-at on counters, data-cam keys); h = start of the clip-only part of a footage scene
  const at = s => {
    const m = String(s ?? "0").trim().match(/^(?:p(\d+)|(h))\s*([+-]\s*[\d.]+)?$/);
    return m ? (m[2] ? S.cap?.h ?? 0 : S.p[m[1] - 1]) + Number((m[3] ?? "0").replace(/\s/g, "")) : Number(s);
  };
  const ease = x => 1 - (1 - x) ** 3, inOut = x => (x < .5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2);
  const fmt = (v, dec) => v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });

  // footage camera: <div class="cam" data-cam="0:full; p2:team; h+1:subs"> eases to each region of the 1920x1080 page and keeps a slow push-in.
  // Regions are boxes of static/index.html at 1920x1080 (capture.mjs take.json boxes); "x y w h" works too.
  const REG = { full: [0, 0, 1920, 1080], team: [6, 60, 300, 250], left: [6, 60, 560, 600], subs: [380, 880, 1100, 130], panel: [300, 64, 1280, 840],
    ptop: [300, 64, 1280, 470], pbottom: [300, 470, 1280, 440], cards: [1570, 60, 345, 640], hud: [1680, 850, 240, 230], center: [300, 64, 1290, 960] };
  // ponytail: zoom capped at 2.2x, the screencast is 1920 wide and text past that goes soft. The region is centred at y 470 (above the captions)
  // and the page may lift up to 220 px off the bottom edge, so the page's own subtitle bar is not hidden under the burned captions.
  const fit = r => {
    const s = Math.min(2.2, Math.max(1, Math.min(1920 / r[2], 1080 / r[3]) * 0.9)), lift = Math.min(220, (s - 1) * 400);
    const x = Math.min(0, Math.max(1920 - 1920 * s, 960 - (r[0] + r[2] / 2) * s)), y = Math.min(0, Math.max(1080 - 1080 * s - lift, 470 - (r[1] + r[3] / 2) * s));
    return [s, x, y];
  };
  let cams, capImg, capSrc;
  const camAt = (keys, t) => {
    let k = 0; keys.forEach((x, i) => { if (t >= x.t) k = i; });
    const to = keys[k].v, from = keys[Math.max(0, k - 1)].v, e = t < keys[0].t ? 0 : inOut(Math.min(1, Math.max(0, (t - keys[k].t) / 1.1)));
    const v = from.map((a, i) => a + (to[i] - a) * e), push = 1 + 0.012 * Math.max(0, t - keys[k].t); // Ken Burns drift about the frame centre
    return `translate(960px,540px) scale(${push}) translate(-960px,-540px) translate(${v[1]}px,${v[2]}px) scale(${v[0]})`;
  };
  const frameAt = c => { const f = S.cap.f; let lo = 0, hi = f.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (f[m] <= c) lo = m; else hi = m - 1; } return lo; };

  let anims, counters, steps;
  window.__seek = t => {
    anims ??= document.getAnimations();
    for (const a of anims) { a.pause(); a.currentTime = t * 1000; }
    // <b data-to="1250" data-from="0" data-at="p2+0.5" data-dur="1.2" data-dec="2"></b>: display-only count-up
    counters ??= [...document.querySelectorAll("[data-to]")];
    for (const el of counters) {
      const a0 = at(el.dataset.at), d = Number(el.dataset.dur ?? 1.2), from = Number(el.dataset.from ?? 0), to = Number(el.dataset.to);
      el.textContent = fmt(from + (to - from) * ease(Math.min(1, Math.max(0, (t - a0) / d))), Number(el.dataset.dec ?? 0));
    }
    // <b data-steps="0:1,p3+1:2">: text switches at each time and pops
    steps ??= [...document.querySelectorAll("[data-steps]")].map(el => ({ el, st: el.dataset.steps.split(",").map(x => { const i = x.lastIndexOf(":"); return { t: at(x.slice(0, i)), v: x.slice(i + 1) }; }) }));
    for (const { el, st } of steps) {
      let k = -1; st.forEach((x, i) => { if (t >= x.t) k = i; });
      el.textContent = st[Math.max(k, 0)].v;
      const d = k > 0 ? (t - st[k].t) / 0.45 : 1;
      el.style.transform = d < 1 ? `scale(${1 + 0.45 * Math.sin(Math.PI * d)})` : "";
    }
    cams ??= [...document.querySelectorAll("[data-cam]")].map(el => ({ el, keys: el.dataset.cam.split(";").map(x => {
      const i = x.indexOf(":"), r = x.slice(i + 1).trim();
      return { t: at(x.slice(0, i)), v: fit(REG[r] ?? r.split(/\s+/).map(Number)) };
    }) }));
    for (const { el, keys } of cams) el.style.transform = camAt(keys, t);
    window.onSeek?.(t, S);
    // footage: the take's screencast frame for take second min(base + t, to); resolves once it is decoded
    capImg ??= document.querySelector("img.cap");
    if (!S.cap || !capImg) return;
    const src = `${S.cap.dir}/f${String(frameAt(Math.max(0, Math.min(S.cap.base + t, S.cap.to)))).padStart(5, "0")}.jpg`;
    if (src === capSrc) return;
    capSrc = src; capImg.src = src;
    return capImg.decode().catch(() => {});
  };
  if (S.preview) addEventListener("load", () => { const t0 = performance.now(); const tick = () => { __seek(((performance.now() - t0) / 1000) % S.dur); requestAnimationFrame(tick); }; tick(); });
})();
