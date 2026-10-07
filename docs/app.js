(function () {
  "use strict";

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var SVGNS = "http://www.w3.org/2000/svg";

  function icons() { if (window.lucide) window.lucide.createIcons(); }

  /* ---------- Nav: background on scroll, mobile menu ---------- */
  var nav = $("#nav");
  function onScroll() { nav.classList.toggle("scrolled", window.scrollY > 24); }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  var toggle = $(".nav-toggle");
  var menu = $(".nav nav");
  toggle.addEventListener("click", function () {
    toggle.setAttribute("aria-expanded", String(menu.classList.toggle("open")));
  });
  menu.addEventListener("click", function (e) {
    if (e.target.tagName === "A") { menu.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }
  });

  /* ---------- Typewriter for the three hero lines (types once) ---------- */
  (function () {
    var lines = $$(".ln-content[data-type]").map(function (host) {
      var full = host.getAttribute("data-type");
      var ghost = document.createElement("span");   // reserves the final size so nothing reflows
      ghost.className = "ln-ghost"; ghost.textContent = full; ghost.setAttribute("aria-hidden", "true");
      var text = document.createElement("span");
      text.className = "ln-text";
      var typed = document.createElement("span");
      text.appendChild(typed);
      host.appendChild(ghost); host.appendChild(text);
      host.setAttribute("aria-label", full);
      return { full: full, text: text, typed: typed };
    });
    if (!lines.length) return;
    if (reduceMotion.matches) { lines.forEach(function (l) { l.typed.textContent = l.full; }); return; }

    var caret = document.createElement("span");
    caret.className = "caret"; caret.setAttribute("aria-hidden", "true");
    var li = 0, ci = 0;
    function next() {
      if (li >= lines.length) { setTimeout(function () { caret.remove(); }, 2400); return; }
      lines[li].text.appendChild(caret);
      ci = 0; tick();
    }
    function tick() {
      var l = lines[li];
      l.typed.textContent = l.full.slice(0, ++ci);
      if (ci < l.full.length) setTimeout(tick, 16 + Math.random() * 26);
      else { li++; setTimeout(next, 520); }
    }
    setTimeout(next, 600);
  })();

  /* ---------- Teaser video: plays while on screen, with a pause button ---------- */
  $$("video.anim").forEach(function (video) {
    var btn = $(".anim-toggle", video.closest("figure"));
    var userPaused = reduceMotion.matches;
    video.muted = true;
    function play() { var p = video.play(); if (p && p.catch) p.catch(function () {}); }
    function paint() {
      var playing = !video.paused;
      btn.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
      btn.innerHTML = '<i data-lucide="' + (playing ? "pause" : "play") + '" aria-hidden="true"></i>';
      icons();
    }
    video.addEventListener("play", paint);
    video.addEventListener("pause", paint);
    btn.addEventListener("click", function () {
      userPaused = !video.paused;
      if (userPaused) video.pause(); else play();
    });
    btn.hidden = false; paint();
    if (!("IntersectionObserver" in window)) { if (!userPaused) play(); return; }
    new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { if (!userPaused) play(); }
        else if (!video.paused) video.pause();
      });
    }, { threshold: 0.2 }).observe(video);
  });

  /* ---------- Waveform shapes, shared by the hero stage and the demo card ---------- */
  function gauss(u, m, s) { var d = (u - m) / s; return Math.exp(-0.5 * d * d); }
  var wave = {
    ecg: function (x, period) {
      var u = (((x % period) + period) % period) / period;
      return 0.12 * gauss(u, 0.20, 0.03) - 0.14 * gauss(u, 0.285, 0.010) + gauss(u, 0.31, 0.012)
           - 0.26 * gauss(u, 0.34, 0.011) + 0.30 * gauss(u, 0.56, 0.05);
    },
    resp: function (x, period) { return Math.sin(2 * Math.PI * x / period) * (0.72 + 0.28 * Math.sin(2 * Math.PI * x / (period * 6.3))); },
    hr:   function (x) { return 0.55 * Math.sin(x / 150) + 0.3 * Math.sin(x / 47 + 1) + 0.15 * Math.sin(x / 19 + 2); },
    cgm:  function (x) { return 0.7 * Math.sin(x / 230) + 0.3 * Math.sin(x / 90 + 0.6); }
  };

  /* ---------- Hero stage: a monitor strip. Four channels scroll past a decision time t0. A typed question
     touches the strip at t0 and the signal blooms open around it; each capability is an analysis window
     drawn on the trace it reads from. ---------- */
  (function () {
    var stage = $(".stage");
    if (!stage) return;
    var canvas = $(".traces", stage), ctx = canvas.getContext("2d");
    var marks = $$(".win", stage), chans = $$(".ch", stage), t0el = $(".t0", stage);
    var probe = $(".probe", stage), probeText = $(".probe-text", probe), QUESTION = "What action follows?";
    var LEFT = 86, RIGHT = 14, T0 = 0.68;                       // channel column, right inset, decision time (fraction of the trace span)
    var LANES = [
      { key: "ecg",  y: 0.21,  color: "#3f7fc4", amp: 0.19,  speed: 34, up: 0.19, dn: 0.08, fn: function (x) { return wave.ecg(x, 128) - 0.3; } },
      { key: "resp", y: 0.43,  color: "#4f9d69", amp: 0.075, speed: 20, up: 0.1,  dn: 0.1,  fn: function (x) { return wave.resp(x, 150); } },
      { key: "hr",   y: 0.645, color: "#d9577a", amp: 0.06,  speed: 12, up: 0.09, dn: 0.09, fn: wave.hr, beads: 30 },
      { key: "cgm",  y: 0.855, color: "#dd8a2f", amp: 0.07,  speed: 8,  up: 0.1,  dn: 0.1,  fn: wave.cgm, beads: 16, beadsOnly: true }
    ];
    var MARKS = {                                               // x and w are fractions of the trace span
      actions:  { lane: 0, x: 0.05, w: 0.19 },
      unseen:   { lane: 0, x: 0.47, w: 0.14, dashed: true },
      state:    { lane: 1, x: 0.30, w: 0.17 },
      evidence: { lane: 2, x: 0.11, w: 0.18 },
      cohort:   { lane: 2, x: 0.49, w: 0.15, ghost: true },
      future:   { lane: 3, x: T0 + 0.015, w: 1 - T0 - 0.015, future: true }
    };
    var w = 0, h = 0, active = null, spot = null;
    var reveal = { phase: "fog", r: 0 };                        // fog → typing → bloom → done
    var invite = $(".invite", stage);
    var FOG = [];                                               // drifting patches of fog over the strip
    for (var fi = 0; fi < 9; fi++) FOG.push({ x: Math.random(), y: Math.random(), r: 0.18 + Math.random() * 0.22, s: 0.004 + Math.random() * 0.006, p: Math.random() * 6.28 });

    function geom(m) {
      var L = LANES[m.lane], span = w - LEFT - RIGHT, cy = L.y * h;
      return { x: LEFT + m.x * span, y: cy - L.up * h, w: m.w * span, h: (L.up + L.dn) * h, cy: cy, lane: L };
    }
    function layout() {
      var dpr = window.devicePixelRatio || 1;
      w = stage.clientWidth; h = stage.clientHeight;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      marks.forEach(function (el) {
        var m = MARKS[el.getAttribute("data-mark")], g = geom(m);
        el.style.left = g.x + "px"; el.style.top = g.y + "px"; el.style.width = g.w + "px"; el.style.height = g.h + "px";
        el.style.setProperty("--mc", g.lane.color); el.style.setProperty("--mcl", g.lane.color + "55"); el.style.setProperty("--mx", g.x);
      });
      chans.forEach(function (el, i) { el.style.setProperty("--y", LANES[i].y); });
      var tx = (LEFT + T0 * (w - LEFT - RIGHT)) / w;
      stage.style.setProperty("--x", tx);            // t0 marker, probe, invitation all hang off it
    }

    function roundRect(x, y, rw, rh, r) {
      ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + rw, y, x + rw, y + rh, r); ctx.arcTo(x + rw, y + rh, x, y + rh, r);
      ctx.arcTo(x, y + rh, x, y, r); ctx.arcTo(x, y, x + rw, y, r); ctx.closePath();
    }
    function trace(L, off, x0, x1, alpha, width, dash, fn, color) {
      var cy = L.y * h, a = L.amp * h, x, f = fn || L.fn;
      ctx.globalAlpha = alpha; ctx.strokeStyle = color || L.color; ctx.fillStyle = color || L.color;
      ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.setLineDash(dash || []);
      if (!L.beadsOnly) {
        ctx.beginPath();
        for (x = x0; x <= x1; x += 2) { var y = cy - a * f(x + off); if (x === x0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (L.beads && !dash) {
        var start = x0 + ((-(off % L.beads)) % L.beads + L.beads) % L.beads;
        for (x = start; x <= x1; x += L.beads) { ctx.beginPath(); ctx.arc(x, cy - a * f(x + off), L.beadsOnly ? 2.3 : 2.6, 0, 6.2832); ctx.fill(); }
      }
    }

    function drawPaper() {                                       // a sheet of recording paper
      roundRect(0.5, 0.5, w - 1, h - 1, 14);
      ctx.fillStyle = "rgba(255,255,255,.62)"; ctx.fill();
      ctx.save(); ctx.clip();
      ctx.strokeStyle = "rgba(102,76,188,.07)"; ctx.lineWidth = 1;
      for (var gx = LEFT; gx < w; gx += 12) { ctx.globalAlpha = ((gx - LEFT) % 60 === 0) ? 1 : 0.45; ctx.beginPath(); ctx.moveTo(gx + 0.5, 0); ctx.lineTo(gx + 0.5, h); ctx.stroke(); }
      for (var gy = 0; gy < h; gy += 12) { ctx.globalAlpha = (gy % 60 === 0) ? 1 : 0.45; ctx.beginPath(); ctx.moveTo(LEFT, gy + 0.5); ctx.lineTo(w, gy + 0.5); ctx.stroke(); }
      ctx.globalAlpha = 1;
      if (spot) {                                                 // a soft light under the reader's pointer
        var g = ctx.createRadialGradient(spot.x, spot.y, 0, spot.x, spot.y, 110);
        g.addColorStop(0, "rgba(255,255,255,.8)"); g.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = g; ctx.fillRect(spot.x - 110, spot.y - 110, 220, 220);
      }
      ctx.restore();
      roundRect(0.5, 0.5, w - 1, h - 1, 14); ctx.strokeStyle = "rgba(102,76,188,.14)"; ctx.lineWidth = 1; ctx.stroke();
    }
    function drawRaw(t) {                                        // before language touches it: the signal is there, barely legible
      LANES.forEach(function (L) { trace(L, t * L.speed, LEFT, w - RIGHT, 0.16, 1.2); });
    }
    function drawFog(t, hole) {                                  // a veil over the paper, with a clear circle once the question lands
      ctx.save(); roundRect(0.5, 0.5, w - 1, h - 1, 14); ctx.clip();
      if (hole > 0) { ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.arc(w * 0 + hole.x, hole.y, hole.r, 0, 6.2832, true); ctx.clip("evenodd"); }
      ctx.globalAlpha = 1; ctx.fillStyle = "rgba(247,245,252,.78)"; ctx.fillRect(0, 0, w, h);
      FOG.forEach(function (f) {
        var cx = ((f.x + Math.sin(t * f.s * 6 + f.p) * 0.06) % 1) * w, cy = ((f.y + Math.cos(t * f.s * 5 + f.p) * 0.05) % 1) * h, r = f.r * w;
        var g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g.addColorStop(0, "rgba(255,255,255,.75)"); g.addColorStop(0.6, "rgba(244,240,250,.35)"); g.addColorStop(1, "rgba(244,240,250,0)");
        ctx.fillStyle = g; ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);
      });
      ctx.restore();
      if (hole > 0) {                                             // a soft, bright rim where the fog parts
        var rim = ctx.createRadialGradient(hole.x, hole.y, Math.max(0, hole.r - 26), hole.x, hole.y, hole.r + 2);
        rim.addColorStop(0, "rgba(255,255,255,0)"); rim.addColorStop(1, "rgba(255,255,255,.9)");
        ctx.save(); roundRect(0.5, 0.5, w - 1, h - 1, 14); ctx.clip(); ctx.fillStyle = rim; ctx.beginPath(); ctx.arc(hole.x, hole.y, hole.r + 2, 0, 6.2832); ctx.fill(); ctx.restore();
      }
    }
    function drawLive(t) {
      var span = w - LEFT - RIGHT, x0 = LEFT, xt = LEFT + T0 * span, x1 = w - RIGHT, dimOthers = !!active;
      ctx.save(); roundRect(0.5, 0.5, w - 1, h - 1, 14); ctx.clip();
      ctx.fillStyle = "rgba(102,76,188,.035)"; ctx.fillRect(xt, 0, x1 - xt + RIGHT, h);     // the future, right of t0
      ctx.restore();
      marks.forEach(function (el) {                               // analysis windows
        var m = MARKS[el.getAttribute("data-mark")], g = geom(m), on = el === active;
        ctx.globalAlpha = on ? 0.14 : dimOthers ? 0.04 : 0.075; ctx.fillStyle = g.lane.color;
        roundRect(g.x, g.y, g.w, g.h, 8); ctx.fill();
        if (m.dashed || on) { ctx.globalAlpha = on ? 0.9 : 0.5; ctx.strokeStyle = g.lane.color; ctx.lineWidth = on ? 1.5 : 1; ctx.setLineDash(m.dashed && !on ? [5, 4] : []); ctx.stroke(); ctx.setLineDash([]); }
      });
      LANES.forEach(function (L) {                                // the traces: observed up to t0, dashed beyond
        var off = t * L.speed, base = dimOthers ? 0.22 : 0.42;
        trace(L, off, x0, xt, base, 1.5);
        if (spot && !dimOthers) {                                  // legible where the pointer is
          ctx.save(); ctx.beginPath(); ctx.arc(spot.x, spot.y, 80, 0, 6.2832); ctx.clip();
          trace(L, off, Math.max(x0, spot.x - 84), Math.min(xt, spot.x + 84), 0.95, 2); ctx.restore();
        }
        if (L.beadsOnly) {                                         // CGM: the model reads out the next two hours
          var cy = L.y * h, a = L.amp * h, x, pts = [];
          for (x = xt; x <= x1; x += 4) pts.push([x, cy - a * L.fn(x + off), 2 + 12 * (x - xt) / (x1 - xt)]);
          ctx.globalAlpha = dimOthers ? 0.08 : 0.14; ctx.fillStyle = L.color; ctx.beginPath();
          pts.forEach(function (p, k) { if (k === 0) ctx.moveTo(p[0], p[1] - p[2]); else ctx.lineTo(p[0], p[1] - p[2]); });
          for (var k = pts.length - 1; k >= 0; k--) ctx.lineTo(pts[k][0], pts[k][1] + pts[k][2]);
          ctx.closePath(); ctx.fill();
          ctx.globalAlpha = dimOthers ? 0.3 : 0.6; ctx.strokeStyle = L.color; ctx.lineWidth = 1.5; ctx.setLineDash([4, 5]);
          ctx.beginPath(); pts.forEach(function (p, k) { if (k === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }); ctx.stroke(); ctx.setLineDash([]);
        } else {
          trace(L, off, xt, x1, 0.14, 1.2, [3, 6]);
        }
      });
      marks.forEach(function (el) {                               // the window's own segment, bright
        var m = MARKS[el.getAttribute("data-mark")], g = geom(m), on = el === active, L = g.lane, off = t * L.speed;
        if (m.future) return;
        ctx.save(); roundRect(g.x, g.y, g.w, g.h, 8); ctx.clip();
        trace(L, off, g.x - 4, g.x + g.w + 4, on || !dimOthers ? 1 : 0.5, on ? 2.6 : 2.1);
        if (m.ghost) trace(L, off, g.x - 4, g.x + g.w + 4, on ? 0.7 : 0.45, 1.6, [6, 4], function (x) { return 0.85 * wave.hr(x + 420) - 0.15; }, "#6b6477");
        ctx.restore();
      });
      ctx.globalAlpha = 1; ctx.strokeStyle = "rgba(79,56,160,.5)"; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);   // decision time
      ctx.beginPath(); ctx.moveTo(xt + 0.5, 48); ctx.lineTo(xt + 0.5, h - 6); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#4f38a0"; ctx.beginPath(); ctx.moveTo(xt - 4, 41); ctx.lineTo(xt + 4, 41); ctx.lineTo(xt, 47); ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }
    function draw(t) {
      ctx.clearRect(0, 0, w, h);
      drawPaper();
      if (reveal.phase === "done") { drawLive(t); return; }
      drawRaw(t);
      var cx = LEFT + T0 * (w - LEFT - RIGHT), cy = h / 2;
      if (reveal.phase !== "bloom") { drawFog(t, 0); return; }
      ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, reveal.r, 0, 6.2832); ctx.clip(); drawLive(t); ctx.restore();
      drawFog(t, { x: cx, y: cy, r: reveal.r });
      ctx.globalAlpha = Math.max(0, 0.7 - reveal.r / (w * 0.9)); ctx.strokeStyle = "#664cbc"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, reveal.r, 0, 6.2832); ctx.stroke(); ctx.globalAlpha = 1;
      marks.forEach(function (el) {                               // windows pop in as the bloom reaches them
        var g = geom(MARKS[el.getAttribute("data-mark")]);
        if (Math.hypot(g.x + g.w / 2 - cx, g.cy - cy) <= reveal.r) el.classList.add("in");
      });
    }

    var tLast = -1;
    function readouts(t) {                                       // live numbers at t0, from the same waveforms
      if (t - tLast < 0.9) return; tLast = t;
      var xt = LEFT + T0 * (w - LEFT - RIGHT);
      var hr = Math.round(94 + 9 * wave.hr(xt + t * 12)), rr = Math.round(19 + 3 * wave.resp(xt + t * 20, 150)), gl = Math.round(142 + 26 * wave.cgm(xt + t * 8));
      chans.forEach(function (el) {
        var v = $(".ch-val", el);
        switch (el.getAttribute("data-ch")) {
          case "ecg": v.textContent = (hr + 2) + " bpm proxy"; break;
          case "resp": v.textContent = rr + " /min"; break;
          case "hr": v.textContent = hr + " bpm"; break;
          case "cgm": v.textContent = gl + " mg/dL"; break;
        }
      });
    }

    // the opening: a question is typed at t0, touches the strip, and the signal blooms open around it
    var bloomStart = 0, typeTimer = 0;
    function finish() {
      reveal.phase = "done"; marks.forEach(function (el) { el.classList.add("in"); });
      stage.classList.add("revealed"); stage.classList.remove("fog"); probe.classList.remove("on", "touch"); probe.classList.add("docked");
    }
    function fog() {                                             // back under the veil, with the invitation
      clearTimeout(typeTimer);
      reveal.phase = "fog"; reveal.r = 0; bloomStart = 0; select(null);
      marks.forEach(function (el) { el.classList.remove("in"); });
      stage.classList.remove("revealed"); stage.classList.add("fog"); probe.classList.remove("on", "touch", "docked");
    }
    function play() {
      clearTimeout(typeTimer);
      if (reduceMotion.matches) { finish(); draw(0); return; }
      reveal.phase = "typing"; reveal.r = 0; bloomStart = 0;
      marks.forEach(function (el) { el.classList.remove("in"); });
      stage.classList.remove("revealed", "fog"); probe.classList.remove("docked", "touch"); probe.classList.add("on");
      probeText.textContent = "";
      var ci = 0;
      (function tick() {
        probeText.textContent = QUESTION.slice(0, ++ci);
        if (ci < QUESTION.length) typeTimer = setTimeout(tick, 42 + Math.random() * 30);
        else typeTimer = setTimeout(function () { probe.classList.add("touch"); typeTimer = setTimeout(function () { reveal.phase = "bloom"; }, 350); }, 320);
      })();
    }
    function frame(ms) {
      if (reveal.phase === "bloom") {
        if (!bloomStart) bloomStart = ms;
        var p = Math.min((ms - bloomStart) / 1700, 1), e = 1 - Math.pow(1 - p, 3);
        reveal.r = e * Math.hypot(w, h) * 0.75;
        if (p >= 1) finish();
      }
      draw(ms / 1000); readouts(ms / 1000);
      if (!reduceMotion.matches) requestAnimationFrame(frame);
    }
    function start() { layout(); if (reduceMotion.matches) { finish(); draw(0); readouts(1); } else { stage.classList.add("fog"); requestAnimationFrame(frame); } }
    window.addEventListener("resize", function () { layout(); if (reduceMotion.matches) draw(0); }, { passive: true });
    reduceMotion.addEventListener("change", start);
    start();

    function select(el) {
      active = el;
      marks.forEach(function (m) { m.classList.toggle("active", m === el); });
      stage.classList.toggle("has-active", !!el);
      if (reduceMotion.matches) draw(0);
    }
    marks.forEach(function (el) {
      el.addEventListener("mouseenter", function () { select(el); });
      el.addEventListener("focus", function () { select(el); });
      el.addEventListener("blur", function () { select(null); });
      el.addEventListener("click", function (e) { e.stopPropagation(); select(el); });
    });
    t0el.addEventListener("click", function (e) { e.stopPropagation(); if (reveal.phase === "done") fog(); });
    stage.addEventListener("click", function (e) { if (reveal.phase === "fog") { e.stopPropagation(); play(); } });
    invite.addEventListener("keydown", function (e) { if ((e.key === "Enter" || e.key === " ") && reveal.phase === "fog") { e.preventDefault(); play(); } });
    stage.addEventListener("pointermove", function (e) { var r = stage.getBoundingClientRect(); spot = { x: e.clientX - r.left, y: e.clientY - r.top }; });
    stage.addEventListener("pointerleave", function () { spot = null; select(null); });
    document.addEventListener("click", function () { select(null); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") select(null); });
  })();

  /* ---------- Reveal on scroll ---------- */
  var reveals = $$(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en, idx) {
        if (!en.isIntersecting) return;
        en.target.style.transitionDelay = (Math.min(idx, 4) * 70) + "ms";
        en.target.classList.add("in");
        io.unobserve(en.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- Count-up stats ---------- */
  (function () {
    var nums = $$(".num[data-count]");
    if (!nums.length || reduceMotion.matches || !("IntersectionObserver" in window)) return;
    function run(el) {
      var target = parseFloat(el.getAttribute("data-count"));
      var suffix = el.getAttribute("data-suffix") || "";
      var start = null, dur = 1300;
      function frame(ts) {
        if (start === null) start = ts;
        var p = Math.min((ts - start) / dur, 1);
        el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))) + suffix;
        if (p < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    }
    var io2 = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { run(en.target); io2.unobserve(en.target); } });
    }, { threshold: 0.5 });
    nums.forEach(function (el) { io2.observe(el); });
  })();

  /* ---------- Chat: a conversation about one window. The reader (or a timer) asks; the answers stream in. ---------- */
  (function () {
    var chat = $("#chat");
    if (!chat) return;
    var log = $(".chat-log", chat), chips = $$(".chip", chat), replay = $(".replay", chat);
    var ORDER = ["state", "action", "why"];
    var caret = document.createElement("span"); caret.className = "caret"; caret.setAttribute("aria-hidden", "true");
    var run = 0, timer = 0, asked = {}, busy = false;

    function el(cls, tag) { var n = document.createElement(tag || "div"); n.className = cls; return n; }
    function down() { log.scrollTop = log.scrollHeight; }
    function setBusy(b) { busy = b; chat.classList.toggle("busy", b); }
    function userMsg(text, withAttach) {
      var m = el("msg user"), av = el("avatar"), b = el("bubble");
      av.textContent = "You";
      if (withAttach) { b.appendChild($("#chat-attach").content.cloneNode(true)); var im = $("img", b); if (im) im.addEventListener("load", down); }
      var p = document.createElement("p"); p.textContent = text; b.appendChild(p);
      m.appendChild(av); m.appendChild(b); log.appendChild(m); down();
    }
    function botBubble() {
      var m = el("msg bot"), av = el("avatar"), b = el("bubble");
      av.textContent = "S";
      b.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
      m.appendChild(av); m.appendChild(b); log.appendChild(m); down();
      return b;
    }
    function stream(b, id, done) {
      var token = run;
      b.innerHTML = ""; b.appendChild($("#turn-" + id).content.cloneNode(true));
      var leaves = $$("[data-type]", b);
      if (reduceMotion.matches) { leaves.forEach(function (l) { l.textContent = l.getAttribute("data-type"); }); down(); done(); return; }
      var li = 0;
      (function next() {
        if (token !== run) return;
        if (li >= leaves.length) { caret.remove(); done(); return; }
        var leaf = leaves[li++], full = leaf.getAttribute("data-type"), ci = 0;
        leaf.textContent = ""; leaf.appendChild(caret);
        (function tick() {
          if (token !== run) return;
          ci++; leaf.textContent = full.slice(0, ci); leaf.appendChild(caret); down();
          if (ci < full.length) timer = setTimeout(tick, 9 + Math.random() * 16);
          else timer = setTimeout(next, 220);
        })();
      })();
    }
    function ask(id) {
      if (asked[id] || busy) return;
      var invite = $(".chat-invite", log); if (invite) invite.remove();
      chat.classList.remove("idle");
      clearTimeout(timer); asked[id] = true; setBusy(true);
      var chip = chips.filter(function (c) { return c.getAttribute("data-turn") === id; })[0];
      chip.classList.add("done"); chip.disabled = true;
      userMsg(chip.textContent, Object.keys(asked).length === 1);
      var b = botBubble(), token = run;
      timer = setTimeout(function () {
        if (token !== run) return;
        stream(b, id, function () {
          setBusy(false);
          var nextId = ORDER.filter(function (k) { return !asked[k]; })[0];   // the conversation goes on by itself
          if (nextId) timer = setTimeout(function () { if (token === run) ask(nextId); }, reduceMotion.matches ? 0 : 1700);
        });
      }, reduceMotion.matches ? 0 : 850);
    }
    function reset() {
      run++; clearTimeout(timer); setBusy(false); asked = {}; log.innerHTML = "";
      chips.forEach(function (c) { c.classList.remove("done"); c.disabled = false; });
    }
    chips.forEach(function (c) { c.addEventListener("click", function () { ask(c.getAttribute("data-turn")); }); });
    replay.addEventListener("click", function () { reset(); ask("state"); });
    chat.classList.add("idle");
    $$(".chat-ring, .chat-start", chat).forEach(function (b) { b.addEventListener("click", function () { ask("state"); }); });
  })();

  /* ---------- Grounding: hover a caption phrase, the region it comes from lights up ----------
     Regions are hand-marked boxes on the figure crops, in image pixels. */
  var REGIONS = {
    fig2: {   // assets/fig2-signals.png, 1086 x 687
      "complaint": [104, 141, 304, 206],
      "ecg":       [358, 38, 1065, 137],  "ecg-box":  [652, 41, 728, 131],
      "resp":      [358, 192, 1065, 251],
      "rr":        [358, 254, 1065, 323], "rr-box":   [847, 258, 1047, 319],
      "hr":        [358, 319, 1065, 381],
      "pain-box":  [706, 374, 793, 416],
      "target":    [782, 560, 1054, 608]
    },
    fig4: {   // assets/fig4-signals.png, 1131 x 653
      "ecg":       [44, 13, 1110, 88],    "ecg-box":  [384, 13, 593, 91],
      "hr":        [44, 450, 1110, 516]
    },
    fig3: {   // assets/architecture.png, 2160 x 440
      "enc":  [[36, 306, 530, 428], [580, 306, 1090, 428]],
      "llm":  [[36, 118, 530, 266], [580, 118, 1090, 266]],
      "hm":   [[850, 268, 1100, 338]],
      "cmp":  [[1190, 36, 2150, 428]]
    }
  };
  $$(".ground").forEach(function (g) {
    var regions = REGIONS[g.getAttribute("data-ground")], svg = $(".ground-mask", g);
    if (!regions || !svg) return;
    var vb = svg.getAttribute("viewBox").split(" ").map(Number), W = vb[2], H = vb[3];
    var maskId = "m-" + g.getAttribute("data-ground");
    svg.innerHTML =
      '<defs><mask id="' + maskId + '"><rect width="' + W + '" height="' + H + '" fill="#fff"/><g class="holes"></g></mask></defs>' +
      '<rect class="dim" width="' + W + '" height="' + H + '" mask="url(#' + maskId + ')"/><g class="boxes"></g>';
    var holes = $(".holes", svg), boxes = $(".boxes", svg);
    var pinned = null, current = null;
    var img = $(".ground-stage img", g);
    var lines = document.createElementNS(SVGNS, "svg");             // leader lines, drawn in the block's own coordinates
    lines.setAttribute("class", "ground-lines"); lines.setAttribute("aria-hidden", "true");
    g.appendChild(lines);
    var stacked = window.matchMedia("(max-width: 1020px)");

    function connect(span, color) {
      lines.innerHTML = "";
      if (!span || stacked.matches) return;
      var gr = g.getBoundingClientRect(), sr = span.getBoundingClientRect(), ir = img.getBoundingClientRect();
      var sx = ir.width / W, sy = ir.height / H, pad = 6;
      var vertical = ir.bottom <= sr.top + 4;                          // figure above the text (How it works)
      var sideways = ir.right <= sr.left + 4;                          // figure to the left of the text
      if (!vertical && !sideways) return;
      var x0 = (sideways ? sr.left : sr.left + sr.width / 2) - gr.left, y0 = (sideways ? sr.top + sr.height / 2 : sr.top) - gr.top;
      span.getAttribute("data-to").split(",").forEach(function (key) {
        var r0 = regions[key.trim()]; if (!r0) return;
        (Array.isArray(r0[0]) ? r0 : [r0]).forEach(function (b) {
          var bx = ir.left - gr.left, by = ir.top - gr.top;
          var x1 = sideways ? bx + (b[2] + pad) * sx : bx + (b[0] + b[2]) / 2 * sx;
          var y1 = sideways ? by + (b[1] + b[3]) / 2 * sy : by + (b[3] + pad) * sy;
          var d = sideways
            ? "M" + x0 + " " + y0 + " C" + (x0 - Math.max(48, (x0 - x1) / 2)) + " " + y0 + ", " + (x1 + Math.max(48, (x0 - x1) / 2)) + " " + y1 + ", " + x1 + " " + y1
            : "M" + x0 + " " + y0 + " C" + x0 + " " + (y0 - Math.max(40, (y0 - y1) / 2)) + ", " + x1 + " " + (y1 + Math.max(40, (y0 - y1) / 2)) + ", " + x1 + " " + y1;
          var p = document.createElementNS(SVGNS, "path");
          p.setAttribute("d", d); p.setAttribute("stroke", color); p.setAttribute("pathLength", "1"); lines.appendChild(p);
          var dot = document.createElementNS(SVGNS, "circle");
          dot.setAttribute("cx", x1); dot.setAttribute("cy", y1); dot.setAttribute("r", 4); dot.setAttribute("fill", color); lines.appendChild(dot);
        });
      });
      var o = document.createElementNS(SVGNS, "circle");
      o.setAttribute("cx", x0); o.setAttribute("cy", y0); o.setAttribute("r", 4); o.setAttribute("fill", color); lines.appendChild(o);
    }
    function relink() { if (current) connect(current, getComputedStyle(current.closest(".gblock") || current).getPropertyValue("--acc").trim() || "#664cbc"); }
    window.addEventListener("scroll", relink, { passive: true });   // the figure is sticky, so the geometry moves while scrolling
    window.addEventListener("resize", relink, { passive: true });

    function show(span) {
      holes.innerHTML = ""; boxes.innerHTML = ""; current = span;
      if (!span) { svg.classList.remove("on"); lines.innerHTML = ""; return; }
      var color = getComputedStyle(span.closest(".gblock") || span).getPropertyValue("--acc").trim() || "#664cbc";
      span.getAttribute("data-to").split(",").forEach(function (key) {
        var r0 = regions[key.trim()]; if (!r0) return;
        (Array.isArray(r0[0]) ? r0 : [r0]).forEach(function (b) {
        var pad = 6;
        [["holes", "#000"], ["boxes", null]].forEach(function (t) {
          var r = document.createElementNS(SVGNS, "rect");
          r.setAttribute("x", b[0] - pad); r.setAttribute("y", b[1] - pad);
          r.setAttribute("width", b[2] - b[0] + 2 * pad); r.setAttribute("height", b[3] - b[1] + 2 * pad);
          r.setAttribute("rx", 8);
          if (t[1]) { r.setAttribute("fill", t[1]); holes.appendChild(r); }
          else { r.setAttribute("class", "box"); r.setAttribute("stroke", color); boxes.appendChild(r); }
        });
        });
      });
      svg.classList.add("on");
      connect(span, color);
    }
    // steps light the figure in turn until the reader points at one
    var auto = +g.getAttribute("data-auto");
    if (auto && !reduceMotion.matches && "IntersectionObserver" in window) {
      var spans = $$(".gs", g), k = 0, timer = 0;
      function step() { spans.forEach(function (s, i) { s.classList.toggle("auto", i === k); }); show(spans[k]); k = (k + 1) % spans.length; }
      function stop() { clearInterval(timer); timer = 0; spans.forEach(function (s) { s.classList.remove("auto"); }); if (!pinned) show(null); }
      new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) { if (!timer) { step(); timer = setInterval(step, auto); } } else stop();
      }, { threshold: 0.4 }).observe(g);
      g.addEventListener("pointerenter", stop);
    }
    $$(".gs", g).forEach(function (span) {
      span.setAttribute("tabindex", "0");
      span.addEventListener("mouseenter", function () { if (!pinned) show(span); });
      span.addEventListener("mouseleave", function () { if (!pinned) show(null); });
      span.addEventListener("focus", function () { if (!pinned) show(span); });
      span.addEventListener("blur", function () { if (!pinned) show(null); });
      span.addEventListener("click", function (e) {
        e.stopPropagation();
        if (pinned === span) { pinned = null; span.classList.remove("on"); show(span); return; }
        if (pinned) pinned.classList.remove("on");
        pinned = span; span.classList.add("on"); show(span);
      });
    });
    document.addEventListener("click", function () { if (pinned) { pinned.classList.remove("on"); pinned = null; show(null); } });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && pinned) { pinned.classList.remove("on"); pinned = null; show(null); } });
  });

  /* ---------- Alpha: slide along the interpolation path in Figure 6 ---------- */
  (function () {
    var box = $(".alpha");
    if (!box) return;
    var range = $(".alpha-range", box), val = $(".alpha-val", box), dose = $(".alpha-dose", box), tag = $(".alpha-tag", box);
    var halo = $(".alpha-halo", box), dot = $(".alpha-dot", box);
    // the four stops on the dashed path in panel (A), in image pixels of action-representations.png (2789 x 701)
    var STOPS = [
      { a: 0.0, x: 78,  y: 141, dose: "0.05 U", tag: "endpoint" },
      { a: 0.4, x: 343, y: 307, dose: "1.2 U",  tag: "retrieved" },
      { a: 0.8, x: 611, y: 465, dose: "7.7 U",  tag: "retrieved" },
      { a: 1.0, x: 750, y: 543, dose: "14.4 U", tag: "endpoint" }
    ];
    function place(a) {
      var i = 0; while (i < STOPS.length - 2 && a > STOPS[i + 1].a) i++;
      var s0 = STOPS[i], s1 = STOPS[i + 1], t = (a - s0.a) / (s1.a - s0.a);
      var x = s0.x + (s1.x - s0.x) * t, y = s0.y + (s1.y - s0.y) * t;
      [halo, dot].forEach(function (c) { c.setAttribute("cx", x.toFixed(1)); c.setAttribute("cy", y.toFixed(1)); });
      var near = STOPS.reduce(function (best, s) { return Math.abs(s.a - a) < Math.abs(best.a - a) ? s : best; }, STOPS[0]);
      val.innerHTML = "&alpha; = " + a.toFixed(2);
      dose.textContent = near.dose; tag.textContent = near.tag;
    }
    range.addEventListener("input", function () { place(+range.value); });
    place(0);
    // a gentle demo sweep the first time the figure is seen, unless the reader has already touched it
    if ("IntersectionObserver" in window && !reduceMotion.matches) {
      var touched = false;
      range.addEventListener("pointerdown", function () { touched = true; }, { once: true });
      var seen = new IntersectionObserver(function (entries) {
        if (!entries.some(function (en) { return en.isIntersecting; })) return;
        seen.disconnect();
        var t0 = null;
        (function sweep(ts) {
          if (touched) return;
          if (t0 === null) t0 = ts;
          var p = Math.min((ts - t0) / 3200, 1), e = 0.5 - 0.5 * Math.cos(Math.PI * p);
          range.value = e.toFixed(2); place(e);
          if (p < 1) requestAnimationFrame(sweep);
        })(performance.now());
      }, { threshold: 0.5 });
      seen.observe(box);
    }
  })();

  /* ---------- Click-to-unfold bullet list ---------- */
  $$(".bl-head").forEach(function (head) {
    head.addEventListener("click", function () {
      head.setAttribute("aria-expanded", String(head.parentNode.classList.toggle("open")));
    });
  });

  /* ---------- Action-prediction chart (balanced accuracy, Tables 2-4) ---------- */
  (function () {
    var chart = $("#action-chart");
    if (!chart) return;
    var body = $(".chart-body", chart);
    var LO = 45, HI = 85, TICKS = [50, 60, 70, 80];
    // ours = better of OpenSLA-B / OpenSLA-H; base = best of the eight compared baselines
    var DATA = [
      { group: "Clinical", name: "MC-MED",
        necessity: { ours: 72.5, v: "B", base: 64.6, who: "SensorLM" },
        category:  { ours: 68.7, v: "H", base: 63.1, who: "GPT-5.6-Luna" } },
      { name: "MIMIC-III",
        necessity: { ours: 75.4, v: "B", base: 71.2, who: "OpenTSLM" },
        category:  { ours: 74.5, v: "H", base: 66.0, who: "OpenTSLM" } },
      { group: "Operating room", name: "MOVER",
        necessity: { ours: 80.6, v: "H", base: 71.5, who: "OpenTSLM" },
        category:  { ours: 69.9, v: "H", base: 60.5, who: "Chronos-2" } },
      { name: "VitalDB",
        necessity: { ours: 66.3, v: "H", base: 52.1, who: "Chronos-2" },
        category:  { ours: 80.8, v: "H", base: 66.0, who: "ConvTrans" } },
      { group: "CGM", name: "MetaboNet",
        necessity: { ours: 65.0, v: "B", base: 62.9, who: "SensorLM" },
        category:  { ours: 59.6, v: "H", base: 58.7, who: "MMIM" } },
      { name: "PEDAP",
        necessity: { ours: 75.5, v: "B", base: 72.8, who: "ConvTrans" },
        category:  { ours: 56.3, v: "B", base: 66.2, who: "ConvTrans" } }
    ];
    var pos = function (v) { return ((v - LO) / (HI - LO) * 100) + "%"; };
    var el = function (cls, tag) { var n = document.createElement(tag || "div"); n.className = cls; return n; };

    var axis = el("c-axis");
    axis.appendChild(el("c-name"));
    var axisTrack = el("c-track");
    TICKS.forEach(function (t) {
      var s = document.createElement("span");
      s.style.left = pos(t); s.textContent = String(t);
      axisTrack.appendChild(s);
    });
    axis.appendChild(axisTrack); axis.appendChild(el("c-val"));
    axis.setAttribute("aria-hidden", "true");
    body.appendChild(axis);

    var rows = DATA.map(function (d) {
      if (d.group) { var g = el("c-group"); g.textContent = d.group; body.appendChild(g); }
      var row = el("c-row"); row.tabIndex = 0;
      var name = el("c-name"); name.textContent = d.name;
      var track = el("c-track");
      TICKS.forEach(function (t) { var gl = el("c-grid" + (t === 50 ? " chance" : "")); gl.style.left = pos(t); track.appendChild(gl); });
      var link = el("c-link"), base = el("c-dot base"), ours = el("c-dot ours");
      track.appendChild(link); track.appendChild(base); track.appendChild(ours);
      var val = el("c-val"), tip = el("c-tip");
      row.appendChild(name); row.appendChild(track); row.appendChild(val); row.appendChild(tip);
      body.appendChild(row);
      return { d: d, row: row, link: link, base: base, ours: ours, val: val, tip: tip };
    });

    function render(level) {
      rows.forEach(function (r) {
        var m = r.d[level], diff = m.ours - m.base;
        r.ours.style.left = pos(m.ours);
        r.base.style.left = pos(m.base);
        r.link.style.left = pos(Math.min(m.ours, m.base));
        r.link.style.width = (Math.abs(diff) / (HI - LO) * 100) + "%";
        r.val.innerHTML = "<b>" + m.ours.toFixed(1) + "</b>vs " + m.base.toFixed(1) + " <i>" + m.who + "</i>";
        var delta = (diff >= 0 ? "+" : "−") + Math.abs(diff).toFixed(1) + " pts";
        r.tip.textContent = "OpenSLA-" + m.v + " " + m.ours.toFixed(1) + "  ·  " + m.who + " " + m.base.toFixed(1) + "  ·  " + delta;
        r.row.setAttribute("aria-label", r.d.name + ": OpenSLA-" + m.v + " " + m.ours.toFixed(1) + ", " + m.who + " " + m.base.toFixed(1));
      });
    }
    $$(".seg button", chart).forEach(function (btn) {
      btn.addEventListener("click", function () {
        $$(".seg button", chart).forEach(function (b) {
          b.classList.toggle("is-on", b === btn);
          b.setAttribute("aria-pressed", String(b === btn));
        });
        render(btn.getAttribute("data-level"));
      });
    });
    // dots start on the chance line and slide out when the chart is first seen
    rows.forEach(function (r) { r.ours.style.left = r.base.style.left = r.link.style.left = pos(50); r.link.style.width = "0%"; });
    if ("IntersectionObserver" in window && !reduceMotion.matches) {
      var seen = new IntersectionObserver(function (entries) {
        if (entries.some(function (en) { return en.isIntersecting; })) { seen.disconnect(); setTimeout(function () { render("necessity"); }, 150); }
      }, { threshold: 0.3 });
      seen.observe(chart);
    } else { render("necessity"); }
  })();

  /* ---------- Figure zoom ---------- */
  (function () {
    var dialog = $("#zoom"), big = $("#zoom-img"), title = $("#zoom-title");
    if (!dialog || !dialog.showModal) { $$(".zoom-btn").forEach(function (b) { b.hidden = true; }); return; }
    $$(".zoom-btn").forEach(function (btn) {
      var fig = btn.closest("figure"), img = fig ? $("img", fig) : null, cap = fig ? $("figcaption", fig) : null;
      var video = fig ? $("video", fig) : null;
      btn.addEventListener("click", function () {
        if (video) {
          if (video.requestFullscreen) video.requestFullscreen();
          else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
          return;
        }
        big.src = btn.getAttribute("data-src") || (img && (img.currentSrc || img.src)) || "";
        big.alt = img ? img.alt : "";
        title.textContent = btn.getAttribute("data-title") || (cap ? cap.textContent : "");
        dialog.showModal();
      });
    });
    $(".zoom-close", dialog).addEventListener("click", function () { dialog.close(); });
    dialog.addEventListener("click", function (e) { if (e.target === dialog) dialog.close(); });
  })();

  /* ---------- Copy BibTeX ---------- */
  (function () {
    var btn = $("#copy-bib"), code = $("#bibtex"), status = $(".copy-status");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var text = code.textContent;
      var done = function (msg) { status.textContent = msg; setTimeout(function () { status.textContent = ""; }, 2400); };
      var fallback = function () {
        var range = document.createRange();
        range.selectNodeContents(code);
        var sel = window.getSelection();
        sel.removeAllRanges(); sel.addRange(range);
        try { done(document.execCommand("copy") ? "Copied" : "Selected — press Ctrl+C"); }
        catch (err) { done("Selected — press Ctrl+C"); }
      };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(function () { done("Copied"); }, fallback);
      else fallback();
    });
  })();

  icons();
})();
