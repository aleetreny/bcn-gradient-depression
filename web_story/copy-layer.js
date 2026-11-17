/* ============================================================================
   copy-layer.js — the coach's notes column on the tactics board
   ----------------------------------------------------------------------------
   The chapter text sits straight on the slate, left of the pitch. The headline
   is written word by word in chalk hand, then the prose settles in. When the
   eraser sweeps into the next chapter it wipes these notes with the same edge
   it wipes the drawing (board.js reports the eraser's x in sw:scroll).
   ========================================================================== */

function mountCopyLayer(container, cfg) {
  const SECTIONS = (cfg && cfg.sections) || [];
  const N = SECTIONS.length;
  if (!N) return;
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ERASE_TO = 0.06, WRITE_LEN = 0.42;
  const MOBILE_READ = 0.95, MOBILE_READ0 = 0.7;   // phones: notes step aside after this (keep in sync with board.js)
  const isSmall = () => window.innerWidth < 860;

  const layer = document.createElement('div');
  layer.className = 'nt-layer';

  const esc = x => String(x).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const words = t => String(t).trim().split(/\s+/).map(w => {
    const m = w.match(/^\*(.+)\*([:.,;!?]*)$/);
    return `<span class="nt-w${m ? ' nt-w--stress' : ''}">${esc(m ? m[1] + m[2] : w)}</span>`;
  }).join(' ');

  const blocks = SECTIONS.map((s, i) => {
    const art = document.createElement('article');
    art.className = 'nt';
    art.dataset.id = s.id;
    art.innerHTML =
      `<${i === 0 ? 'h1' : 'h2'} class="nt-title">${words(s.title)}</${i === 0 ? 'h1' : 'h2'}>` +
      `<div class="nt-prose">${(s.paragraphs || []).map(p => `<p>${p}</p>`).join('')}</div>` +
      (s.metrics && s.metrics.length
        ? `<ul class="nt-figs">${s.metrics.map(m => `<li><b>${esc(m.value)}</b><span>${esc(m.label)}</span></li>`).join('')}</ul>`
        : '') +
      (s.cta
        ? `<div class="nt-cta">` +
          (s.cta.primary ? `<a class="nt-btn nt-btn--primary" href="${esc(s.cta.primary.href)}">${esc(s.cta.primary.label)}</a>` : '') +
          (s.cta.secondary ? `<a class="nt-btn" href="${esc(s.cta.secondary.href)}">${esc(s.cta.secondary.label)}</a>` : '') +
          `</div>`
        : '');
    layer.appendChild(art);
    return {
      el: art,
      words: [...art.querySelectorAll('.nt-w')],
      rest: [...art.querySelectorAll('.nt-prose p, .nt-figs, .nt-cta')],
      shown: null,
    };
  });
  container.appendChild(layer);

  let ranges = null;
  let introW = reduce ? 1 : 0, introStart = 0;
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const outExpo = x => (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x));

  function setWrite(b, w) {
    const n = b.words.length;
    b.words.forEach((el, j) => {
      const t = clamp(w * 1.7 * n - j);
      el.style.clipPath = t >= 1 ? '' : `inset(-0.2em ${((1 - t) * 100).toFixed(1)}% -0.3em -0.1em)`;
    });
    const r = outExpo(clamp((w - 0.45) / 0.55));
    b.rest.forEach((el, j) => {
      const rj = clamp(r * 1.4 - j * 0.15);
      el.style.opacity = rj.toFixed(3);
      el.style.transform = rj >= 1 ? '' : `translateY(${((1 - rj) * 10).toFixed(1)}px)`;
    });
  }

  function update(d) {
    if (!ranges) return;
    const vh = d.vh;
    blocks.forEach((b, i) => {
      const visible = d.activeIndex === i;
      if (!visible) {
        if (b.shown !== false) { b.el.style.visibility = 'hidden'; b.el.style.pointerEvents = 'none'; b.shown = false; }
        return;
      }
      if (b.shown !== true) { b.el.style.visibility = 'visible'; b.el.style.pointerEvents = 'auto'; b.shown = true; }
      const w = reduce ? 1 : i === 0 ? introW : clamp((d.y - (ranges[i].start - ERASE_TO * vh)) / (WRITE_LEN * vh));
      setWrite(b, w);
      const er = d.eraser;
      if (isSmall()) {
        // Phones: the notes panel appears with its first words, and steps aside once read
        // so the board underneath can be drawn
        const readEnd = ranges[i].start + (i === 0 ? MOBILE_READ0 : MOBILE_READ) * vh;
        const away = reduce ? (d.y > readEnd ? 1 : 0) : clamp((d.y - readEnd) / (0.2 * vh));
        const appear = clamp(w * 6);
        b.el.style.clipPath = '';
        b.el.style.opacity = (appear * (1 - away)).toFixed(3);
        b.el.style.transform = away > 0 ? `translateY(${(away * 28).toFixed(1)}px)` : '';
        if (away >= 1) { b.el.style.visibility = 'hidden'; b.el.style.pointerEvents = 'none'; b.shown = false; }
        return;
      }
      b.el.style.transform = '';
      // wiped by the eraser heading into the next chapter
      // by the same felt passes as the canvas (board.js wipeAt): the notes stay only where
      // the felt has not been yet = right of the finished passes AND outside the current one
      if (er && er.seam === i + 1 && er.e > 0) {
        const r = b.el.getBoundingClientRect();
        const px = v => `${v.toFixed(1)}px`;
        const done = er.xDone - r.left, x2 = er.x2 - r.left, yc = er.yc - r.top;
        const notDone = `linear-gradient(90deg, transparent ${px(done - 10)}, #000 ${px(done + 16)})`;
        const rightOfPass = `linear-gradient(90deg, transparent ${px(x2 - 14)}, #000 ${px(x2 + 10)})`;
        const aheadOfFelt = er.down
          ? `linear-gradient(180deg, transparent ${px(yc - 12)}, #000 ${px(yc + 18)})`
          : `linear-gradient(180deg, #000 ${px(yc - 18)}, transparent ${px(yc + 12)})`;
        const m = `${notDone}, ${rightOfPass}, ${aheadOfFelt}`;
        b.el.style.webkitMaskImage = m;
        b.el.style.maskImage = m;
        b.el.style.webkitMaskComposite = 'source-in, source-over';
        b.el.style.maskComposite = 'intersect, add';
        b.el.style.opacity = '';
      } else {
        b.el.style.webkitMaskImage = '';
        b.el.style.maskImage = '';
        b.el.style.webkitMaskComposite = '';
        b.el.style.maskComposite = '';
        b.el.style.opacity = '';
      }
    });
  }

  let last = null;
  container.addEventListener('sw:layout', e => { ranges = e.detail.sections; });
  container.addEventListener('sw:scroll', e => { last = e.detail; update(last); });

  if (!reduce) {
    const tick = now => {
      if (!introStart) introStart = now;
      introW = clamp((now - introStart - 250) / 1500);
      if (last) update(last);
      if (introW < 1 && window.scrollY < 40) requestAnimationFrame(tick);
      else { introW = 1; if (last) update(last); }
    };
    requestAnimationFrame(tick);
  }
}

if (typeof window !== 'undefined') window.mountCopyLayer = mountCopyLayer;
