(function(){
'use strict';
var pre = document.getElementById('preloader');
if(!window.gsap || !window.ScrollTrigger){
  if(pre) pre.style.display='none';
  return;
}
gsap.registerPlugin(ScrollTrigger, Flip);
gsap.config({nullTargetWarn:false});
var reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- helpers ---------- */
function visLoop(el, fn){
  if(!el) return;
  var running=false, raf=0;
  function loop(t){ fn(t); if(running) raf=requestAnimationFrame(loop); }
  new IntersectionObserver(function(ents){
    ents.forEach(function(e){
      if(e.isIntersecting && !running){ running=true; raf=requestAnimationFrame(loop); }
      else if(!e.isIntersecting && running){ running=false; cancelAnimationFrame(raf); }
    });
  },{rootMargin:'180px'}).observe(el);
}
function clamp(v,a,b){ return Math.max(a,Math.min(b,v)); }

/* ================= 17 · PRELOADER (0 -> 100) ================= */
var heroLines = gsap.utils.toArray('#heroTitle .hline span');
gsap.set(heroLines,{yPercent:0});
gsap.set('#heroKick,.hero-sub,.hero-kpis,.hero-actions',{opacity:1,y:0});

function heroIntro(){
  gsap.fromTo(heroLines,{yPercent:100},{yPercent:0,duration:.75,ease:'power4.out',stagger:.08,overwrite:true});
  gsap.fromTo('#heroKick,.hero-sub,.hero-kpis,.hero-actions',{opacity:0,y:16},{opacity:1,y:0,duration:.6,delay:.15,stagger:.07,ease:'power3.out',overwrite:true});
}

function runPreloader(done){
  if(!pre){ if(done) done(); return; }
  var p1=pre.querySelector('.p1'), p2=pre.querySelector('.p2'),
      cntEl=document.getElementById('plCount'),
      metaEl=pre.querySelector('.pl-meta'),
      cntWrap=pre.querySelector('.pl-count-wrap'),
      n={v:0};
  pre.style.display='flex';
  gsap.set([p1,p2],{yPercent:0});
  gsap.set(p1,{scaleY:1});
  gsap.set(p2,{scaleY:0,transformOrigin:'bottom'});
  gsap.set([cntWrap,metaEl],{opacity:1});
  cntEl.textContent='0';
  gsap.timeline({onComplete:function(){
    pre.style.display='none';
    ScrollTrigger.refresh();
    if(done) done();
  }})
    .to(n,{v:100,duration:.55,ease:'power2.inOut',onUpdate:function(){ cntEl.textContent=Math.round(n.v); }})
    .to([cntWrap,metaEl],{opacity:0,duration:.16},'+=.03')
    .to(p2,{scaleY:1,duration:.24,ease:'power3.inOut'},'<')
    .to(p2,{yPercent:-100,duration:.38,ease:'power4.inOut'},'+=.02')
    .to(p1,{yPercent:-100,duration:.38,ease:'power4.inOut'},'-=.28');
}

if(reduced || !document.body.hasAttribute('data-preload')){
  if(pre) pre.style.display='none';
} else {
  runPreloader(heroIntro);
}

var plrb=document.getElementById('plReplay');
if(plrb) plrb.addEventListener('click',function(){ runPreloader(heroIntro); });

/* ================= 12 · MORPHING CURSOR ================= */
var cursor=document.getElementById('cursor');
if(cursor && matchMedia('(pointer:fine)').matches){
  var cx=-100, cy=-100, tx=-100, ty=-100, moved=false;
  cursor.style.opacity='0';
  addEventListener('pointermove',function(e){
    tx=e.clientX; ty=e.clientY;
    if(!moved){ cx=tx; cy=ty; moved=true; cursor.style.opacity='1'; }
  });
  gsap.ticker.add(function(){
    if(!moved) return;
    cx+=(tx-cx)*.22; cy+=(ty-cy)*.22;
    cursor.style.transform='translate('+cx+'px,'+cy+'px) translate(-50%,-50%)';
  });
  document.addEventListener('mouseover',function(e){
    var t=e.target;
    cursor.classList.remove('is-link','is-view','is-drag');
    if(t.closest && t.closest('[data-cursor="drag"]')){
      cursor.classList.add('is-drag'); cursor.textContent='EXPLORAR';
    } else if(t.closest && t.closest('[data-cursor="view"]')){
      cursor.classList.add('is-view'); cursor.textContent='VER';
    } else if(t.closest && t.closest('a,button')){
      cursor.classList.add('is-link'); cursor.textContent='';
    } else {
      cursor.textContent='';
    }
  });
  document.addEventListener('mouseleave',function(){ cursor.classList.add('is-hidden'); });
  document.addEventListener('mouseenter',function(){ cursor.classList.remove('is-hidden'); });
} else if(cursor){
  cursor.style.display='none';
}

/* ================= 40/41 · NAV SHRINK + SCROLLSPY PILL ================= */
var nav=document.getElementById('nav'), pill=document.getElementById('navPill');
addEventListener('scroll',function(){
  if(nav) nav.classList.toggle('scrolled', scrollY>12);
},{passive:true});

var spyLinks=gsap.utils.toArray('#navLinks a');
function movePill(link){
  if(!pill) return;
  spyLinks.forEach(function(l){ l.classList.remove('active'); });
  if(!link){ pill.classList.remove('on'); return; }
  link.classList.add('active');
  pill.style.left=link.offsetLeft+'px';
  pill.style.width=link.offsetWidth+'px';
  pill.classList.add('on');
}

spyLinks.filter(function(l){ return l.dataset.spy; }).forEach(function(link){
  var id=link.dataset.spy;
  var el=document.getElementById(id);
  if(!el) return;
  ScrollTrigger.create({
    trigger:el, start:'top 55%', end:'bottom 55%',
    refreshPriority:-1,
    onToggle:function(self){ if(self.isActive) movePill(link); }
  });
  link.addEventListener('click',function(e){
    e.preventDefault();
    var targetEl=document.getElementById(id);
    if(targetEl) targetEl.scrollIntoView({behavior:'smooth'});
  });
});

/* ================= FLOATING SCORE HUD (61.2% -> 73.1% -> 80.0% -> 86.9% -> 88.8%) ================= */
var scoreState = { val: 61.2 };
var shValEl = document.getElementById('shVal');
var shStageEl = document.getElementById('shStage');
var shDeltaEl = document.getElementById('shDelta');
var shBarEl = document.getElementById('shBar');
var mtChapEl = document.getElementById('mtChap');
var mtDescEl = document.getElementById('mtDesc');

function setScoreMilestone(targetScore, stageLabel, deltaLabel, barPct, toastChap, toastDesc){
  gsap.to(scoreState, {
    val: targetScore,
    duration: 0.7,
    ease: 'power3.out',
    onUpdate: function(){
      if(shValEl) shValEl.textContent = scoreState.val.toFixed(1) + '%';
    }
  });
  if(shStageEl) shStageEl.textContent = stageLabel;
  if(shDeltaEl) shDeltaEl.textContent = deltaLabel;
  if(shBarEl) shBarEl.style.width = barPct + '%';
  if(mtChapEl && toastChap) mtChapEl.textContent = toastChap;
  if(mtDescEl && toastDesc) mtDescEl.textContent = toastDesc;
}

var milestones = [
  {
    id: 'sec-intro',
    score: 61.2,
    stage: 'Punto de Partida',
    delta: 'Base · 61.2%',
    bar: 18,
    chap: 'Intro · De 61.2% a 88.8%',
    desc: '199,319 fotogramas a 25 veces por segundo en el Barça vs Athletic.'
  },
  {
    id: 'sec-cap1',
    score: 61.2,
    stage: 'Cap 01 · El Juez',
    delta: 'Base · 61.2%',
    bar: 24,
    chap: 'Cap 01 · La Trampa del Juez',
    desc: 'Dudar medio segundo rompe una jugada de 30s y cuesta 3 fallos.'
  },
  {
    id: 'sec-cap2',
    score: 73.1,
    stage: 'Cap 02 · Física TV',
    delta: '+11.9 pts ↑',
    bar: 48,
    chap: 'Cap 02 · Pelotas Fantasma Reparadas',
    desc: 'Detector de leyes físicas + entrenamiento con gafas borrosas (73.1%).'
  },
  {
    id: 'sec-cap3',
    score: 80.0,
    stage: 'Cap 03 · 6 Expertos',
    delta: '+18.8 pts ↑',
    bar: 68,
    chap: 'Cap 03 · Comité de 6 Especialistas',
    desc: 'Foto fija + Película 8s + Bandada 22 + Simetría Espejo (80.0%).'
  },
  {
    id: 'sec-cap4',
    score: 80.0,
    stage: 'Cap 04 · Lecciones',
    delta: '+18.8 pts · Blindado',
    bar: 72,
    chap: 'Cap 04 · Ideas Descartadas vs Regla de Oro',
    desc: 'Solo actuar cuando la física del balón y los 6 expertos coinciden.'
  },
  {
    id: 'sec-cap5',
    score: 86.9,
    stage: 'Cap 05 · 4 Llaves',
    delta: '+25.7 pts ↑',
    bar: 90,
    chap: 'Cap 05 · Autopsia Forense (+5.7 pts en Banda)',
    desc: 'Esperar al latigazo exacto del saque de banda y filtrar rebotes.'
  },
  {
    id: 'sec-cap6',
    score: 88.8,
    stage: 'Campeón Final 🏆',
    delta: '+27.6 pts · 88.8%',
    bar: 100,
    chap: 'Cap 06 · Victoria en los 90 Minutos (88.8%)',
    desc: '98.6% Juego Vivo vs Parado · 61.5% Barça · 2.27 posesiones/min.'
  }
];

milestones.forEach(function(m){
  var el = document.getElementById(m.id);
  if(!el) return;
  ScrollTrigger.create({
    trigger: el,
    start: 'top 60%',
    end: 'bottom 60%',
    refreshPriority: -1,
    onToggle: function(self){
      if(self.isActive){
        setScoreMilestone(m.score, m.stage, m.delta, m.bar, m.chap, m.desc);
      }
    }
  });
});

/* ================= 39 · FULLSCREEN OVERLAY MENU ================= */
var omenu=document.getElementById('omenu'),
    burger=document.getElementById('burger'),
    omLinks=gsap.utils.toArray('#omenuLinks a'),
    ocKicker=document.getElementById('ocKicker'),
    ocTitle=document.getElementById('ocTitle'),
    ocDesc=document.getElementById('ocDesc'),
    ocScore=document.getElementById('ocScore'),
    menuOpen=false, menuTl=null;

omLinks.forEach(function(a){
  a.addEventListener('mouseenter',function(){
    if(ocKicker) ocKicker.textContent = a.dataset.kicker || '';
    if(ocTitle) ocTitle.textContent = a.dataset.title || '';
    if(ocDesc) ocDesc.textContent = a.dataset.desc || '';
    if(ocScore) ocScore.textContent = a.dataset.score || '';
  });
  a.addEventListener('click',function(e){
    e.preventDefault();
    var targetId = a.getAttribute('href').replace('#','');
    toggleMenu(false);
    var targetEl = document.getElementById(targetId);
    if(targetEl) setTimeout(function(){ targetEl.scrollIntoView({behavior:'smooth'}); }, 300);
  });
});

function toggleMenu(open){
  if(!omenu || !burger || open===menuOpen) return;
  menuOpen=open;
  burger.classList.toggle('open',open);
  if(menuTl) menuTl.kill();
  if(open){
    omenu.style.visibility='visible';
    menuTl=gsap.timeline()
      .fromTo(omenu,{clipPath:'inset(0 0 100% 0)'},{clipPath:'inset(0 0 0% 0)',duration:.65,ease:'power4.inOut'})
      .fromTo(omLinks.map(function(a){return a.querySelector('.oml');}),
        {yPercent:120},{yPercent:0,duration:.65,ease:'power3.out',stagger:.05},'-=.25');
  } else {
    menuTl=gsap.timeline({onComplete:function(){ omenu.style.visibility='hidden'; }})
      .to(omenu,{clipPath:'inset(0 0 100% 0)',duration:.5,ease:'power4.inOut'});
  }
}
if(burger) burger.addEventListener('click',function(){ toggleMenu(!menuOpen); });

/* ================= PICKS TRAY (SHORTLIST DE EFECTOS) ================= */
(function(){
  var KEY='bcnFxLabPicks',
      picks=new Set(JSON.parse(localStorage.getItem(KEY)||'[]')),
      tray=document.getElementById('tray'),
      list=document.getElementById('trayList'),
      count=document.getElementById('trayCount'),
      copyBtn=document.getElementById('trayCopy'),
      trayBtn=document.getElementById('trayBtn');
  if(!tray || !trayBtn) return;

  function save(){ localStorage.setItem(KEY,JSON.stringify(Array.from(picks))); }
  function render(){
    count.textContent=picks.size;
    list.innerHTML='';
    if(!picks.size){
      list.innerHTML='<li style="background:none;color:var(--mut);padding:12px 0">Pulsa ♥ en cualquier efecto o capítulo para guardarlo en tus favoritos.</li>';
      return;
    }
    Array.from(picks).sort().forEach(function(name){
      var li=document.createElement('li');
      li.textContent=name;
      var rm=document.createElement('button');
      rm.textContent='✕'; rm.setAttribute('aria-label','remove');
      rm.addEventListener('click',function(){
        picks.delete(name); save(); render(); syncBtns();
      });
      li.appendChild(rm); list.appendChild(li);
    });
  }
  function syncBtns(){
    document.querySelectorAll('.fx').forEach(function(fx){
      var b=fx.querySelector('.pick');
      if(b) b.classList.toggle('on',picks.has(fx.dataset.name));
    });
  }
  document.querySelectorAll('.fx').forEach(function(fx){
    var b=fx.querySelector('.pick'), name=fx.dataset.name;
    if(!b||!name) return;
    b.addEventListener('click',function(){
      picks.has(name) ? picks.delete(name) : picks.add(name);
      save(); render(); syncBtns();
      gsap.fromTo(b,{scale:1.35},{scale:1,duration:.5,ease:'elastic.out(1,.4)'});
    });
  });
  trayBtn.addEventListener('click',function(){ tray.classList.toggle('open'); });
  copyBtn.addEventListener('click',function(){
    var text='BCN GRADIENT DEPRESSION (61.2% -> 88.8%) — Efectos Favoritos:\n'+Array.from(picks).sort().map(function(n){ return '• '+n; }).join('\n');
    if(navigator.clipboard) navigator.clipboard.writeText(text);
    copyBtn.textContent='¡Copiado al portapapeles! ✓';
    setTimeout(function(){ copyBtn.textContent='Copiar resumen de capítulos →'; },2000);
  });
  render(); syncBtns();
})();

var actLink=document.querySelector('#navLinks a.active');
if(actLink){ setTimeout(function(){ movePill(actLink); },100); }

window.FX={visLoop:visLoop,clamp:clamp,runPreloader:runPreloader,setScoreMilestone:setScoreMilestone};
})();
