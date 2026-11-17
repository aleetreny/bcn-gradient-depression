(function(){
'use strict';
if(!window.gsap || !window.ScrollTrigger || !window.FX) return;
var visLoop = FX.visLoop, clamp = FX.clamp;

/* ================= 08 · INTRO: ZOOM-THROUGH SCROLL (ESTADIO -> 199,319 INSTANTES) ================= */
if(document.getElementById('zoomStage')){
  gsap.timeline({
    scrollTrigger:{
      trigger:'#zoomStage',
      start:'top top',
      end:'+=190%',
      pin:true,
      scrub:0.8
    }
  })
    .to('#zoomFrame',{scale:5.6,borderRadius:'0px',ease:'power2.inOut',duration:0.38},0)
    .to('.zoom-cap',{opacity:0,y:-36,duration:0.08,ease:'power2.out'},0)
    .to('#zoomPitchSvg circle',{opacity:0,duration:0.12,ease:'power2.out'},0.08)
    .to('#zoomFrame',{opacity:0,duration:0.16,ease:'power2.out'},0.18)
    .fromTo('#zoomNext',{opacity:0,scale:0.95},{opacity:1,scale:1,duration:0.20,ease:'power2.out'},0.18)
    .to('#zoomNext',{opacity:1,duration:0.46},0.38);

  // Live ticker inside #zoomNext (simulating 25 frames per second)
  var znFrameEl = document.getElementById('znFrameCounter');
  var znBallCoordEl = document.getElementById('znBallCoord');
  if(znFrameEl){
    var frameNum = 14250;
    visLoop(document.getElementById('zoomStage'), function(t){
      frameNum = 14250 + Math.floor((t * 0.025) % 4500);
      znFrameEl.textContent = frameNum.toLocaleString('es-ES') + ' / 199.319';
      if(znBallCoordEl){
        var bx = (Math.sin(t * 0.0018) * 28.4).toFixed(2);
        var by = (Math.cos(t * 0.0013) * 16.2).toFixed(2);
        znBallCoordEl.textContent = 'X: ' + (bx >= 0 ? '+' : '') + bx + 'm · Y: ' + (by >= 0 ? '+' : '') + by + 'm';
      }
    });
  }
}

/* ================= 04 · TEXT MASK REVEAL (FRASE MANIFIESTO LETRA A LETRA) ================= */
document.querySelectorAll('.tmask').forEach(function(el){
  var txt = el.textContent.trim();
  el.textContent = '';
  txt.split(' ').forEach(function(word, wi, arr){
    var wSpan = document.createElement('span');
    wSpan.style.display = 'inline-block';
    wSpan.style.whiteSpace = 'nowrap';
    word.split('').forEach(function(ch){
      var s = document.createElement('span');
      s.className = 'tm-ch';
      s.textContent = ch;
      var f = document.createElement('span');
      f.className = 'tm-fill';
      f.textContent = ch;
      s.appendChild(f);
      wSpan.appendChild(s);
    });
    el.appendChild(wSpan);
    if(wi < arr.length - 1){
      el.appendChild(document.createTextNode(' '));
    }
  });
  gsap.to(el.querySelectorAll('.tm-fill'),{
    clipPath:'inset(0 0% 0 0)',
    stagger:0.015,
    ease:'none',
    scrollTrigger:{
      trigger:el,
      start:'top 88%',
      end:'top 32%',
      scrub:0.4
    }
  });
});

/* ================= 05 · PARALLAX DEPTH LAYERS ================= */
gsap.utils.toArray('#plaxStage .pl-layer').forEach(function(l){
  var sp = parseFloat(l.dataset.speed || 0.25);
  gsap.fromTo(l,
    {y: function(){ return -sp * 70; }},
    {
      y: function(){ return sp * 70; },
      ease:'none',
      scrollTrigger:{
        trigger:'#plaxStage',
        start:'top bottom',
        end:'bottom top',
        scrub:true
      }
    }
  );
});

/* ================= 02 · CAPÍTULO 1: CANVAS SEQUENCE SCRUB (23 PUNTOS + LA GRIETA + EL PUENTE CIEGO) ================= */
(function(){
  var cv = document.getElementById('seqCanvas');
  if(!cv) return;
  var ctx = cv.getContext('2d');
  var TOTAL = 150;
  var st = {f: 0};
  var seqTitle = document.getElementById('seqTitle');
  var seqSub   = document.getElementById('seqSub');
  var tabs     = document.querySelectorAll('.seq-tab');

  // Base positions for 11 Barça and 11 Athletic players (normalized -1..1 pitch coords)
  var bcnBase = [
    [-0.86, 0.00, 'Portero'],
    [-0.58, -0.62, 'Lateral Izq'], [-0.62, -0.22, 'Central Izq'], [-0.60, 0.24, 'Central Der'], [-0.54, 0.65, 'Lateral Der'],
    [-0.22, -0.38, 'Interior Izq'], [-0.32, 0.06, 'Mediocentro'], [-0.18, 0.40, 'Interior Der'],
    [0.20, -0.66, 'Extremo Izq'], [0.36, -0.04, 'Delantero'], [0.24, 0.62, 'Extremo Der']
  ];
  var athBase = [
    [0.88, 0.00, 'Portero Rival'],
    [0.56, -0.58, 'Defensa'], [0.52, -0.20, 'Central'], [0.54, 0.22, 'Central'], [0.58, 0.60, 'Defensa'],
    [0.18, -0.42, 'Medio'], [0.12, -0.05, 'Pivote'], [0.16, 0.36, 'Medio'],
    [0.02, -0.58, 'Banda'], [-0.24, 0.04, 'Punta'], [-0.12, 0.48, 'Banda']
  ];

  function roundRect(ctx, x, y, w, h, r){
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function updatePhaseUI(phaseIdx, f){
    tabs.forEach(function(t, idx){
      t.classList.toggle('active', idx === phaseIdx);
    });
    if(phaseIdx === 0){
      if(seqTitle) seqTitle.textContent = '1. Lo Único que Ve la Máquina: 23 Puntitos de Luz en la Oscuridad';
      if(seqSub)   seqSub.textContent   = 'FOTOGRAMA ' + String(f + 1).padStart(3,'0') + ' / 150 · 25 VECES POR SEGUNDO · CERO VÍDEO';
    } else if(phaseIdx === 1){
      if(seqTitle) seqTitle.textContent = '2. La Trampa de "La Grieta": Dudar 0,5 Segundos Cuesta 3 Fallos';
      if(seqSub)   seqSub.textContent   = 'FOTOGRAMA ' + String(f + 1).padStart(3,'0') + ' / 150 · EL JUEZ EXIGE JUGADAS ENTERAS';
    } else {
      if(seqTitle) seqTitle.textContent = '3. La Trampa de "El Puente Ciego": Ignorar un Robo de 1s Destruye 3 Jugadas';
      if(seqSub)   seqSub.textContent   = 'FOTOGRAMA ' + String(f + 1).padStart(3,'0') + ' / 150 · SUAVIZAR A CIEGAS ES MORTAL';
    }
  }

  function drawPitchBackground(w, h, pitchRect, alpha){
    ctx.save();
    ctx.globalAlpha = alpha;
    var px = pitchRect.x, py = pitchRect.y, pw = pitchRect.w, ph = pitchRect.h;
    roundRect(ctx, px, py, pw, ph, 14);
    ctx.fillStyle = '#0c1220';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Subtle tactical grid
    ctx.strokeStyle = 'rgba(255,255,255,0.035)';
    ctx.lineWidth = 1;
    for(var gx = 1; gx < 10; gx++){
      var x = px + (pw / 10) * gx;
      ctx.beginPath(); ctx.moveTo(x, py); ctx.lineTo(x, py + ph); ctx.stroke();
    }
    for(var gy = 1; gy < 6; gy++){
      var y = py + (ph / 6) * gy;
      ctx.beginPath(); ctx.moveTo(px, y); ctx.lineTo(px + pw, y); ctx.stroke();
    }

    // Center line & circle
    ctx.strokeStyle = 'rgba(255,255,255,0.14)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px + pw / 2, py);
    ctx.lineTo(px + pw / 2, py + ph);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(px + pw / 2, py + ph / 2, ph * 0.18, 0, Math.PI * 2);
    ctx.stroke();

    // Penalty boxes
    var boxW = pw * 0.15, boxH = ph * 0.54;
    ctx.strokeRect(px, py + (ph - boxH) / 2, boxW, boxH);
    ctx.strokeRect(px + pw - boxW, py + (ph - boxH) / 2, boxW, boxH);
    ctx.restore();
  }

  function drawSeq(f){
    var w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);

    // Deep dark lab background
    var bgGrad = ctx.createRadialGradient(w * 0.5, h * 0.42, 40, w * 0.5, h * 0.5, w * 0.7);
    bgGrad.addColorStop(0, '#13192b');
    bgGrad.addColorStop(1, '#070a12');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    var phaseIdx = f < 50 ? 0 : (f < 100 ? 1 : 2);
    updatePhaseUI(phaseIdx, f);

    // Top telemetry header inside canvas
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    roundRect(ctx, 36, 18, w - 72, 40, 10);
    ctx.fill();
    ctx.font = '600 12.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#f5b841';
    ctx.textAlign = 'left';
    ctx.fillText('SIMULADOR DEL JUEZ OFICIAL · FOTOGRAMA ' + String(f + 1).padStart(3, '0') + ' / 150', 54, 43);
    ctx.textAlign = 'right';
    ctx.fillStyle = 'rgba(244,246,251,0.65)';
    ctx.fillText(
      phaseIdx === 0 ? 'FASE 1: 11 BARÇA + 11 ATHLETIC + 1 BALÓN' :
      phaseIdx === 1 ? 'FASE 2: LA GRIETA (1 DUDA = 3 FALLOS)' :
                       'FASE 3: EL PUENTE CIEGO (FUSIÓN ILEGAL)',
      w - 54, 43
    );

    if(phaseIdx === 0){
      // PHASE 0: 23 DOTS IN THE DARK + PASS IN FLIGHT DOUBT
      var p = f / 49; // 0..1
      var pitchRect = {x: 56, y: 72, w: w - 112, h: h - 220};
      drawPitchBackground(w, h, pitchRect, 1);

      var cx = pitchRect.x + pitchRect.w / 2;
      var cy = pitchRect.y + pitchRect.h / 2;
      var sx = pitchRect.w * 0.46;
      var sy = pitchRect.h * 0.43;

      var bcnPts = bcnBase.map(function(pt, i){
        var dx = Math.sin(p * Math.PI * 2 + i * 0.7) * 0.04 + p * 0.07;
        var dy = Math.cos(p * Math.PI * 2 + i * 1.1) * 0.035;
        return {
          x: cx + (pt[0] + dx) * sx,
          y: cy + (pt[1] + dy) * sy,
          mx: ((pt[0] + dx) * 52.5).toFixed(1),
          my: ((pt[1] + dy) * 34.0).toFixed(1),
          role: pt[2]
        };
      });
      var athPts = athBase.map(function(pt, i){
        var dx = Math.cos(p * Math.PI * 2 + i * 0.9) * 0.035 - p * 0.04;
        var dy = Math.sin(p * Math.PI * 2 + i * 0.8) * 0.04;
        return {
          x: cx + (pt[0] + dx) * sx,
          y: cy + (pt[1] + dy) * sy,
          mx: ((pt[0] + dx) * 52.5).toFixed(1),
          my: ((pt[1] + dy) * 34.0).toFixed(1),
          role: pt[2]
        };
      });

      var passNodes = [bcnPts[6], bcnPts[5], bcnPts[8], bcnPts[9]];
      var segFloat = p * (passNodes.length - 1);
      var segIdx = Math.min(passNodes.length - 2, Math.floor(segFloat));
      var segT = segFloat - segIdx;
      var fromP = passNodes[segIdx], toP = passNodes[segIdx + 1];
      var ballX = fromP.x + (toP.x - fromP.x) * segT;
      var ballY = fromP.y + (toP.y - fromP.y) * segT;
      var inFlight = (segT > 0.24 && segT < 0.76);

      // Pass trajectory line
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = 'rgba(245,184,65,0.5)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(fromP.x, fromP.y);
      ctx.lineTo(toP.x, toP.y);
      ctx.stroke();
      ctx.restore();

      // Athletic players (Cyan)
      athPts.forEach(function(pt){
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 7.5, 0, Math.PI * 2);
        ctx.fillStyle = '#22b8ff';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(255,255,255,0.75)';
        ctx.stroke();
      });

      // Barça players (Blaugrana Crimson)
      bcnPts.forEach(function(pt){
        var isFrom = (pt === fromP);
        var isTo   = (pt === toP);
        if(isFrom || isTo){
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 17, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(230,25,82,0.25)';
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 8.5, 0, Math.PI * 2);
        ctx.fillStyle = '#e61952';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      });

      // Clean non-overlapping pill labels for Passer and Receiver on the outside of the pass vector
      var passDown = (toP.y >= fromP.y);
      var pBoxY = passDown ? (fromP.y - 34) : (fromP.y + 13);
      var rBoxY = passDown ? (toP.y + 13)   : (toP.y - 34);
      ctx.font = '600 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      roundRect(ctx, fromP.x - 62, pBoxY, 124, 22, 6);
      ctx.fillStyle = 'rgba(12,16,28,0.92)';
      ctx.fill();
      ctx.fillStyle = '#ff9bb5';
      ctx.fillText('PASADOR: ' + fromP.role, fromP.x, pBoxY + 15);

      roundRect(ctx, toP.x - 66, rBoxY, 132, 22, 6);
      ctx.fillStyle = 'rgba(12,16,28,0.92)';
      ctx.fill();
      ctx.fillStyle = '#f5b841';
      ctx.fillText('RECEPTOR: ' + toP.role, toP.x, rBoxY + 15);

      // Draw Ball
      ctx.beginPath();
      ctx.arc(ballX, ballY, 15, 0, Math.PI * 2);
      ctx.fillStyle = inFlight ? 'rgba(255,71,87,0.3)' : 'rgba(245,184,65,0.3)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(ballX, ballY, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = '#f5b841';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#fff';
      ctx.stroke();

      // Callout badge docked cleanly at top-center of pitch so it never overlaps players or labels
      var badgeX = pitchRect.x + pitchRect.w / 2;
      var badgeY = pitchRect.y + 12;
      if(inFlight){
        roundRect(ctx, badgeX - 142, badgeY, 284, 30, 8);
        ctx.fillStyle = 'rgba(255,71,87,0.94)';
        ctx.fill();
        ctx.font = '700 11.5px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.fillText('⚠️ PASE EN VUELO: Nadie a menos de 4m', badgeX, badgeY + 19);
      } else {
        roundRect(ctx, badgeX - 114, badgeY, 228, 28, 8);
        ctx.fillStyle = 'rgba(16,185,129,0.92)';
        ctx.fill();
        ctx.font = '700 11.5px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#06130e';
        ctx.textAlign = 'center';
        ctx.fillText('✓ Balón en el pie (<1.2m)', badgeX, badgeY + 18);
      }

      // Bottom comparison strip inside canvas
      var by = h - 134;
      roundRect(ctx, 56, by, w - 112, 114, 14);
      ctx.fillStyle = 'rgba(12,16,28,0.94)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.stroke();

      var barW = w - 156;
      ctx.textAlign = 'left';
      ctx.font = '700 11.5px "Space Grotesk", sans-serif';
      ctx.fillStyle = '#10b981';
      ctx.fillText('HUMANO EN LA TELE → 1 SOLA JUGADA CONTINUA DEL BARÇA (30 SEGUNDOS):', 78, by + 25);
      roundRect(ctx, 78, by + 32, barW, 18, 5);
      ctx.fillStyle = '#e61952';
      ctx.fill();

      ctx.fillStyle = '#ff6b78';
      ctx.fillText('ORDENADOR CIEGO → DUDA CADA VEZ QUE EL BALÓN VUELA (BARRA ROTA EN 5 TROZOS):', 78, by + 72);
      roundRect(ctx, 78, by + 79, barW * 0.24, 18, 5); ctx.fillStyle = '#e61952'; ctx.fill();
      roundRect(ctx, 78 + barW * 0.25, by + 79, barW * 0.08, 18, 5); ctx.fillStyle = '#22b8ff'; ctx.fill();
      roundRect(ctx, 78 + barW * 0.34, by + 79, barW * 0.26, 18, 5); ctx.fillStyle = '#e61952'; ctx.fill();
      roundRect(ctx, 78 + barW * 0.61, by + 79, barW * 0.07, 18, 5); ctx.fillStyle = '#64748b'; ctx.fill();
      roundRect(ctx, 78 + barW * 0.69, by + 79, barW * 0.31, 18, 5); ctx.fillStyle = '#e61952'; ctx.fill();

      var phX = 78 + barW * p;
      ctx.strokeStyle = '#f5b841';
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(phX, by + 28); ctx.lineTo(phX, by + 103); ctx.stroke();

    } else if(phaseIdx === 1){
      // PHASE 1: "LA GRIETA" (0.5s hesitation splits a 30s Barça play into 2 invalid halves = 3 errors)
      var p2 = (f - 50) / 49; // 0..1
      var crackAmt = clamp((p2 - 0.1) / 0.5, 0, 1);
      var bx = 80, bw = w - 160;

      ctx.textAlign = 'center';
      ctx.font = '700 25px "Space Grotesk", sans-serif';
      ctx.fillStyle = '#f4f6fb';
      ctx.fillText('LA TRAMPA DE "LA GRIETA": DUDAR 0,5 SEGUNDOS CUESTA 3 FALLOS', w / 2, 104);
      ctx.font = '400 14.5px "Inter", sans-serif';
      ctx.fillStyle = '#9aa3b8';
      ctx.fillText('El juez exige que cada barra predicha coincida de principio a fin con la jugada real:', w / 2, 130);

      // 1. REALITY BAR (30s Barça)
      ctx.textAlign = 'left';
      ctx.font = '700 12.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#10b981';
      ctx.fillText('① REALIDAD EN EL CÉSPED · 1 JUGADA CONTINUA DEL BARÇA (30,0 SEGUNDOS)', bx, 174);

      roundRect(ctx, bx, 186, bw, 52, 12);
      var realGrad = ctx.createLinearGradient(bx, 186, bx + bw, 186);
      realGrad.addColorStop(0, '#c81446');
      realGrad.addColorStop(1, '#e61952');
      ctx.fillStyle = realGrad;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.textAlign = 'center';
      ctx.font = '700 16px "Space Grotesk", sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.fillText('POSESIÓN REAL DEL FC BARCELONA · 00:00 → 00:30 (1 BARRA ENTERA DE 30s)', bx + bw / 2, 218);

      // 2. SPLIT BAR ("LA GRIETA")
      ctx.textAlign = 'left';
      ctx.font = '700 12.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#ff6b78';
      ctx.fillText('② TU PREDICCIÓN · DUDAS 0,5 SEGUNDOS EN EL SEGUNDO 15 ("LA GRIETA")', bx, 286);

      var gapW = 24 + crackAmt * 56;
      var halfW = (bw - gapW) / 2;

      roundRect(ctx, bx, 298, halfW, 52, 12);
      ctx.fillStyle = 'rgba(230,25,82,0.55)';
      ctx.fill();
      ctx.strokeStyle = '#ff4757';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.font = '700 15px "Space Grotesk", sans-serif';
      ctx.fillStyle = '#fff';
      ctx.fillText('TROZO A: 14,7s (Sólo 49% de la real)', bx + halfW / 2, 330);

      roundRect(ctx, bx + halfW + gapW, 298, halfW, 52, 12);
      ctx.fillStyle = 'rgba(230,25,82,0.55)';
      ctx.fill();
      ctx.strokeStyle = '#ff4757';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.fillText('TROZO B: 14,8s (Sólo 49% de la real)', bx + halfW + gapW + halfW / 2, 330);

      // Lightning crack in center
      var midX = bx + halfW + gapW / 2;
      roundRect(ctx, midX - 30, 290, 60, 68, 10);
      ctx.fillStyle = '#22b8ff';
      ctx.fill();
      ctx.font = '700 11.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#061018';
      ctx.fillText('⚡ 0,5s', midX, 320);
      ctx.fillText('DUDA', midX, 338);

      // 3. THE 3 VISUAL PENALTY CARDS
      var cardY = 398, cardW = (bw - 36) / 3, cardH = 220;
      var penalties = [
        {
          badge: 'FALLO #1 · JUGADA PERDIDA',
          stat: '-1 ACIERTO',
          title: 'La barra real de 30s se da por perdida',
          line1: 'Al partirse por la mitad, ningún trozo',
          line2: 'llega al tamaño mínimo que exige el juez.'
        },
        {
          badge: 'FALLO #2 · TROZO A INVÁLIDO',
          stat: '+1 ALARMA FALSA',
          title: 'El primer trozo de 14,7s penaliza',
          line1: 'Aunque acertaste que era del Barça,',
          line2: 'cuenta como una jugada suelta inventada.'
        },
        {
          badge: 'FALLO #3 · TROZO B INVÁLIDO',
          stat: '+1 ALARMA FALSA',
          title: 'El segundo trozo de 14,8s penaliza',
          line1: 'Acertaste el 98% de los fotogramas,',
          line2: '¡pero recibes 0 aciertos y 3 castigos!'
        }
      ];

      penalties.forEach(function(pen, idx){
        var cxCard = bx + idx * (cardW + 18);
        roundRect(ctx, cxCard, cardY, cardW, cardH, 16);
        ctx.fillStyle = 'rgba(255,71,87,0.11)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,71,87,0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.textAlign = 'left';
        ctx.font = '700 11px "JetBrains Mono", monospace';
        ctx.fillStyle = '#ff8a93';
        ctx.fillText('❌ ' + pen.badge, cxCard + 22, cardY + 34);

        ctx.font = '700 24px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#ff4757';
        ctx.fillText(pen.stat, cxCard + 22, cardY + 74);

        ctx.font = '700 16px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(pen.title, cxCard + 22, cardY + 114);

        ctx.font = '400 13.5px "Inter", sans-serif';
        ctx.fillStyle = '#cbd5e1';
        ctx.fillText(pen.line1, cxCard + 22, cardY + 150);
        ctx.fillText(pen.line2, cxCard + 22, cardY + 172);
      });

    } else {
      // PHASE 2: "EL PUENTE CIEGO" (Ignoring a 1s Athletic interception fuses 2 Barça plays into 1 monster bar)
      var bx3 = 80, bw3 = w - 160;

      ctx.textAlign = 'center';
      ctx.font = '700 25px "Space Grotesk", sans-serif';
      ctx.fillStyle = '#f4f6fb';
      ctx.fillText('LA TRAMPA DE "EL PUENTE CIEGO": BORRAR UN ROBO DE 1s DESTRUYE 3 JUGADAS', w / 2, 104);
      ctx.font = '400 14.5px "Inter", sans-serif';
      ctx.fillStyle = '#9aa3b8';
      ctx.fillText('Si suavizas a ciegas para evitar "La Grieta", te tragas los robos rápidos del rival:', w / 2, 130);

      // 1. REALITY: 3 DISTINCT PLAYS
      ctx.textAlign = 'left';
      ctx.font = '700 12.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#10b981';
      ctx.fillText('① REALIDAD EN EL CÉSPED · 3 JUGADAS SEPARADAS POR UN ROBO RÁPIDO DE 1 SEGUNDO', bx3, 174);

      var seg1W = bw3 * 0.42, segMidW = bw3 * 0.16, seg2W = bw3 * 0.42;
      roundRect(ctx, bx3, 186, seg1W - 8, 52, 12);
      ctx.fillStyle = '#e61952'; ctx.fill();
      ctx.textAlign = 'center'; ctx.font = '700 15px "Space Grotesk", sans-serif'; ctx.fillStyle = '#fff';
      ctx.fillText('JUGADA 1: Barça (15s)', bx3 + seg1W / 2, 218);

      roundRect(ctx, bx3 + seg1W, 186, segMidW - 8, 52, 12);
      ctx.fillStyle = '#22b8ff'; ctx.fill();
      ctx.fillStyle = '#04121d';
      ctx.fillText('Robo Athletic (1s)', bx3 + seg1W + (segMidW - 8) / 2, 218);

      roundRect(ctx, bx3 + seg1W + segMidW, 186, seg2W, 52, 12);
      ctx.fillStyle = '#e61952'; ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText('JUGADA 3: Barça (15s)', bx3 + seg1W + segMidW + seg2W / 2, 218);

      // 2. BLIND BRIDGE BAR (Fused 31s bar)
      ctx.textAlign = 'left';
      ctx.font = '700 12.5px "JetBrains Mono", monospace';
      ctx.fillStyle = '#f5b841';
      ctx.fillText('② PREDICCIÓN SUAVIZADA A CIEGAS · TE TRAGAS EL ROBO Y FUSIONAS LAS DOS BARRAS', bx3, 286);

      roundRect(ctx, bx3, 298, bw3, 52, 12);
      ctx.fillStyle = 'rgba(230,25,82,0.5)';
      ctx.fill();
      ctx.strokeStyle = '#f5b841';
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Wide 260px yellow badge in center so text never touches edges
      var centerBridgeX = bx3 + bw3 / 2;
      roundRect(ctx, centerBridgeX - 135, 290, 270, 68, 12);
      ctx.fillStyle = '#f5b841';
      ctx.fill();
      ctx.textAlign = 'center';
      ctx.font = '700 12px "JetBrains Mono", monospace';
      ctx.fillStyle = '#090b10';
      ctx.fillText('⚠️ PUENTE CIEGO ILEGAL', centerBridgeX, 320);
      ctx.fillText('(BORRA EL ROBO Y UNE LAS DOS)', centerBridgeX, 340);

      // 3. 3 SUMMARY COLUMNS AT BOTTOM
      var colY = 398, colW = (bw3 - 36) / 3, colH = 220;
      var cols = [
        {
          kicker: 'PELIGRO 1 · MODELO NERVIOSO',
          title: 'Cae en "La Grieta"',
          stat: '-3 Fallos',
          color: '#ff6b78',
          l1: 'Cada pase largo en el aire',
          l2: 'parte la jugada en dos mitades.'
        },
        {
          kicker: 'PELIGRO 2 · MODELO TRANQUILO',
          title: 'Cae en "Puente Ciego"',
          stat: '-3 Fallos',
          color: '#f5b841',
          l1: 'Borrar cambios cortos destruye',
          l2: 'los robos reales de 1 segundo.'
        },
        {
          kicker: 'NUESTRA SOLUCIÓN GANADORA',
          title: 'Física + 6 Expertos',
          stat: '88.8% 🏆',
          color: '#10b981',
          l1: 'Sólo cortamos cuando el balón',
          l2: 'frena de verdad a <1.5m del rival.'
        }
      ];

      cols.forEach(function(col, idx){
        var cxCol = bx3 + idx * (colW + 18);
        roundRect(ctx, cxCol, colY, colW, colH, 16);
        ctx.fillStyle = idx === 2 ? 'rgba(16,185,129,0.12)' : 'rgba(255,255,255,0.04)';
        ctx.fill();
        ctx.strokeStyle = idx === 2 ? 'rgba(16,185,129,0.55)' : 'rgba(255,255,255,0.14)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.textAlign = 'left';
        ctx.font = '700 11px "JetBrains Mono", monospace';
        ctx.fillStyle = col.color;
        ctx.fillText(col.kicker, cxCol + 22, colY + 34);

        ctx.font = '700 26px "Space Grotesk", sans-serif';
        ctx.fillStyle = col.color;
        ctx.fillText(col.stat, cxCol + 22, colY + 76);

        ctx.font = '700 17px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(col.title, cxCol + 22, colY + 116);

        ctx.font = '400 13.5px "Inter", sans-serif';
        ctx.fillStyle = '#cbd5e1';
        ctx.fillText(col.l1, cxCol + 22, colY + 152);
        ctx.fillText(col.l2, cxCol + 22, colY + 174);
      });
    }
  }

  drawSeq(0);

  ScrollTrigger.create({
    trigger:'#seqStage',
    start:'top top',
    end:'+=240%',
    pin:true,
    scrub:0.5,
    onUpdate:function(self){
      st.f = self.progress * (TOTAL - 1);
      drawSeq(Math.round(st.f));
    }
  });

  tabs.forEach(function(tab){
    tab.addEventListener('click', function(){
      var targetPhase = parseInt(tab.dataset.phase || '0', 10);
      var targetFrame = targetPhase === 0 ? 18 : (targetPhase === 1 ? 75 : 125);
      st.f = targetFrame;
      drawSeq(targetFrame);
    });
  });
})();

/* ================= 53 · CAPÍTULO 2: SPLIT-SCREEN STICKY STORY (CÁMARAS DE TV Y PELOTAS FANTASMA) ================= */
(function(){
  var stage = document.getElementById('splitStage');
  if(!stage) return;
  var visuals = stage.querySelectorAll('.sp-visual');
  var chaps   = stage.querySelectorAll('.split-chap');
  var idxEl   = document.getElementById('splitIdx');

  function activateSplit(i){
    visuals.forEach(function(v, j){ v.classList.toggle('on', j === i); });
    if(idxEl) idxEl.textContent = '0' + (i + 1) + ' / 0' + chaps.length + (i === 2 ? ' · +11.9 PTS (73.1%)' : ' · ANOMALÍA TV');
  }

  chaps.forEach(function(c, i){
    ScrollTrigger.create({
      trigger: c,
      start: 'top 58%',
      end: 'bottom 58%',
      onEnter: function(){ activateSplit(i); },
      onEnterBack: function(){ activateSplit(i); }
    });
  });

  // Interactive toggle inside Visual 3 (Compare Raw Ghost Ball vs Repaired Trajectory)
  var repairBtn = document.getElementById('toggleRepairBtn');
  var ghostPath = document.getElementById('svgGhostSpike');
  var cleanPath = document.getElementById('svgCleanCurve');
  var repairStatus = document.getElementById('repairStatusBadge');
  var repaired = true;
  if(repairBtn){
    repairBtn.addEventListener('click', function(){
      repaired = !repaired;
      if(ghostPath) ghostPath.style.opacity = repaired ? '0.22' : '1';
      if(cleanPath) cleanPath.style.opacity = repaired ? '1' : '0.15';
      if(repairStatus){
        repairStatus.textContent = repaired
          ? '✓ DETECTOR FÍSICO ACTIVO · Teletransporte eliminado (73.1%)'
          : '❌ CÁMARA CRUDA SIN FILTRO · Pelota fantasma en la grada (61.2%)';
        repairStatus.style.color = repaired ? '#10b981' : '#ff6b78';
      }
      repairBtn.textContent = repaired ? 'Ver fallo original →' : 'Activar Reparador Físico →';
    });
  }
})();

/* ================= 03 · CAPÍTULO 3: HORIZONTAL SCROLL SECTION (5 PANELES + SIMETRÍA ESPEJO) ================= */
if(document.getElementById('hWrap')){
  gsap.to('#hTrack',{
    xPercent: -80, // 5 panels = 500% width; moving -80% of 500% lands cleanly on Panel 5
    ease: 'none',
    scrollTrigger:{
      trigger: '#hWrap',
      start: 'top top',
      end: '+=360%',
      pin: true,
      scrub: 0.8
    }
  });

  // Interactive Mirror Symmetry Flip inside Panel 4
  var mirrorBtn = document.getElementById('mirrorFlipBtn');
  var mirrorSvg = document.getElementById('mirrorPitchGroup');
  var mirrorLabel = document.getElementById('mirrorStateLabel');
  var isMirrored = false;
  if(mirrorBtn && mirrorSvg){
    mirrorBtn.addEventListener('click', function(){
      isMirrored = !isMirrored;
      gsap.to(mirrorSvg, {
        scaleX: isMirrored ? -1 : 1,
        transformOrigin: '50% 50%',
        duration: 0.65,
        ease: 'power3.inOut'
      });
      if(mirrorLabel){
        mirrorLabel.textContent = isMirrored
          ? 'VISTA ESPEJO INVERTIDA (X → -X · Equipos intercambiados) · Favoritismo: 0.000000'
          : 'VISTA ORIGINAL (X normal) · Favoritismo: 0.000000';
      }
    });
  }
}

/* ================= 06 · CAPÍTULO 4: STICKY STACKING CARDS (3 IDEAS DESCARTADAS + 1 GANADORA) ================= */
var cards = gsap.utils.toArray('#stackZone .stack-card');
cards.forEach(function(card, i){
  if(i === cards.length - 1) return;
  gsap.fromTo(card,
    { scale: 1, filter: 'brightness(1)' },
    {
      scale: 1 - (cards.length - 1 - i) * 0.025,
      filter: 'brightness(0.84)',
      ease: 'none',
      scrollTrigger:{
        trigger: cards[i + 1],
        start: 'top 82%',
        end: 'top 20%',
        scrub: true
      }
    }
  );
});

/* ================= 51 + 52 · CAPÍTULO 5: SVG LINE DRAW + SECTION COLOR MORPH (LAS 4 LLAVES) ================= */
(function(){
  var p = document.getElementById('ldrawPath');
  var d = document.getElementById('ldrawDot');
  var keyScoreEl = document.getElementById('ldrawLiveScore');
  if(p && d){
    var len = p.getTotalLength();
    p.style.strokeDasharray = len;
    p.style.strokeDashoffset = len;
    gsap.to(p,{
      strokeDashoffset: 0,
      ease: 'none',
      scrollTrigger:{
        trigger: '#forensicGrid',
        start: 'top 72%',
        end: 'bottom 68%',
        scrub: 0.5,
        onUpdate: function(s){
          var pt = p.getPointAtLength(s.progress * len);
          d.setAttribute('cx', pt.x);
          d.setAttribute('cy', pt.y);
          if(keyScoreEl){
            var sc = 80.0 + s.progress * 8.8;
            keyScoreEl.textContent = sc.toFixed(1) + '%';
          }
        }
      }
    });
  }

  var cm = document.getElementById('cmorph');
  if(cm){
    gsap.utils.toArray('#cmorph .cm-sec').forEach(function(s){
      ScrollTrigger.create({
        trigger: s,
        start: 'top 60%',
        end: 'bottom 60%',
        onEnter: function(){
          gsap.to(cm,{backgroundColor: s.dataset.bg, duration: 0.65, ease: 'power2.out'});
        },
        onEnterBack: function(){
          gsap.to(cm,{backgroundColor: s.dataset.bg, duration: 0.65, ease: 'power2.out'});
        }
      });
    });
  }
})();

/* ================= 75 · CAPÍTULO 6: 3D FLY-THROUGH TUNNEL (VUELO POR LOS 90 MINUTOS) ================= */
(function(){
  var stage = document.getElementById('tunnelStage'),
      cam   = document.getElementById('tunnelCam'),
      pctEl = document.getElementById('tunnelPct'),
      barEl = document.getElementById('tunnelBar'),
      beatEl= document.getElementById('tunnelBeat');
  if(!stage || !cam) return;

  var ITEMS = [
    {type:'w', text:'MINUTO 00:00 · PITIDO INICIAL', cls:'a', x:0, y:0, z:-350, sz:42},
    {type:'card', min:'MIN 02:14 · JUEGO VIVO', title:'199.319 Instantes en Marcha', desc:'El Barça toma el mando con pases rápidos en campo propio.', stat:'98.6% Vivo vs Parado', x:-380, y:-185, z:-950, w:280, h:185, rz:-2},
    {type:'card', min:'MIN 08:40 · CÁMARA TV', title:'Cero Pelotas Fantasma', desc:'La cámara hace zoom; nuestro filtro físico mantiene la jugada intacta.', stat:'0 Teletransportes', x:380, y:185, z:-1500, w:280, h:185, rz:2},
    {type:'w', text:'61.2% → 73.1% · FÍSICA REAL', cls:'o', x:0, y:0, z:-2100, sz:44},
    {type:'card', min:'MIN 19:05 · SAQUE DE BANDA', title:'El Reloj Exacto en Banda', desc:'Ignoramos los 4s con el balón en las manos y abrimos en el latigazo exacto.', stat:'+5.7 Puntos de Golpe', x:-380, y:185, z:-2700, w:285, h:190, rz:2},
    {type:'card', min:'MIN 31:22 · DUELO EN MEDIOCAMPO', title:'Rebote Casual vs Robo Real', desc:'Un despeje golpea la espinilla rival a 18 m/s sin control: la jugada sigue.', stat:'Sin Grietas Falsas', x:380, y:-185, z:-3250, w:285, h:190, rz:-2},
    {type:'w', text:'SIMETRÍA ESPEJO · 0.000000 SESGO', cls:'a', x:0, y:0, z:-3850, sz:42},
    {type:'card', min:'MIN 45:00 · DESCANSO (1ª PARTE)', title:'Dominio Blaugrana 1ª Parte', desc:'Auditoría de los primeros 45 minutos sin intervención humana.', stat:'64.5% Barça · 35.5% Ath', x:-380, y:-185, z:-4450, w:285, h:190, rz:-2},
    {type:'card', min:'MIN 54:18 · CAZADOR RELÁMPAGO', title:'Robo Real de 0,88 Segundos', desc:'El Athletic frena en seco un pase largo en el círculo central.', stat:'+1.8 Puntos Limpios', x:380, y:185, z:-5050, w:285, h:190, rz:2},
    {type:'w', text:'6 ESPECIALISTAS EN CONSENSO', cls:'o', x:0, y:0, z:-5650, sz:44},
    {type:'card', min:'MIN 68:50 · PRESIÓN ALTA', title:'2ª Parte Más Disputada', desc:'El Athletic adelanta líneas; el modelo capta el cambio táctico real.', stat:'57.6% Barça · 42.4% Ath', x:-380, y:185, z:-6250, w:285, h:190, rz:2},
    {type:'card', min:'MIN 79:12 · RITMO LALIGA', title:'2,27 Posesiones por Minuto', desc:'229 jugadas vivas en todo el partido con CERO micro-posesiones (<0,5s).', stat:'0 Parpadeos (<0.5s)', x:380, y:-185, z:-6850, w:285, h:190, rz:-2},
    {type:'card', min:'MIN 86:45 · 4 JUECES EXTERNOS', title:'7 de 7 Partidos Mejorados', desc:'Validado en 11 partidos completos de 3 sistemas de cámaras distintos.', stat:'7 / 7 Pleno Total', x:-380, y:-185, z:-7450, w:285, h:185, rz:-2},
    {type:'w', text:'88.8% · CAMPEONES DEL RETO 🏆', cls:'a', x:0, y:0, z:-8150, sz:52}
  ];
  var MAX_Z = 8150;
  var nodes = [];

  ITEMS.forEach(function(it){
    var el = document.createElement('div');
    if(it.type === 'card'){
      el.className = 't-plane';
      el.style.width  = it.w + 'px';
      el.style.minHeight = it.h + 'px';
      el.style.marginLeft = (-it.w / 2) + 'px';
      el.style.marginTop  = (-it.h / 2) + 'px';
      el.innerHTML =
        '<div><div class="tp-min">' + it.min + '</div>' +
        '<h5>' + it.title + '</h5>' +
        '<p>' + it.desc + '</p></div>' +
        '<div class="tp-stat">' + it.stat + '</div>';
    } else {
      el.className = 't-word ' + (it.cls || 'o');
      el.style.fontSize = it.sz + 'px';
      el.textContent = it.text;
    }
    cam.appendChild(el);
    nodes.push({el: el, it: it});
  });

  var st = {z: 0, mx: 0, my: 0, tx: 0, ty: 0};
  stage.addEventListener('pointermove',function(e){
    var r = stage.getBoundingClientRect();
    st.tx = ((e.clientX - r.left) / r.width  - 0.5) * 2;
    st.ty = ((e.clientY - r.top)  / r.height - 0.5) * 2;
  });
  stage.addEventListener('pointerleave',function(){ st.tx = 0; st.ty = 0; });

  var BEATS = [
    [0.22, '01 / 04 · Pitido Inicial y Filtro Físico de TV'],
    [0.50, '02 / 04 · El Reloj Exacto de los Saques de Banda (+5.7 pts)'],
    [0.78, '03 / 04 · Robo Relámpago y Ritmo Real de los 90 Minutos'],
    [1.01, '04 / 04 · Marcador Final: 88.8% de Precisión en Jugadas Completas 🏆']
  ];

  function render(){
    st.mx += (st.tx - st.mx) * 0.08;
    st.my += (st.ty - st.my) * 0.08;
    var lookX = -st.my * 6, lookY = st.mx * 9;
    var swayX = -st.mx * 55, swayY = -st.my * 35;
    cam.style.transform = 'rotateX(' + lookX.toFixed(2) + 'deg) rotateY(' + lookY.toFixed(2) + 'deg)';

    for(var i = 0; i < nodes.length; i++){
      var n = nodes[i], it = n.it, relZ = it.z + st.z;
      var minZ = (it.type === 'w') ? -1050 : -1750;
      var maxZ = (it.type === 'w') ? 220   : 280;
      if(relZ > maxZ || relZ < minZ){
        n.el.style.opacity = 0;
        n.el.style.visibility = 'hidden';
        continue;
      }
      n.el.style.visibility = 'visible';
      n.el.style.zIndex = String(Math.round(10000 + relZ));
      var a = 1;
      if(it.type === 'w'){
        if(relZ < -650) a = clamp((relZ - minZ) / 400, 0, 1);
        else if(relZ > 40) a = clamp(1 - (relZ - 40) / (maxZ - 40), 0, 1);
      } else {
        if(relZ < -1250) a = clamp((relZ - minZ) / 500, 0, 1);
        else if(relZ > 40) a = clamp(1 - (relZ - 40) / (maxZ - 40), 0, 1);
      }
      n.el.style.opacity = a.toFixed(3);
      var perspComp = (it.type === 'card') ? (1 + Math.max(0, -relZ) / 820) : 1;
      var px = it.x * perspComp + swayX * (1 + relZ / 3600);
      var py = it.y * perspComp + swayY * (1 + relZ / 3600);
      if(it.type === 'w'){
        n.el.style.transform = 'translate3d(' + px.toFixed(1) + 'px,' + py.toFixed(1) + 'px,' + relZ.toFixed(1) + 'px) translate(-50%,-50%)';
      } else {
        n.el.style.transform = 'translate3d(' + px.toFixed(1) + 'px,' + py.toFixed(1) + 'px,' + relZ.toFixed(1) + 'px) rotateZ(' + (it.rz || 0) + 'deg)';
      }
    }
  }

  ScrollTrigger.create({
    trigger: stage,
    start: 'top top',
    end: '+=360%',
    pin: true,
    scrub: 0.8,
    onUpdate: function(s){
      st.z = s.progress * MAX_Z;
      var minPlayed = Math.round(s.progress * 90);
      if(pctEl) pctEl.textContent = 'MIN ' + String(minPlayed).padStart(2, '0') + ':00';
      if(barEl) barEl.style.width = (s.progress * 100) + '%';
      for(var i = 0; i < BEATS.length; i++){
        if(s.progress <= BEATS[i][0]){
          if(beatEl) beatEl.textContent = BEATS[i][1];
          break;
        }
      }
      render();
    }
  });
  visLoop(stage, render);
  render();
})();

/* ================= 07 · SCROLL-TRIGGERED STAT COUNT-UPS ================= */
gsap.utils.toArray('#stats [data-count]').forEach(function(el){
  var target   = parseFloat(el.dataset.count);
  var decimals = parseInt(el.dataset.decimals || '0', 10);
  var o = {v: target * 0.92};
  ScrollTrigger.create({
    trigger: el,
    start: 'top 92%',
    once: true,
    onEnter: function(){
      gsap.to(o,{
        v: target,
        duration: 0.14,
        ease: 'power2.out',
        onUpdate: function(){
          el.textContent = decimals > 0 ? o.v.toFixed(decimals) : Math.round(o.v).toLocaleString('es-ES');
        }
      });
    }
  });
});

/* ================= 54 · SPOTLIGHT BORDER GRID (AUDITORÍA DE LOS 85 MINUTOS SECRETOS) ================= */
(function(){
  var g = document.getElementById('sg2');
  if(!g) return;
  var cards = g.querySelectorAll('.sg2-card');
  g.addEventListener('pointermove',function(e){
    cards.forEach(function(c){
      var r = c.getBoundingClientRect();
      c.style.setProperty('--x',(e.clientX - r.left) + 'px');
      c.style.setProperty('--y',(e.clientY - r.top)  + 'px');
    });
  });
})();

/* ================= 10 · MAGNETIC BUTTONS & 13 · 3D TILT CARDS ================= */
document.querySelectorAll('.mag').forEach(function(b){
  var s = b.querySelector('span') || b;
  b.addEventListener('pointermove',function(e){
    var r = b.getBoundingClientRect();
    var x = e.clientX - (r.left + r.width / 2);
    var y = e.clientY - (r.top  + r.height / 2);
    gsap.to(b,{x: x * 0.28, y: y * 0.28, duration: 0.35, ease: 'power3.out'});
    if(s !== b) gsap.to(s,{x: x * 0.15, y: y * 0.15, duration: 0.35, ease: 'power3.out'});
  });
  b.addEventListener('pointerleave',function(){
    gsap.to([b, s],{x: 0, y: 0, duration: 0.7, ease: 'elastic.out(1, 0.4)'});
  });
});

document.querySelectorAll('.tilt').forEach(function(card){
  var g = card.querySelector('.glare');
  card.addEventListener('pointermove',function(e){
    var r  = card.getBoundingClientRect();
    var px = (e.clientX - r.left) / r.width;
    var py = (e.clientY - r.top)  / r.height;
    var rx = (0.5 - py) * 14, ry = (px - 0.5) * 18;
    card.style.transform = 'perspective(900px) rotateX(' + rx + 'deg) rotateY(' + ry + 'deg) scale3d(1.015,1.015,1.015)';
    if(g) g.style.background = 'radial-gradient(380px circle at ' + (px * 100) + '% ' + (py * 100) + '%, rgba(245,184,65,.22), transparent 60%)';
  });
  card.addEventListener('pointerleave',function(){
    card.style.transition = 'transform .55s cubic-bezier(.2,.8,.2,1)';
    card.style.transform  = 'perspective(900px) rotateX(0) rotateY(0) scale3d(1,1,1)';
    setTimeout(function(){ card.style.transition = ''; }, 550);
  });
});

/* Recalculate all ScrollTrigger positions after pinned stages are mounted */
ScrollTrigger.sort();
ScrollTrigger.refresh();
window.addEventListener('load', function(){ ScrollTrigger.refresh(); });

})();
