/**
 * BCN Gradient Depression — board rail & summary sheet
 * ---------------------------------------------------------------------
 * 1. The chapter magnets on the board's top rail: they follow `sw:scroll`
 *    from board.js and jump to a chapter on click.
 * 2. The summary sheet (`#sw-explorer-modal`) with its three tabs.
 * The running score lives on the board itself (board.js draws it in chalk).
 */
(function () {
  'use strict';

  function initRail() {
    var magnets = Array.prototype.slice.call(document.querySelectorAll('.sw-step-pill'));
    var current = -1;

    magnets.forEach(function (m, idx) {
      m.addEventListener('click', function (e) {
        e.preventDefault();
        if (window.bcnWorld && typeof window.bcnWorld.jumpTo === 'function') window.bcnWorld.jumpTo(idx);
      });
    });

    document.addEventListener('sw:scroll', function (e) {
      var idx = (e.detail && typeof e.detail.activeIndex === 'number') ? e.detail.activeIndex : 0;
      if (idx === current) return;
      current = idx;
      magnets.forEach(function (m, i) {
        m.classList.toggle('is-active', i === idx);
        m.classList.toggle('is-past', i < idx);
        if (i === idx) m.setAttribute('aria-current', 'step'); else m.removeAttribute('aria-current');
      });
    });
  }

  function initExplorerModal() {
    var modal = document.getElementById('sw-explorer-modal');
    if (!modal) return;

    var closeBtns = modal.querySelectorAll('[data-close-explorer]');
    var tabBtns = modal.querySelectorAll('[data-explorer-tab]');
    var tabPanels = modal.querySelectorAll('[data-explorer-panel]');

    var returnFocus = null;
    function openExplorer(tabName) {
      returnFocus = document.activeElement;
      modal.classList.add('is-open');
      modal.setAttribute('aria-hidden', 'false');
      if (window.bcnLenis) window.bcnLenis.stop();
      if (tabName) activateTab(tabName);
      var closeBtn = modal.querySelector('[data-close-explorer]');
      if (closeBtn) closeBtn.focus();
    }

    function closeExplorer() {
      modal.classList.remove('is-open');
      modal.setAttribute('aria-hidden', 'true');
      if (window.bcnLenis) window.bcnLenis.start();
      if (returnFocus && returnFocus !== document.body && returnFocus.focus) returnFocus.focus();
      else if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    }

    modal.addEventListener('click', function (e) {
      if (e.target === modal) closeExplorer();
    });

    function activateTab(tabName) {
      tabBtns.forEach(function (btn) {
        var on = btn.getAttribute('data-explorer-tab') === tabName;
        btn.classList.toggle('is-active', on);
        btn.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      tabPanels.forEach(function (panel) {
        panel.classList.toggle('is-active', panel.getAttribute('data-explorer-panel') === tabName);
      });
    }

    tabBtns.forEach(function (btn) {
      btn.setAttribute('role', 'tab');
      btn.addEventListener('click', function () {
        activateTab(btn.getAttribute('data-explorer-tab'));
      });
    });

    closeBtns.forEach(function (btn) {
      btn.addEventListener('click', closeExplorer);
    });

    window.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.classList.contains('is-open')) {
        closeExplorer();
      }
    });

    document.addEventListener('click', function (e) {
      var link = e.target.closest('a[href^="#"]');
      if (!link) return;
      var href = link.getAttribute('href');
      if (href === '#explorar' || href === '#escalera') {
        e.preventDefault();
        openExplorer('escalera');
      } else if (href === '#jueces') {
        e.preventDefault();
        openExplorer('jueces');
      } else if (href === '#radiografia') {
        e.preventDefault();
        openExplorer('radiografia');
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      initRail();
      initExplorerModal();
    });
  } else {
    initRail();
    initExplorerModal();
  }
})();
