/**
 * Shared sidebar navigation for the Amphion demos.
 *
 * The sidebar lives outside ``<main class="app-main">`` and is mounted
 * exactly once per document load — the SPA router (frontend/router.js)
 * never re-renders it, just calls ``setActive(key)`` after a successful
 * navigation. Anchor clicks inside the sidebar are intercepted at the
 * document level by the router, so the click handlers here only need
 * to handle language switching; the active state is driven from the
 * router so it stays consistent with whatever ``<main>`` is currently
 * mounted (including the case where the user came in via a deep link
 * to /emotion.html or via browser back/forward).
 *
 * Public surface (``window.AmphionSidebar``):
 *   mount()
 *   setActive(key)
 *   setConnectionState(state, label?)
 */
(() => {
  'use strict';

  const i18n = (window.Amphion && window.Amphion.i18n) || null;
  function t(key, vars) {
    if (i18n && typeof i18n.t === 'function') return i18n.t(key, vars);
    if (vars && Object.prototype.hasOwnProperty.call(vars, 'defaultValue')) {
      return vars.defaultValue;
    }
    return key;
  }

  const NAV_ITEMS = [
    {
      key: 'asr',
      href: 'index.html',
      i18nKey: 'nav.asr',
      icon:
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M12 3a3 3 0 00-3 3v6a3 3 0 006 0V6a3 3 0 00-3-3z"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M19 11a7 7 0 01-14 0M12 18v3M8 21h8"/>',
    },
    {
      key: 'emotion',
      href: 'emotion.html',
      i18nKey: 'nav.emotion',
      icon:
        '<circle cx="12" cy="12" r="9" stroke-width="1.8"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M9 10h.01M15 10h.01M9 15c.9.8 1.9 1.2 3 1.2s2.1-.4 3-1.2"/>',
    },
    {
      key: 'enhanced-asr',
      href: 'enhanced-asr.html',
      i18nKey: 'nav.enhancedAsr',
      icon:
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M12 3a3 3 0 00-3 3v6a3 3 0 006 0V6a3 3 0 00-3-3zM19 11a7 7 0 01-14 0M12 18v3"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M18.5 3.5v4M16.5 5.5h4"/>',
    },
    {
      key: 'asr-test',
      href: 'asr-test.html',
      i18nKey: 'nav.asrTest',
      icon:
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M9 3h6M10 3v6.5L5.2 18.2A1.6 1.6 0 006.6 20.6h10.8a1.6 1.6 0 001.4-2.4L14 9.5V3"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M8.5 14h7"/>',
    },
  ];

  const BRAND_ICON_SVG =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" width="18" height="18">'
    + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
    + ' d="M4 12c0-4.4 3.6-8 8-8s8 3.6 8 8"/>'
    + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
    + ' d="M4 12v3a2 2 0 002 2h1v-6H6a2 2 0 00-2 2z"/>'
    + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
    + ' d="M20 12v3a2 2 0 01-2 2h-1v-6h1a2 2 0 012 2z"/>'
    + '</svg>';

  // Last-known connection state, kept so we can re-derive the label after a
  // language switch when the caller hadn't supplied one explicitly.
  let lastConnState = 'idle';
  let lastConnLabelExplicit = false;
  let lastConnLabel = null;

  function getActiveKey() {
    const fromBody = document.body && document.body.dataset.page;
    if (fromBody) return fromBody;
    const path = (location.pathname || '').toLowerCase();
    if (path.endsWith('emotion.html')) return 'emotion';
    if (path.endsWith('enhanced-asr.html')) return 'enhanced-asr';
    if (path.endsWith('asr-test.html')) return 'asr-test';
    return 'asr';
  }

  function renderSidebar(activeKey) {
    const items = NAV_ITEMS.map((item) => {
      const isActive = item.key === activeKey;
      const cls = 'app-nav-item' + (isActive ? ' is-active' : '');
      return (
        '<a class="' + cls + '" href="' + item.href + '"'
        + ' data-nav-key="' + item.key + '"'
        + (isActive ? ' aria-current="page"' : '') + '>'
        + '<svg class="app-nav-icon" viewBox="0 0 24 24" fill="none"'
        + ' stroke="currentColor" aria-hidden="true">'
        + item.icon
        + '</svg>'
        + '<span class="app-nav-label" data-i18n="' + item.i18nKey + '">'
        + t(item.i18nKey)
        + '</span>'
        + '</a>'
      );
    }).join('');

    const langButtons = (i18n ? i18n.SUPPORTED : ['en', 'zh']).map((lng) => {
      const label = lng === 'zh' ? '中' : 'EN';
      const isActive = i18n && i18n.getLang() === lng;
      return (
        '<button type="button" class="app-lang-btn'
        + (isActive ? ' is-active' : '') + '"'
        + ' data-lang="' + lng + '"'
        + ' aria-pressed="' + (isActive ? 'true' : 'false') + '">'
        + label
        + '</button>'
      );
    }).join('');

    return (
      '<div class="app-brand">'
      + '<div class="app-brand-logo" aria-hidden="true">'
      + BRAND_ICON_SVG
      + '</div>'
      + '<div class="app-brand-text">'
      + '<div class="app-brand-title" data-i18n="sidebar.brand.title">'
      + t('sidebar.brand.title') + '</div>'
      + '<div class="app-brand-sub" data-i18n="sidebar.brand.sub">'
      + t('sidebar.brand.sub') + '</div>'
      + '</div>'
      + '</div>'
      + '<nav class="app-nav" aria-label="Primary">'
      + items
      + '</nav>'
      + '<div class="app-lang-toggle" role="group"'
      + ' data-i18n-attr-aria-label="sidebar.lang.aria"'
      + ' aria-label="' + t('sidebar.lang.aria') + '">'
      + langButtons
      + '</div>'
      + '<div class="app-sidebar-foot">'
      + '<span class="app-conn-dot" data-state="idle" aria-hidden="true"></span>'
      + '<span class="app-conn-label">' + t('common.idle') + '</span>'
      + '</div>'
    );
  }

  function mount() {
    const existing = document.querySelector('.app-sidebar');
    if (existing) return existing;
    const activeKey = getActiveKey();
    const aside = document.createElement('aside');
    aside.className = 'app-sidebar';
    aside.setAttribute('data-active', activeKey);
    aside.innerHTML = renderSidebar(activeKey);
    const host = document.querySelector('.app-shell');
    if (host) {
      host.insertBefore(aside, host.firstChild);
    } else {
      document.body.insertBefore(aside, document.body.firstChild);
    }
    attachLangBehaviour(aside);
    if (i18n && typeof i18n.applyTranslations === 'function') {
      i18n.applyTranslations(aside);
    }
    return aside;
  }

  // Only language buttons need a per-mount click handler. Nav anchors
  // are intercepted globally by the router at document level, so we
  // don't bind click listeners to each ``.app-nav-item`` here. The
  // router calls ``setActive(key)`` after a successful navigation to
  // sync the sidebar's visual state.
  function attachLangBehaviour(aside) {
    aside.querySelectorAll('.app-lang-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const lng = btn.getAttribute('data-lang');
        if (!lng || !i18n) return;
        i18n.setLang(lng);
      });
    });
  }

  function setActive(key) {
    const aside = document.querySelector('.app-sidebar');
    if (!aside) return;
    aside.setAttribute('data-active', key || '');
    aside.querySelectorAll('.app-nav-item').forEach((el) => {
      const isActive = el.getAttribute('data-nav-key') === key;
      el.classList.toggle('is-active', isActive);
      if (isActive) el.setAttribute('aria-current', 'page');
      else el.removeAttribute('aria-current');
    });
  }

  function updateLangButtons() {
    if (!i18n) return;
    const current = i18n.getLang();
    document.querySelectorAll('.app-lang-toggle .app-lang-btn').forEach((btn) => {
      const isActive = btn.getAttribute('data-lang') === current;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }

  function defaultLabelForState(state) {
    switch (state) {
      case 'connected':
      case 'ready':
        return t('common.connected');
      case 'pending':
        return t('common.connecting');
      case 'listening':
        return t('common.listening');
      case 'analyzing':
        return t('common.analyzing');
      case 'busy':
        return t('common.busy');
      case 'error':
        return t('common.error');
      case 'offline':
        return t('common.offline');
      case 'idle':
      default:
        return t('common.idle');
    }
  }

  function setConnectionState(state, label) {
    const dot = document.querySelector('.app-conn-dot');
    const lbl = document.querySelector('.app-conn-label');
    lastConnState = state || 'idle';
    if (label != null) {
      lastConnLabelExplicit = true;
      lastConnLabel = label;
    } else {
      lastConnLabelExplicit = false;
      lastConnLabel = null;
    }
    if (dot) {
      dot.dataset.state = lastConnState;
    }
    if (lbl) {
      lbl.textContent = label != null ? label : defaultLabelForState(lastConnState);
    }
  }

  function refreshConnectionLabel() {
    if (lastConnLabelExplicit) return;
    const lbl = document.querySelector('.app-conn-label');
    if (lbl) {
      lbl.textContent = defaultLabelForState(lastConnState);
    }
  }

  if (i18n && typeof i18n.onChange === 'function') {
    i18n.onChange(() => {
      updateLangButtons();
      refreshConnectionLabel();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount, { once: true });
  } else {
    mount();
  }

  window.AmphionSidebar = {
    mount,
    setActive,
    setConnectionState,
  };
})();
