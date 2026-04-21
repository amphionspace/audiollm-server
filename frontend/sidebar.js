/**
 * Shared sidebar navigation for Amphion demos.
 *
 * The host HTML only needs to set `<body data-page="asr|emotion|tsasr">`
 * and include this script; the sidebar DOM is injected at runtime so the
 * three pages don't need to keep duplicated markup in sync.
 *
 * Exposes on window.AmphionSidebar:
 *   setConnectionState(state, label?) - updates the bottom dot + label
 *     state in: idle | ready | listening | analyzing | busy | pending |
 *               connected | error | offline
 */
(() => {
  'use strict';

  const NAV_ITEMS = [
    {
      key: 'asr',
      href: 'index.html',
      label: 'Realtime ASR',
      icon:
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M12 3a3 3 0 00-3 3v6a3 3 0 006 0V6a3 3 0 00-3-3z"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M19 11a7 7 0 01-14 0M12 18v3M8 21h8"/>',
    },
    {
      key: 'emotion',
      href: 'emotion.html',
      label: 'Emotion',
      icon:
        '<circle cx="12" cy="12" r="9" stroke-width="1.8"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M9 10h.01M15 10h.01M9 15c.9.8 1.9 1.2 3 1.2s2.1-.4 3-1.2"/>',
    },
    {
      key: 'tsasr',
      href: 'tsasr.html',
      label: 'Target Speaker',
      icon:
        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M16 11a4 4 0 10-8 0 4 4 0 008 0z"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M4 20c1.5-3 4.5-4.5 8-4.5s6.5 1.5 8 4.5"/>'
        + '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"'
        + ' d="M19 4l2 2m0-2l-2 2"/>',
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

  function getActiveKey() {
    const fromBody = document.body && document.body.dataset.page;
    if (fromBody) return fromBody;
    const path = (location.pathname || '').toLowerCase();
    if (path.endsWith('emotion.html')) return 'emotion';
    if (path.endsWith('tsasr.html')) return 'tsasr';
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
        + '<span class="app-nav-label">' + item.label + '</span>'
        + '</a>'
      );
    }).join('');

    return (
      '<div class="app-brand">'
      + '<div class="app-brand-logo" aria-hidden="true">'
      + BRAND_ICON_SVG
      + '</div>'
      + '<div class="app-brand-text">'
      + '<div class="app-brand-title">Amphion</div>'
      + '<div class="app-brand-sub">Speech Demo</div>'
      + '</div>'
      + '</div>'
      + '<nav class="app-nav" aria-label="Primary">'
      + items
      + '</nav>'
      + '<div class="app-sidebar-foot">'
      + '<span class="app-conn-dot" data-state="idle" aria-hidden="true"></span>'
      + '<span class="app-conn-label">Idle</span>'
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
    attachBehaviors(aside);
    return aside;
  }

  function attachBehaviors(aside) {
    const mainEl = document.querySelector('.app-main');
    const supportsVT =
      typeof document !== 'undefined'
      && typeof document.startViewTransition === 'function';

    aside.querySelectorAll('.app-nav-item').forEach((a) => {
      a.addEventListener('click', (ev) => {
        const isActive = a.classList.contains('is-active');
        if (isActive) {
          ev.preventDefault();
          return;
        }
        aside.querySelectorAll('.app-nav-item').forEach((el) => {
          el.classList.toggle('is-active', el === a);
        });
        if (!supportsVT && mainEl) {
          mainEl.classList.add('is-page-leaving');
        }
      });
    });
  }

  function setConnectionState(state, label) {
    const dot = document.querySelector('.app-conn-dot');
    const lbl = document.querySelector('.app-conn-label');
    if (dot) {
      dot.dataset.state = state || 'idle';
    }
    if (lbl) {
      lbl.textContent = label || defaultLabelForState(state);
    }
  }

  function defaultLabelForState(state) {
    switch (state) {
      case 'connected':
      case 'ready':
        return 'Connected';
      case 'pending':
        return 'Connecting...';
      case 'listening':
        return 'Listening';
      case 'analyzing':
        return 'Analyzing';
      case 'busy':
        return 'Working';
      case 'error':
        return 'Error';
      case 'offline':
        return 'Offline';
      case 'idle':
      default:
        return 'Idle';
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount, { once: true });
  } else {
    mount();
  }

  window.AmphionSidebar = {
    mount,
    setConnectionState,
  };
})();
