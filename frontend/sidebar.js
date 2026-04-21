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
 *
 * Labels are looked up via window.Amphion.i18n.t(); when the language
 * changes we re-render the navigation labels and re-derive the connection
 * label from its last known state.
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
      key: 'tsasr',
      href: 'tsasr.html',
      i18nKey: 'nav.tsasr',
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
    attachBehaviors(aside);
    if (i18n && typeof i18n.applyTranslations === 'function') {
      i18n.applyTranslations(aside);
    }
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

    aside.querySelectorAll('.app-lang-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const lng = btn.getAttribute('data-lang');
        if (!lng || !i18n) return;
        i18n.setLang(lng);
      });
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
    setConnectionState,
  };
})();
