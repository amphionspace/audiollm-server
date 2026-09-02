/**
 * Client-side router for the Amphion demos.
 *
 * The demo pages used to be independent HTML documents that the
 * browser navigated between via cross-document View Transitions. That
 * forced the entire JS context (i18n dictionaries, audio worklets,
 * Tailwind styles, font cache) to be re-bootstrapped on every nav,
 * which is the physical ceiling of an MPA. We replaced that with this
 * SPA shell: the entry HTML still works as a static document so deep
 * links keep working, but once the page loads the router intercepts
 * sidebar clicks and the browser back/forward stack, fetches the next
 * page's HTML, and swaps the ``<main class="app-main">`` body in place
 * inside a same-document ``startViewTransition``.
 *
 * Each page-specific script registers a module on
 * ``window.AmphionPages`` that exposes a single ``init()`` returning a
 * ``dispose()`` callback. The router calls dispose on the outgoing
 * page (closing WebSockets, releasing AudioContexts, aborting fetches)
 * before mounting the new one, so resources never leak across nav.
 *
 * Public surface (``window.AmphionRouter``):
 *   navigate(href, { replace? })  - programmatic nav
 *   bootstrap()                   - called once on first load to
 *                                   attach listeners and run the
 *                                   current page's init
 *
 * Falls back to ``location.assign(href)`` whenever anything goes
 * wrong (fetch failure, missing module, init throws), so the worst
 * case is the pre-SPA behaviour rather than a broken page.
 */
(() => {
  'use strict';

  let currentKey = null;
  let currentDispose = noop;
  let inFlight = null;
  let booted = false;

  function noop() {}

  function getI18n() {
    return (window.Amphion && window.Amphion.i18n) || null;
  }

  function getPageModule(key) {
    return (window.AmphionPages && window.AmphionPages[key]) || null;
  }

  function safeDispose(fn) {
    if (typeof fn !== 'function') return;
    try { fn(); } catch (err) {
      console.error('[router] dispose failed:', err);
    }
  }

  function safeInit(key) {
    const mod = getPageModule(key);
    if (!mod || typeof mod.init !== 'function') {
      console.warn('[router] no module registered for page key:', key);
      return noop;
    }
    try {
      const dispose = mod.init();
      return typeof dispose === 'function' ? dispose : noop;
    } catch (err) {
      console.error('[router] init failed for', key, err);
      return noop;
    }
  }

  function setSidebarActive(key) {
    const sb = window.AmphionSidebar;
    if (sb && typeof sb.setActive === 'function') {
      try { sb.setActive(key); } catch (_) { /* ignore */ }
    }
  }

  function hardNavigate(href) {
    try {
      window.location.assign(href);
    } catch (_) {
      window.location.href = href;
    }
  }

  async function fetchAndParse(href, signal) {
    const resp = await fetch(href, {
      signal,
      // Avoid serving a stale ETag-cached document with a different
      // Content-Type from the static mount.
      headers: { 'Accept': 'text/html' },
      credentials: 'same-origin',
    });
    if (!resp.ok) {
      const err = new Error('HTTP ' + resp.status);
      err.status = resp.status;
      throw err;
    }
    const html = await resp.text();
    const doc = new DOMParser().parseFromString(html, 'text/html');
    return doc;
  }

  async function navigate(href, opts) {
    const options = opts || {};
    const replace = !!options.replace;
    const url = new URL(href, location.href);
    const sameOrigin = url.origin === location.origin;
    if (!sameOrigin) {
      hardNavigate(href);
      return;
    }
    // Already there: nothing to do.
    if (!options.force && url.pathname === location.pathname) {
      return;
    }

    if (inFlight) {
      try { inFlight.abort(); } catch (_) { /* ignore */ }
    }
    const ctrl = new AbortController();
    inFlight = ctrl;

    let doc;
    try {
      doc = await fetchAndParse(url.pathname + url.search, ctrl.signal);
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      console.error('[router] fetch failed:', err);
      hardNavigate(href);
      return;
    } finally {
      if (inFlight === ctrl) inFlight = null;
    }

    const newMain = doc.querySelector('main.app-main');
    const newKey = doc.body && doc.body.dataset ? doc.body.dataset.page : null;
    if (!newMain || !newKey) {
      console.warn('[router] missing <main.app-main> or <body data-page>; falling back');
      hardNavigate(href);
      return;
    }

    // Translate the new fragment before it enters the live DOM so the
    // user never sees the hard-coded English placeholders flash by.
    const i18n = getI18n();
    if (i18n && typeof i18n.applyTranslations === 'function') {
      try { i18n.applyTranslations(newMain); } catch (_) { /* ignore */ }
    }

    // The fetched <title> is the raw English source; resolve via i18n
    // (using the page's ``data-i18n-doc-title`` key) so we don't blow
    // away a Chinese page title with English on every nav.
    let newTitle = null;
    const newTitleNode = doc.querySelector('[data-i18n-doc-title]')
      || doc.querySelector('title');
    if (newTitleNode) {
      const titleKey = newTitleNode.getAttribute
        ? newTitleNode.getAttribute('data-i18n-doc-title')
        : null;
      if (titleKey && i18n && typeof i18n.t === 'function') {
        try { newTitle = i18n.t(titleKey); } catch (_) { newTitle = null; }
      }
      if (!newTitle) newTitle = newTitleNode.textContent || null;
    }

    // Take a snapshot of the dispose to call AFTER the cross-fade has
    // captured the old state — calling it before would let the page
    // tear down its DOM before VT can capture it.
    const outgoingDispose = currentDispose;
    currentDispose = noop;

    const performSwap = () => {
      const oldMain = document.querySelector('main.app-main');
      if (oldMain) {
        oldMain.replaceWith(newMain);
      } else {
        // Defensive: if the document somehow lost its main element,
        // attach the fragment to .app-shell.
        const shell = document.querySelector('.app-shell') || document.body;
        shell.appendChild(newMain);
      }
      if (document.body && document.body.dataset) {
        document.body.dataset.page = newKey;
      }
      if (newTitle) document.title = newTitle;
      setSidebarActive(newKey);
    };

    if (typeof document.startViewTransition === 'function') {
      const transition = document.startViewTransition(() => {
        // Run the dispose first so any UI-state cleanup the outgoing
        // page wants to commit lands before the new <main> replaces
        // its DOM. We're already inside the VT callback, so VT has
        // already snapshotted the old state.
        safeDispose(outgoingDispose);
        performSwap();
      });
      try { await transition.updateCallbackDone; } catch (_) { /* ignore */ }
    } else {
      safeDispose(outgoingDispose);
      performSwap();
    }

    // Mount the new page after the swap so its init() sees the
    // already-attached DOM.
    currentKey = newKey;
    currentDispose = safeInit(newKey);

    const target = url.pathname + url.search + url.hash;
    try {
      if (replace) history.replaceState({ key: newKey }, '', target);
      else history.pushState({ key: newKey }, '', target);
    } catch (_) { /* ignore */ }
  }

  function onAnchorClick(ev) {
    if (ev.defaultPrevented) return;
    if (ev.button !== 0) return;
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;

    const path = (ev.composedPath && ev.composedPath()) || [];
    let a = null;
    for (const node of path) {
      if (node && node.nodeType === 1 && node.tagName === 'A') {
        a = node;
        break;
      }
    }
    if (!a) {
      // Fallback for older composedPath implementations.
      a = ev.target && ev.target.closest ? ev.target.closest('a') : null;
    }
    if (!a) return;
    if (!a.classList.contains('app-nav-item')) return;
    if (a.target && a.target !== '' && a.target !== '_self') return;
    if (a.hasAttribute('download')) return;

    const href = a.getAttribute('href');
    if (!href) return;
    let url;
    try { url = new URL(href, location.href); } catch (_) { return; }
    if (url.origin !== location.origin) return;

    ev.preventDefault();
    if (url.pathname === location.pathname) return;
    navigate(url.pathname + url.search + url.hash);
  }

  function onPopState(ev) {
    // Always reload the current path; replace=true so we don't push a
    // duplicate entry on top of the one the browser just popped.
    navigate(location.pathname + location.search, {
      replace: true,
      force: true,
    });
  }

  function bootstrap() {
    if (booted) return;
    booted = true;
    currentKey = (document.body && document.body.dataset)
      ? document.body.dataset.page
      : null;
    currentDispose = currentKey ? safeInit(currentKey) : noop;
    setSidebarActive(currentKey);
    document.addEventListener('click', onAnchorClick, { capture: true });
    window.addEventListener('popstate', onPopState);
    // Anchor the current entry in history so popstate works on the
    // very first back/forward action.
    try {
      history.replaceState(
        { key: currentKey },
        '',
        location.pathname + location.search + location.hash
      );
    } catch (_) { /* ignore */ }
  }

  window.AmphionRouter = {
    navigate,
    bootstrap,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootstrap, { once: true });
  } else {
    bootstrap();
  }
})();
