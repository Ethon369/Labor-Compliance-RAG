/**
 * 主题与外观面板（三页共用，零构建、原生 JS）。
 *
 * 只负责三件事，页面自身逻辑一概不碰：
 *   1. 主题（cool / dark）读写 localStorage['lcr_theme']，无记录时跟随系统
 *   2. 信息密度（comfortable / compact）读写 localStorage['lcr_density']
 *   3. 外观面板的开关与分段按钮的选中态
 *
 * 约定：页面里放 <div class="tw" id="tw" data-open="false"> 结构即可获得面板；
 *       [data-key="theme"] / [data-key="density"] 的 .seg 由本文件接管，
 *       其它 data-key（如 index 的视图预览）由页面自己的脚本处理。
 */
'use strict';

(function () {
  var root = document.documentElement;
  var THEME_KEY = 'lcr_theme';
  var DENSITY_KEY = 'lcr_density';

  function store(key, val) {
    try { localStorage.setItem(key, val); } catch (e) { /* 隐私模式忽略 */ }
  }
  function read(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function syncSeg(key, val) {
    document.querySelectorAll('[data-key="' + key + '"] button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.val === val));
    });
  }

  function setTheme(name) {
    if (name !== 'cool' && name !== 'dark') return;
    root.dataset.theme = name;
    store(THEME_KEY, name);
    syncSeg('theme', name);
  }

  function setDensity(name) {
    if (name !== 'comfortable' && name !== 'compact') return;
    root.dataset.density = name;
    store(DENSITY_KEY, name);
    syncSeg('density', name);
  }

  /* ---- 初始化：已存值优先，否则跟随系统 ---- */
  var savedTheme = read(THEME_KEY);
  if (savedTheme === 'cool' || savedTheme === 'dark') {
    root.dataset.theme = savedTheme;
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.dataset.theme = 'dark';
  }
  var savedDensity = read(DENSITY_KEY);
  if (savedDensity === 'comfortable' || savedDensity === 'compact') {
    root.dataset.density = savedDensity;
  }

  /* ---- 分段按钮 ---- */
  document.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('.seg button') : null;
    if (!btn) return;
    var seg = btn.closest('.seg');
    var key = seg.dataset.key;
    if (key === 'theme') { setTheme(btn.dataset.val); return; }
    if (key === 'density') { setDensity(btn.dataset.val); return; }
    /* 其它 key（视图预览等）由页面脚本处理，这里只同步选中态 */
    seg.querySelectorAll('button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b === btn));
    });
  });

  /* ---- 面板开关 ---- */
  var fab = document.getElementById('twFab');
  var tw = document.getElementById('tw');
  if (fab && tw) {
    fab.addEventListener('click', function () {
      tw.dataset.open = tw.dataset.open === 'true' ? 'false' : 'true';
    });
  }

  /* 暴露给页面脚本：切主题后需要重绘的场景可用 */
  window.__appearance = { setTheme: setTheme, setDensity: setDensity };
})();
