/**
 * 多页共用的认证与请求工具（零构建、原生 JS，供 index/login/kb 三个页面 include）。
 *
 * 约定：token 存 localStorage（lcr_token / lcr_username / lcr_role），刷新不丢；
 * api() 自动带 Authorization；401 时的行为由调用方决定——登录才允许的页面（kb.html）
 * 用默认的"跳登录页"，游客可用的页面（index.html）传 no401 阻止跳转。
 */
'use strict';

function getToken() { return localStorage.getItem('lcr_token') || ''; }
function getUsername() { return localStorage.getItem('lcr_username') || ''; }
function getRole() { return localStorage.getItem('lcr_role') || 'user'; }
function isAdmin() { return getRole() === 'admin'; }

function saveAuth(data) {
  localStorage.setItem('lcr_token', data.token);
  localStorage.setItem('lcr_username', data.username);
  localStorage.setItem('lcr_role', data.role || 'user');
}

function clearAuth() {
  localStorage.removeItem('lcr_token');
  localStorage.removeItem('lcr_username');
  localStorage.removeItem('lcr_role');
}

/**
 * fetch 封装：自动带 token、统一错误为 {status, detail}。
 * opts: { method, json, form, no401 }  — json 与 form 二选一。
 * 抛出的 Error 带 .status 和 .detail 字段，调用方按需展示。
 */
async function api(path, opts) {
  opts = opts || {};
  const headers = {};
  if (getToken()) headers.Authorization = 'Bearer ' + getToken();

  const init = { method: opts.method || 'GET', headers };
  if (opts.json !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.json);
  } else if (opts.form !== undefined) {
    init.body = opts.form;                       // FormData 由浏览器设 Content-Type
  }

  let resp;
  try {
    resp = await fetch(path, init);
  } catch (e) {
    const err = new Error('网络错误：' + e.message);
    err.status = 0;
    throw err;
  }

  // 401 = token 缺失/过期/被禁用：默认清状态跳登录页；游客页用 no401 免跳
  if (resp.status === 401 && !opts.no401) {
    clearAuth();
    location.href = '/login.html';
    const err = new Error('登录已过期');
    err.status = 401;
    throw err;
  }

  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.detail || data.error || ('请求失败 ' + resp.status));
    err.status = resp.status;
    err.detail = data.detail || data.error || '';
    throw err;
  }
  return data;
}

/** 基础 HTML 转义（拼模板时用；DOM 构建优先 textContent） */
function escapeHTML(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
