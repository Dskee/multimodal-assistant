# -*- coding: utf-8 -*-
"""
联网搜索模块 WebSearch
无需 API Key：DuckDuckGo 与 Bing **并行**搜索后合并去重
（哪个引擎慢都不拖累整体，最多等最慢的那个）
仅依赖 Python 标准库（urllib），不引入任何第三方依赖
"""

import re
import html as html_mod
import urllib.request
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_FETCH_TIMEOUT = 5      # 单次抓取超时（秒）
_SEARCH_BUDGET = 10     # 整个搜索的硬预算（秒）：到点立即返回已拿到的结果


def _fetch(url, timeout=_FETCH_TIMEOUT):
    """抓取网页文本（UTF-8 优先）"""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="ignore")


def _clean(text):
    """去掉 HTML 标签并反转义实体"""
    text = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(text).strip()


class WebSearch:
    """联网搜索类
    用法：
        results, err = WebSearch().search("今天天气")
        # results: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """

    def search(self, query, top_n=5):
        """搜索入口：DuckDuckGo 与 Bing 并行搜索，结果合并去重
        带硬预算：超过 _SEARCH_BUDGET 秒立即返回已拿到的结果，绝不长时间卡住
        :return: (结果列表, 错误信息)
        """
        def _run_ddg():
            try:
                return self._search_ddg(query)
            except Exception:
                return []

        def _run_bing():
            try:
                return self._search_bing(query)
            except Exception:
                return []

        merged = []
        seen = set()
        ex = ThreadPoolExecutor(max_workers=2)
        try:
            futs = [ex.submit(_run_ddg), ex.submit(_run_bing)]
            deadline = time.time() + _SEARCH_BUDGET
            done = 0
            while done < len(futs):
                remain = deadline - time.time()
                if remain <= 0:
                    break
                try:
                    for fut in as_completed(futs, timeout=remain):
                        done += 1
                        for r in fut.result() or []:
                            key = r.get("url", "").rstrip("/")
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            merged.append(r)
                except Exception:
                    break  # 预算耗尽或异常，停止收集
        finally:
            ex.shutdown(wait=False)  # 不阻塞主流程，残留请求很快自行结束

        if not merged:
            return [], "搜索无结果"
        return merged[:top_n], ""

    # -------------------- DuckDuckGo（lite 轻量版，更快） --------------------

    def _search_ddg(self, query):
        url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
        html_text = _fetch(url)

        # 每条结果是一个 <a rel="nofollow" href="...">标题</a> + 摘要
        # lite 版页面结构：<a href="//duckduckgo.com/l/?uddg=...">标题</a>
        # 摘要为紧跟其后的第二行
        blocks = re.split(r'<a rel="nofollow" href="', html_text)[1:]
        results = []
        for block in blocks:
            m_href = re.match(r'([^"]+)">(.*?)</a>', block, re.S)
            if not m_href:
                continue
            title = _clean(m_href.group(2))
            if not title:
                continue
            url = self._resolve_ddg_url(m_href.group(1))
            # lite 版摘要紧跟在链接块后面（通常是下一段文字）
            m_s = re.search(r'</a>(.{0,400}?)<a href="//duckduckgo.com/l/', block, re.S)
            snippet = _clean(m_s.group(1)) if m_s else ""
            results.append({"title": title, "url": url, "snippet": snippet})
        return results

    @staticmethod
    def _resolve_ddg_url(href):
        """DDG 结果链接是重定向地址，取出真实 URL"""
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            return urllib.parse.unquote(m.group(1))
        if href.startswith("//"):
            return "https:" + href
        return href

    # -------------------- Bing（cn 域名，国内更快） --------------------

    def _search_bing(self, query):
        url = (
            "https://cn.bing.com/search?q=" + urllib.parse.quote(query)
            + "&setlang=zh-hans&mkt=zh-CN"
        )
        html_text = _fetch(url)

        blocks = re.split(r'<li class="b_algo"', html_text)[1:]
        results = []
        for block in blocks:
            m = re.search(
                r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S,
            )
            if not m:
                continue
            title = _clean(m.group(2))
            if not title:
                continue
            p = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
            snippet = _clean(p.group(1)) if p else ""
            results.append({"title": title, "url": m.group(1), "snippet": snippet})
        return results


# ========================= 测试入口 =========================
if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "今天的新闻"
    results, err = WebSearch().search(query)
    if err:
        print("错误:", err)
    else:
        print(f"共 {len(results)} 条结果：")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}")
            print(f"   {r['snippet']}")
            print()
