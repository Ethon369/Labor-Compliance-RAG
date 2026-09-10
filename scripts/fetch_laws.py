"""抓取并解析劳动与社会保障法域的法律/行政法规全文 → data/raw/{law_id}.json（条款级）。

数据源选择、解析策略与已知取舍见 docs/decisions/01-法条数据获取与条款切分.md。
输出结构（入库/检索/引用展示共用同一份）:
    {law_id, name, version, source, articles: [{no, chapter, text}]}

刻意零第三方依赖（urllib + 标准库）：两页 HTML 结构受控（条文全在 <p> 块内），
状态机式解析比引入 bs4 更少依赖、更好讲清正确性；理由见决策记录 01。
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 数据源登记：新增一部法 = 在此加一行（含自校验断言），脚本主体不用改
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LawSource:
    law_id: str
    name: str
    url: str
    expected_articles: int   # 条数硬断言：数量不对说明页面改版或解析漏条，直接失败
    closing_text: str        # 末条收尾语：用于发现"正文被页脚截断/混入噪音"类问题
    amendments: dict[int, str] = field(default_factory=dict)
    # 现行性哨兵：{条号: 修正后措辞}。旧版镜像往往条数与末条收尾语和现行版一字不差，
    # 单查条数会被骗过；修正决定公布的措辞是旧版必然缺失的，抓到旧版立刻红灯。


SOURCES: list[LawSource] = [
    LawSource(
        law_id="labor_law",
        name="中华人民共和国劳动法",
        # 商务部法规平台转载版，实测为 2018-12-29 第二次修正后的现行文本
        # （劳动法经 2009/2018 两次修正，多数镜像页仍停留在旧版，选源时必须按
        #   修正后的关键措辞抽查，见决策记录 01）
        url="https://policy.mofcom.gov.cn/claw/clawContent.shtml?id=66953",
        expected_articles=107,
        closing_text="本法自1995年1月1日起施行",
        # 2018-12-29 修改决定动了 15(2)/69/94 三条措辞
        amendments={
            15: "必须遵守国家有关规定",   # 旧版写"履行审批手续"
            69: "经备案的考核鉴定机构",
            94: "市场监督管理部门",
        },
    ),
    LawSource(
        law_id="labor_contract_law",
        name="中华人民共和国劳动合同法",
        # 新疆人社厅转载版，实测为 2012-12-28 修正后的现行文本
        url="https://rst.xinjiang.gov.cn/xjrst/c113096/202203/"
            "430d729f06c2422ebb606f2ba69125c2.shtml",
        expected_articles=98,
        closing_text="本法自2008年1月1日起施行",
        # 2012-12-28 修改决定动了劳务派遣相关四条（57/63/66/92）
        amendments={
            57: "注册资本不得少于人民币二百万元",
            63: "同工同酬",
            66: "临时性、辅助性或者替代性",
            92: "一倍以上五倍以下",
        },
    ),
    LawSource(
        law_id="labor_dispute_law",
        name="中华人民共和国劳动争议调解仲裁法",
        # 南澳县人社局转载版（HTTP 明文页，实测结构规整、条文各占一段）
        url="http://www.nanao.gov.cn/stnarlsbj/gkmlpt/content/1/1469/post_1469186.html",
        expected_articles=54,
        closing_text="本法自2008年5月1日起施行",
        # 2007 年制定后未修正，无现行性哨兵
    ),
    LawSource(
        law_id="social_insurance_law",
        name="中华人民共和国社会保险法",
        # 北京市政府门户转载版（2019-01 发布，晚于 2018-12-29 修正）
        url="https://www.beijing.gov.cn/zhengce/zhengcefagui/qtwj/201901/"
            "t20190117_780532.html",
        expected_articles=98,
        closing_text="本法自2011年7月1日起施行",
        # 2018-12-29 修改决定动了 57/64/66 三条措辞
        amendments={
            57: "市场监督管理部门",                      # 旧版写"工商行政管理部门"
            64: "除基本医疗保险基金与生育保险基金合并建账及核算外",
            66: "除基本医疗保险基金与生育保险基金预算合并编制外",
        },
    ),
    LawSource(
        law_id="labor_contract_law_reg",
        name="中华人民共和国劳动合同法实施条例",
        # 中国政府网政策库，2008 年公布后未修订
        url="https://www.gov.cn/zhengce/content/2008-09/19/content_6630.htm",
        expected_articles=38,
        closing_text="本条例自公布之日起施行",
    ),
    LawSource(
        law_id="work_injury_reg",
        name="工伤保险条例",
        # 国务院公报页，含 2010-12-20 修订决定 + 修订后全文（现行版）
        url="https://www.gov.cn/gongbao/content/2011/content_1778064.htm",
        expected_articles=67,
        # 末条收尾语取第六十七条的结尾句——该条还含"自2004年1月1日起施行"那句，
        # 但它不是末句，用它收尾会误判
        closing_text="按照本条例的规定执行",
        # 2010 修订动了工伤认定/待遇相关条款，旧版（2003）必然无下列措辞
        amendments={
            14: "非本人主要责任",                        # 旧版写"机动车事故伤害"
            16: "醉酒或者吸毒",                          # 旧版写"醉酒导致伤亡"
            39: "上一年度全国城镇居民人均可支配收入的20倍",
        },
    ),
    LawSource(
        law_id="employment_promotion_law",
        name="中华人民共和国就业促进法",
        # 中国政府网"国情"栏目（法律全文版块），2015-04-24 修正后的现行文本
        url="https://www.gov.cn/guoqing/2021-10/29/content_5647636.htm",
        expected_articles=69,
        closing_text="本法自2008年1月1日起施行",
    ),
    LawSource(
        law_id="unemployment_insurance_reg",
        name="失业保险条例",
        # 福建省政府公报（zfgb）转载国务院令第 258 号
        url="https://zfgb.fujian.gov.cn/1583",
        expected_articles=33,
        # 末条含两句，取末句收尾
        closing_text="《国有企业职工待业保险规定》同时废止",
    ),
    LawSource(
        law_id="public_inst_personnel_reg",
        name="事业单位人事管理条例",
        # 中国政府网政策文件库，国务院令第 652 号
        url="https://www.gov.cn/zhengce/zhengceku/2014-05/15/content_8810.htm",
        expected_articles=44,
        closing_text="本条例自2014年7月1日起施行",
    ),
    LawSource(
        law_id="collective_contract_reg",
        name="集体合同规定",
        # 国务院公报页转载劳动和社会保障部令第 22 号
        url="https://www.gov.cn/gongbao/content/2004/content_62937.htm",
        expected_articles=57,
        # 末条含两句，取末句收尾
        closing_text="《集体合同规定》同时废止",
    ),
    LawSource(
        law_id="labor_supervision_reg",
        name="劳动保障监察条例",
        # 中国政府网政策库，国务院令第 423 号
        url="https://www.gov.cn/zhengce/content/2008-03/28/content_7383.htm",
        expected_articles=36,
        closing_text="本条例自2004年12月1日起施行",
    ),
    LawSource(
        law_id="occupational_disease_law",
        name="中华人民共和国职业病防治法",
        # 北京市政府门户转载版（2017-11-04 第三次修正）。注：2018-12-29 第四次修正
        # 仅调整职业卫生监管机构名称，条文实质未变；故不设现行性哨兵
        url="https://www.beijing.gov.cn/zhengce/zhengcefagui/qtwj/201711/"
            "t20171104_779851.html",
        expected_articles=88,
        closing_text="本法自2002年5月1日起施行",
    ),
    LawSource(
        law_id="female_worker_protection_reg",
        name="女职工劳动保护特别规定",
        # 中国政府网法律法规栏目，国务院令第 619 号。正文后附"禁忌劳动范围"附录，
        # 靠 closing_text 提前终止解析，附录不会并入末条
        url="https://www.gov.cn/flfg/2012-05/07/content_2131582.htm",
        expected_articles=16,
        # 末条含两句，取末句收尾
        closing_text="《女职工劳动保护规定》同时废止",
    ),
]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = ROOT / "data" / "raw"
HTML_DIR = ROOT / "data" / "raw" / "sources"   # 原始页面快照，gitignore

# ---------------------------------------------------------------------------
# 输出模型：scripts 阶段没有 API，但结构校验统一走 Pydantic，入库/检索阶段复用
# ---------------------------------------------------------------------------
class Article(BaseModel):
    no: int = Field(ge=1)
    chapter: str = ""          # 归属章（原样去除空白后的标题，如"第一章总则"）
    text: str = Field(min_length=1)


class SourceMeta(BaseModel):
    primary: str
    cross_checked: list[str] = []
    fetched_at: datetime


class Law(BaseModel):
    law_id: str
    name: str
    version: str = ""          # 页面沿革行（通过/修正时间线），提取失败时为空
    source: SourceMeta
    articles: list[Article]


# ---------------------------------------------------------------------------
# 解析：HTML → 段落流 → 状态机
#
# 为什么不直接在全文上正则抽取：条款可能跨多个 <p>（续款段无条号），页脚导航
# 文本若与末条同段就会出现"条文里混进按钮文字"。按块级标签切出段落流后，逐段
# 判定【章标题/条标题/正文续段/噪音】并归属，每条规则可单测、可审计。
# ---------------------------------------------------------------------------
_BLOCK_TAGS = r"(?:p|div|li|tr|td|th|h[1-6]|br)"
_BLOCK_SPLIT = re.compile(rf"</?{_BLOCK_TAGS}\b[^>]*>", re.IGNORECASE)
_INLINE_TAG = re.compile(r"<[^>]+>")

_CN = "一二三四五六七八九十百千零〇两"
# 条/章标题允许"第/数字/条"之间夹空白或间隔号（如"第 一 条"），避免漏判
_HEAD = rf"^[\s·]*第[\s·]*([{_CN}](?:[\s·]*[{_CN}])*)[\s·]*"
RE_ARTICLE = re.compile(_HEAD + r"条")
RE_CHAPTER = re.compile(_HEAD + r"章")

# 正文终止标记：只可能出现在页面尾部（操作条/分享条/页脚），命中即停止解析，
# 防止页脚被并入末条。必须用"UI 特征短语"而非裸词——法条正文里会出现
# "责令关闭、撤销"这类词（劳动合同法第 44 条），裸词"关闭"会误杀正文。
STOP_MARKERS = re.compile(
    r"^【|\[打印\]|\[关闭\]|^打印$|^关闭$|打印本页|打印页面|关闭本页|关闭页面|"
    r"扫一扫|相关文章|版权所有|主办单位|^主办[:：]|承办单位|^承办[:：]|"
    r"备案号|ICP|公网安备|政府网站标识码|网站地图|无障碍|繁体|英文版|"
    r"政策咨询|12333|友情链接|上一篇|下一篇|修订公告|纠错|字号|"
    r"^本网站|^。。|不承担任何责任|免责声明|技术支持|客服电话|"
    r"网站管理|邮箱[:：]|投稿|^顶部$"
)
# 正文区域内出现即丢弃的段：正文起点之前的导航/表格碎片（不中断解析）
SKIP_PARAS = re.compile(r"^(?:首页|当前位置|您现在的位置|您的位置)")
_CJK = re.compile(r"[一-鿿]")

_VERSION_RE = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日.*?(?:通过|修正|施行)")


def chinese_to_int(s: str) -> int:
    """中文数字转 int，容忍内部空白/间隔号。仅支持到"百"（法律条数上限内）。"""
    digits = {c: i for i, c in enumerate("一二三四五六七八九", start=1)}
    digits["两"] = 2
    total = cur = 0
    for ch in re.sub(r"[\s·]", "", s):
        if ch in digits:
            cur = digits[ch]
        elif ch == "百":            # 进位：cur 为空说明是"一百"这类省略一
            total += (cur or 1) * 100
            cur = 0
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch in "零〇":
            cur = 0
        else:
            raise ValueError(f"无法识别的中文数字: {s!r}")
    return total + cur


def decode_html(raw: bytes) -> str:
    """按页面声明的 charset 解码；声明缺失或错误时依次回退 utf-8 → gb18030。"""
    meta = re.search(rb'charset=["\']?\s*([\w-]+)', raw[:4096], re.I)
    candidates = []
    if meta:
        candidates.append(meta.group(1).decode("ascii", "replace"))
    candidates += ["utf-8", "gb18030"]
    last_err: Exception | None = None
    for cs in candidates:
        try:
            return raw.decode(cs)
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"无法按 {candidates} 解码页面: {last_err}")


def split_paragraphs(doc: str) -> list[str]:
    """按块级标签切成段落流；块内残留的内联标签剥掉，空白归一。"""
    doc = re.sub(r"<script.*?</script>", "", doc, flags=re.S | re.I)
    doc = re.sub(r"<style.*?</style>", "", doc, flags=re.S | re.I)
    # 注释里会藏 UI 按钮文字（如商务部页"修订对照/下载"），法条正文不可能在注释里
    doc = re.sub(r"<!--.*?-->", "", doc, flags=re.S)
    paras: list[str] = []
    for chunk in _BLOCK_SPLIT.split(doc):
        text = html.unescape(_INLINE_TAG.sub("", chunk))
        text = text.replace("　", " ").replace("\xa0", " ").strip()
        text = re.sub(r"[ \t\r\n]+", " ", text)
        if text:
            paras.append(text)
    return paras


def _clean_heading_num(num: str) -> str:
    return re.sub(r"[\s·]", "", num)


def extract_version(paras: list[str]) -> str:
    """正文起点前扫描"通过/修正"沿革行（含两处以上日期的那句，如商务部页）。

    沿革行是判断页面文本是否为现行版的第一手证据，抓回来写进 JSON 元数据。
    """
    best = ""
    for p in paras[:200]:
        if re.search(r"\d{4}年.*\d{4}年", p) and _VERSION_RE.search(p):
            best = p
    # 去掉行首行尾的括号包裹，只留时间线正文
    return best.strip("（）()")


def check_current(law: Law, src: LawSource) -> None:
    """现行性硬断言：src.amendments 登记的修正后措辞必须逐条出现在对应条款文本里。

    为什么要有这一步：旧版镜像页的"条数/末条收尾语"常与现行版完全相同（决策记录 01
    实测沈阳人社整页无一处 2018 措辞但结构照旧），只看条数与末条拦不住回落旧版。
    修正决定公布的措辞是旧版必然缺的，据此逐条断言，页面换旧版即抛错。
    """
    if not src.amendments:
        return
    by_no = {a.no: a.text for a in law.articles}
    for no, phrase in src.amendments.items():
        if phrase not in by_no.get(no, ""):
            raise ValueError(
                f"{src.law_id}: 第{no}条未含现行（修正后）措辞 {phrase!r}"
                "——页面可能是修正前的旧版镜像"
            )


def parse_document(doc: str, src: LawSource) -> Law:
    """整页 → Law。核心不变量：articles 条号连续且等于 1..N（漏条即失败）。"""
    paras = split_paragraphs(doc)
    articles: list[Article] = []
    in_body = False                 # 见到第一个条标题后正文才算开始
    cur_no: int | None = None       # 正在累积的条号；None = 条间空隙
    cur_parts: list[str] = []
    cur_chapter = ""                # 最近一次正文"第X章"，赋给后续条文

    def commit() -> None:
        if cur_no is not None:
            articles.append(Article(no=cur_no, chapter=cur_chapter,
                                    text="\n".join(cur_parts)))

    for i, p in enumerate(paras):
        m = RE_ARTICLE.match(p)
        if m:
            commit()                      # 收尾上一条，开始新条
            in_body = True
            cur_no = chinese_to_int(_clean_heading_num(m.group(1)))
            cur_parts = []
            rest = p[m.end():].strip(" ：")
            if rest:
                cur_parts.append(rest)    # 多数页面条号与正文同段
            continue
        m = RE_CHAPTER.match(p)
        # 章标题出现在两条路径：正文中（in_body，直接切换章节）或正文最开头
        # （"第一章总则"紧跟第一条，此时还在 in_body=False 的目录区）。
        # 统一判据：章标题后面紧跟着条标题才算正文章——目录里的"第X章"后面
        # 跟的是下一个章标题，天然被排除，无需记住"目录看到第几章"这类状态。
        next_is_article = i + 1 < len(paras) and RE_ARTICLE.match(paras[i + 1])
        if m and (in_body or next_is_article):
            commit()
            cur_no = None
            cur_parts = []
            cur_chapter = re.sub(r"[\s·]", "", p)   # 去空白统一格式，便于比对
            continue
        if not in_body:
            continue                      # 导航/标题/目录/元数据区，不参与正文
        if STOP_MARKERS.search(p):
            break                         # 页脚开始，正文到此为止
        if SKIP_PARAS.match(p) or not _CJK.search(p):
            continue
        # 收尾语出现即正文结束：其后是站点页脚（导航栏/分享条/推荐位），一律不再并入。
        # 比穷举页脚停止词可靠——不同站点的页脚首段差异大（gov.cn 是"全国人大"这类
        # 机构导航，加进 STOP_MARKERS 会误伤正文里的"全国人民代表大会常务委员会"）。
        if src.closing_text and src.closing_text in "".join(cur_parts):
            break
        cur_parts.append(p)               # 无条号的续款段并入当前条

    commit()
    if not articles:
        raise ValueError(f"{src.law_id}: 未解析出任何条款，页面结构可能已改版")
    # 断言连续性 = 漏条检测（正则/页面改版时第一时间暴露）
    got = [a.no for a in articles]
    if got != list(range(1, len(got) + 1)):
        raise ValueError(f"{src.law_id}: 条号不连续: {got[:5]}...")
    if len(articles) != src.expected_articles:
        raise ValueError(
            f"{src.law_id}: 期望 {src.expected_articles} 条，实际 {len(articles)} 条"
        )
    # 末条必须"恰好"以收尾语结尾：任何页脚混入都会破坏 endswith，比 contains 更严
    if not articles[-1].text.endswith(src.closing_text + "。"):
        raise ValueError(
            f"{src.law_id}: 末条不以收尾语结尾（got ...{articles[-1].text[-60:]!r}），"
            "正文可能被页脚污染或被截断"
        )
    law = Law(
        law_id=src.law_id,
        name=src.name,
        version=extract_version(paras),
        source=SourceMeta(primary=src.url, fetched_at=datetime.now(timezone.utc)),
        articles=articles,
    )
    check_current(law, src)   # 现行性防线：页面回落旧版在此当场红灯，不落到 JSON
    return law


# ---------------------------------------------------------------------------
# 抓取与落盘
# ---------------------------------------------------------------------------
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def fetch_html(src: LawSource, timeout: int = 30) -> bytes:
    req = Request(src.url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"下载失败 {src.url}: {e}") from e


def _dump(law: Law, outdir: Path) -> Path:
    out = outdir / f"{law.law_id}.json"
    out.write_text(law.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out


def run_law(src: LawSource, outdir: Path, keep_html: bool) -> Law:
    raw = fetch_html(src)
    if keep_html:
        HTML_DIR.mkdir(parents=True, exist_ok=True)
        (HTML_DIR / f"{src.law_id}.html").write_bytes(raw)
    law = parse_document(decode_html(raw), src)
    path = _dump(law, outdir)
    last = law.articles[-1]
    print(f"[ok] {law.name}：{len(law.articles)} 条 → {path}")
    print(f"     version: {law.version[:80] or '(未提取到)'}")
    print(f"     末条: 第{last.no}条 {last.text[:50]}…")
    return law


def check_json(path: Path) -> None:
    """离线复查：已有 JSON 再按相同不变量校验一遍（CI/定时可跑）。

    与 parse_document 共用同一组断言：条号连续、条数、末条收尾语、现行性措辞。
    这样"抓取当场拦"与"--check 事后拦"走的是同一条防线，不会出现只防一手。
    """
    law = Law.model_validate_json(path.read_text(encoding="utf-8"))
    src = next(s for s in SOURCES if s.law_id == law.law_id)
    got = [a.no for a in law.articles]
    if got != list(range(1, len(got) + 1)):
        raise ValueError(f"{path}: 条号不连续")
    if len(got) != src.expected_articles or src.closing_text not in law.articles[-1].text:
        raise ValueError(f"{path}: 与源登记的自校验断言不符")
    check_current(law, src)
    print(f"[ok] {path} 校验通过：{len(got)} 条，末条含收尾语，"
          f"现行性措辞 {len(src.amendments)} 处")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--law", choices=["all", *[s.law_id for s in SOURCES]],
                    default="all", help="只抓某部法，默认全部")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help=f"JSON 输出目录（默认 {DEFAULT_OUTDIR}）")
    ap.add_argument("--check", type=Path, nargs="*", default=None,
                    help="离线校验 JSON；不带路径默认校验 data/raw/*.json（不联网）")
    ap.add_argument("--keep-html", action="store_true",
                    help="原始页面存到 data/raw/sources/ 供审计")
    args = ap.parse_args(argv)

    if args.check is not None:          # None=未传旗标；空 [] = 传了不带路径 => 校验全部产物
        files = args.check or sorted(DEFAULT_OUTDIR.glob("*.json"))
        if not files:
            print(f"未找到 {DEFAULT_OUTDIR}/ 下的 JSON", file=sys.stderr)
            return 1
        try:
            for p in files:
                check_json(p)
        except Exception as e:
            print(f"[fail] {e}", file=sys.stderr)
            return 1
        return 0
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    targets = SOURCES if args.law == "all" else [s for s in SOURCES if s.law_id == args.law]
    try:
        for src in targets:
            run_law(src, outdir, args.keep_html)
    except Exception as e:                # 任何一条失败整体非零退出，便于人工发现
        print(f"[fail] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
