"""fetch_laws 纯函数与页面解析测试（全部离线，fixture 在 tests/fixtures/）。"""
from __future__ import annotations

from pathlib import Path

import pytest

import fetch_laws as fl

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
REAL_LABOR = FIXTURES / "pages" / "labor_law_mofcom.html"
REAL_CONTRACT = FIXTURES / "pages" / "labor_contract_xinjiang.html"
SYNTHETIC = FIXTURES / "synthetic_gb18030.html"

LABOR = next(s for s in fl.SOURCES if s.law_id == "labor_law")
CONTRACT = next(s for s in fl.SOURCES if s.law_id == "labor_contract_law")


# ---------------------------------------------------------------------------
# 中文数字转换
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s,expected", [
    ("一", 1), ("十", 10), ("十三", 13), ("二十", 20), ("九十八", 98),
    ("一百", 100), ("一百零七", 107), ("二百", 200), ("两", 2),
    ("十一", 11), ("三十三", 33),
])
def test_chinese_to_int(s: str, expected: int) -> None:
    assert fl.chinese_to_int(s) == expected


def test_chinese_to_int_rejects_unsupported():
    with pytest.raises(ValueError):
        fl.chinese_to_int("一千")          # 只支持到"百"，超出直接报错而非算错


# ---------------------------------------------------------------------------
# 解码与段落切分
# ---------------------------------------------------------------------------
def test_decode_html_gb18030_page():
    """gb2312 声明的页面能按声明解码，不依赖默认 utf-8。"""
    raw = SYNTHETIC.read_bytes()
    doc = fl.decode_html(raw)
    assert "中华人民共和国测试法" in doc


def test_decode_html_fallback_and_failure():
    # 无 charset 声明的 utf-8 内容回退成功
    assert "fallback" in fl.decode_html("中文<div>fallback</div>".encode("utf-8"))
    # 声明错误 + 两种回退都失败的字节串应抛错
    with pytest.raises(ValueError):
        fl.decode_html(b"\xff\xfe\x00invalid\xff")


def test_split_paragraphs_strips_comments_and_scripts():
    doc = ("<p>正文A</p><!-- 修订对照/下载 --><p>正文B</p>"
           "<script>var x='第一条';</script><p>正文<b>C</b></p>")
    paras = fl.split_paragraphs(doc)
    assert paras == ["正文A", "正文B", "正文C"]   # 注释/脚本剥离、内联标签不留换行


def test_split_paragraphs_block_and_blank():
    doc = "<div>块一</div><br/><li>列表</li><td>单元格</td><p>&nbsp;</p><p>末段</p>"
    paras = fl.split_paragraphs(doc)
    assert paras == ["块一", "列表", "单元格", "末段"]   # 空段落剔除


# ---------------------------------------------------------------------------
# 真实页面快照端到端解析
# ---------------------------------------------------------------------------
def _parse(fixture: Path, src: fl.LawSource) -> fl.Law:
    return fl.parse_document(fl.decode_html(fixture.read_bytes()), src)


def test_labor_law_full_parse():
    law = _parse(REAL_LABOR, LABOR)
    assert len(law.articles) == 107
    a1, a2, a107 = law.articles[0], law.articles[1], law.articles[-1]
    assert a1.chapter == "第一章总则"                    # 正文首章不再被当目录丢掉
    assert a1.text.startswith("为了保护劳动者的合法权益")
    assert "国家机关、事业组织、社会团体" in a2.text    # 第二款跨独立 <p>，已并入第 2 条
    assert a107.chapter == "第十三章附则"
    assert a107.text == "本法自1995年1月1日起施行。"     # 恰好收尾，无页脚混入


@pytest.mark.parametrize("fixture,src,version_marker", [
    (REAL_LABOR, LABOR, "2018年12月29日"),   # 2018 修正动了 15(2)/69/94
    (REAL_CONTRACT, CONTRACT, "2012年12月28日"),  # 2012 修正动了劳务派遣 57/63/66/92
])
def test_snapshot_passes_current_amendment_guard(
    fixture: Path, src: fl.LawSource, version_marker: str
):
    """现行性防线（数据驱动）：哨兵只在 SOURCES 登记一处，抓取路径与测试共用。

    镜像页若回落旧文本，parse_document 末尾的 check_current 会先抛错——这里再显式
    核对一遍登记本身没写错（措辞确实存在于现行版对应条款）。
    """
    law = _parse(fixture, src)
    texts = {a.no: a.text for a in law.articles}
    for no, phrase in src.amendments.items():
        assert phrase in texts[no], f"{src.law_id} 第{no}条缺修正后措辞 {phrase!r}"
    assert version_marker in law.version          # 版本沿革行同样必须是现行版


def test_labor_contract_law_full_parse():
    law = _parse(REAL_CONTRACT, CONTRACT)
    assert len(law.articles) == 98
    a1, a98 = law.articles[0], law.articles[-1]
    assert a1.chapter == "第一章总则"
    assert a1.text.startswith("为了完善劳动合同制度")
    assert a98.text == "本法自2008年1月1日起施行。"
    chapters = {a.chapter for a in law.articles}
    assert chapters == {"第一章总则", "第二章劳动合同的订立", "第三章劳动合同的履行和变更",
                        "第四章劳动合同的解除和终止", "第五章特别规定", "第六章监督检查",
                        "第七章法律责任", "第八章附则"}


def _with_old_94(law: fl.Law) -> fl.Law:
    """返回把第 94 条换回旧版措辞的副本（旧版写"工商行政管理机关"）。

    多个反例测试共用：验证现行性防线在 parse / check_json / main 三处都会拦。"""
    return fl.Law(
        law_id=law.law_id, name=law.name, version=law.version, source=law.source,
        articles=[
            a if a.no != 94 else fl.Article(no=94, chapter=a.chapter,
                                            text=a.text.replace("市场监督管理部门", "工商行政管理机关"))
            for a in law.articles
        ],
    )


def test_current_guard_rejects_old_wording():
    """反例：现行文本某条改成旧版措辞，check_current 必须拦住，否则防线是纸糊的。"""
    law = _parse(REAL_LABOR, LABOR)
    assert law.articles[93].no == 94
    with pytest.raises(ValueError, match="第94条"):
        fl.check_current(_with_old_94(law), LABOR)


def test_check_json_rejects_old_wording_file(tmp_path):
    """--check 与抓取共用防线：改旧版的 JSON 在 check_json 也会红灯，不是只在 parse 时防。"""
    p = tmp_path / "stale_labor_law.json"
    p.write_text(_with_old_94(_parse(REAL_LABOR, LABOR)).model_dump_json(indent=2),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="第94条"):
        fl.check_json(p)


def test_main_check_defaults_to_committed_data():
    """裸 `--check`（不带路径）应默认扫 data/raw/*.json 且 rc=0——README 演示命令真能跑。"""
    assert fl.main(["--check"]) == 0


def test_main_check_reports_stale_file_rc1(tmp_path, capsys):
    """main() 层面：遇到旧版文件要整体非零退出，而不是抛裸异常或默默 rc=0。"""
    p = tmp_path / "stale_labor_law.json"
    p.write_text(_with_old_94(_parse(REAL_LABOR, LABOR)).model_dump_json(indent=2),
                 encoding="utf-8")
    assert fl.main(["--check", str(p)]) == 1
    assert "第94条" in capsys.readouterr().err


@pytest.mark.parametrize("law,src", [(REAL_LABOR, LABOR), (REAL_CONTRACT, CONTRACT)])
def test_article_text_hygiene(law: Path, src: fl.LawSource):
    """全体条文卫生检查：标题残留/页脚按钮/缩进/空白格式一个都不许进 text。"""
    parsed = _parse(law, src)
    for a in parsed.articles:
        assert a.text == a.text.strip()
        assert not a.text.startswith("第")            # 条号已被剥掉，不在文本里
        assert "【" not in a.text and "】" not in a.text
        assert "　" not in a.text                 # 全角缩进已归一
        assert a.no == int(a.no)                      # 顺带确认类型


# ---------------------------------------------------------------------------
# 合成页：真实页抓不到的刁钻 case
# ---------------------------------------------------------------------------
SYNTH_SRC = fl.LawSource(
    law_id="synthetic", name="中华人民共和国测试法",
    url="https://example.invalid/synthetic", expected_articles=3,
    closing_text="本法自2000年1月1日起施行",
)


def test_synthetic_page_tricks():
    """覆盖：gb18030 解码、双份目录、条号夹空格、无空格条号、续款跨段、
    正文含"责令关闭"不被当页脚、页脚噪音不进末条。"""
    law = _parse(SYNTHETIC, SYNTH_SRC)
    assert [a.no for a in law.articles] == [1, 2, 3]
    a1, a2, a3 = law.articles
    assert a1.chapter == "第一章总则"                  # 双份目录里的"第一章"被忽略
    assert a1.text.startswith("为了测试而制定本法")     # "第 一 条"夹空格也能认
    assert a2.chapter == "第一章总则"
    assert "责令关闭、撤销" in a2.text                 # 正文的"关闭"不会误触发页脚判定
    assert "\n国家机关、事业单位参照执行。" in a2.text  # 跨段续款并入第 2 条
    assert a3.chapter == "第二章附则"
    assert a3.text == "本法自2000年1月1日起施行。"      # 无空格条号 + 页脚在 STOP 之后
