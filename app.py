"""Quiz Factory 网页版：填一句话 → 生成 → 预览 / 下载单文件 HTML。

运行：..\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import model  # noqa: E402
import presets  # noqa: E402
import render  # noqa: E402
from llm import load_env_keys  # noqa: E402

DIST = ROOT / "docs"
QUIZZES = ROOT / "quizzes"

st.set_page_config(page_title="Quiz Factory · 一句话生成测试", page_icon="🎲", layout="wide")

st.markdown(
    """
    <style>
      .block-container{padding-top:2rem}
      .preview{color:#7a7268;font-size:14px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎲 Quiz Factory")
st.caption("一句话生成一个可以分享的人格测试：8 道题 · 2 分钟 · 8 种结果 · 单文件网页")

tab_gallery, tab_generate, tab_about = st.tabs(["🖼 测试库", "✨ 生成新测试", "🧭 怎么用"])

with tab_gallery:
    main_dir = DIST if (DIST / "index.html").exists() else None
    if main_dir:
        st.markdown(f"[在浏览器中打开测试库 →]({main_dir / 'index.html'})")
        st.caption("提示：点开后在浏览器里体验，链接可分享；每个测试都是独立 HTML。")
    manifest = DIST / "quizzes.json"
    items = json.loads(manifest.read_text(encoding="utf-8"))["quizzes"] if manifest.exists() else []
    if not items:
        st.info("还没有生成过测试，去「✨ 生成新测试」做一个，或先渲染内置预设。")
    for item in items:
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"## {item['emoji']}")
            with col2:
                st.markdown(f"**{item['title']}**")
                st.caption(item["subtitle"])
                path = DIST / f"{item['slug']}.html"
                st.download_button(
                    "下载这个测试（HTML）",
                    path.read_bytes() if path.exists() else b"",
                    file_name=f"{item['slug']}.html",
                    mime="text/html",
                    key=f"dl-{item['slug']}",
                )

with tab_generate:
    st.subheader("生成一个新测试")
    theme = st.text_input("主题（一句话）", placeholder="测测你是哪种打工人 / 测测你是哪种 AI 产品经理")

    provider = st.radio("模型来源", ["本地 Ollama（免费）", "云端 API（更快更准）"],
                        horizontal=True)
    provider_key = "ollama" if provider.startswith("本地") else "openai"
    env = load_env_keys()
    col1, col2, col3 = st.columns(3)
    if provider_key == "ollama":
        model_name = col1.text_input("本地模型", value="qwen2.5:7b")
        api_key = ""
        base_url = ""
        col2.caption("生成约 1-3 分钟（CPU）")
    else:
        model_name = col1.text_input("云端模型", value="deepseek-chat")
        api_key = col2.text_input("API Key", type="password",
                                  value=env.get("QUIZ_API_KEY") or env.get("LIT_API_KEY", ""))
        base_url = col3.text_input("接口地址", value=env.get("QUIZ_API_BASE") or env.get("LIT_API_BASE")
                                   or "https://api.deepseek.com/v1")
        col1.caption("Key 只存在当前页面会话，不会写入文件")
    count = st.slider("题目数量", 6, 10, 8)
    slug = st.text_input("文件名（英文，可留空）", placeholder="worker")
    st.caption("生成是三段式的（类型名 → 题目 → 文案），每段都有质量门禁，不合格自动带反馈重试。")

    if st.button("开始生成", type="primary", disabled=not theme):
        try:
            from generate import build_gallery, generate as generate_quiz  # 延迟导入
            status = st.status("正在生成...", expanded=True)
            quiz, report = generate_quiz(
                theme, slug or model._slug(theme), provider=provider_key,
                model_name=model_name, base_url=base_url, api_key=api_key, count=count,
                progress=lambda stage, msg: status.write(f"[{stage}] {msg}"))
            model.save(quiz, QUIZZES / f"{quiz['slug']}.json")
            (QUIZZES / f"{quiz['slug']}.report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            build_gallery()
            status.update(label=f"生成完成：{quiz['title']}（{report['seconds']}s）",
                          state="complete", expanded=False)

            if report["first_pass"]:
                st.success("🎉 三段全部一次通过质量门禁，可以直接发布")
            elif report["needs_polish"]:
                st.warning("生成完成，但仍有以下问题，建议人工润色后再发布：")
                for problem in report["final_problems"][:6]:
                    st.markdown(f"- {problem}")
            else:
                st.info("生成完成：部分段落经过自动重试后通过门禁，可用。")
            for stage in report["stages"]:
                tries = stage["tries"]
                icon = "✅" if stage["ok"] else "❌"
                st.markdown(f"{icon} **{stage['stage']}**：{tries} 次尝试"
                            + ("" if stage["ok"] else "（未通过，建议人工润色）"))
            st.markdown("**结果分布**（穷举校准后）：" + "  ".join(
                f"`{k}={v}%`" for k, v in sorted(report.get("distribution", {}).items())))
            st.page_link(str(DIST / f"{quiz['slug']}.html"), label="在浏览器里试玩 →")
        except SystemExit as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"生成失败：{type(exc).__name__}: {exc}")

    st.divider()
    st.subheader("或者直接渲染内置预设（不需要模型）")
    for path in presets.preset_files():
        quiz = model.load(path)
        st.write(f"- {quiz['emoji']} **{quiz['title']}**（{path.stem}）")

with tab_about:
    st.subheader("这个产品解决什么问题")
    st.markdown(
        """
        做「测测你是哪种 X」这类互动内容，传统方式是找模板站（同质化、要付费、改不动），
        或者自己写（设计题目 + 想 8 种人格 + 写文案 + 做页面，半天起步）。

        这里把整条链路压缩成一句话：

        1. **模型生成**：3 条二元轴 × 8 道生活场景题 × 8 种结果文案（JSON）；
        2. **确定性打分**：每个选项只给 1-2 条轴 ±1，结果由代码算，不靠模型"猜"；
        3. **单文件渲染**：题目、结果、分享、深链（`#r=000`）全部内联，零后端、零成本分发；
        4. **稀有度**：穷举所有作答组合，算出每种结果的比例，结果页显示"隐藏款/稀有/常见"。

        适合：自媒体做互动内容、运营做活动页、求职者做作品集小产品、老师做课堂破冰。
        """
    )
    st.warning("注意：生成的是娱乐性人格测试，不是心理测量工具；文案需人工过一遍再发布。")
