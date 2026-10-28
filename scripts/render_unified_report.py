#!/usr/bin/env python3
"""Unified Report Generator for Labs 02, 03, and 04.

Generates a standalone, full-bleed, edge-to-edge HTML report (LABS_02_03_04_REPORT.html)
and a synchronized technical Markdown report (LABS_02_03_04_REPORT.md).
Retains 100% of all content, figures, code snippets, takeaways, tables, and flow maps
from Lab 02, Lab 03, and Lab 04 without truncation or floating rounded-corner containment.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
LABS_DIR = DOCS_DIR / "labs"


def get_base64_img(rel_path: str) -> str:
    """Read an image or SVG from docs/assets and return a data URI."""
    target = ASSETS_DIR / rel_path
    if not target.exists():
        matches = list(ASSETS_DIR.rglob(Path(rel_path).name))
        if matches:
            target = matches[0]
        else:
            raise FileNotFoundError(f"Asset image not found: {rel_path} in {ASSETS_DIR}")
    mime = "image/png"
    suffix = target.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    b64 = base64.b64encode(target.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# Edge-to-Edge Full Bleed Modern CSS (Zero border-radius, 100% width)
UNIFIED_CSS = """
/* Universal Reset & Full Bleed Layout (Zero Border-Radius) */
*, *::before, *::after {
  box-sizing: border-box;
  border-radius: 0 !important;
}

:root {
  --primary-2: #1e3a8a;
  --primary-3: #183f35;
  --primary-4: #4c1d95;
  --bg-page: #f8fafc;
  --bg-card: #ffffff;
  --border: #cbd5e1;
  --border-dark: #334155;
  --text-main: #0f172a;
  --text-muted: #475569;
  --success: #15803d;
  --success-bg: #f0fdf4;
}

html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
  background-color: var(--bg-page);
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.65;
  scroll-behavior: smooth;
  scroll-padding-top: 110px;
}

main {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Master Top Sticky Navigation Bar */
.master-topbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  width: 100%;
  background: #091310;
  border-bottom: 2px solid #22c55e;
  padding: 10px clamp(16px, 3vw, 40px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  box-shadow: 0 4px 12px rgba(0,0,0,0.18);
}

.master-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #f8fafc;
  font-weight: 700;
  font-size: 15px;
  text-decoration: none;
}

.master-brand .tag {
  background: #15803d;
  color: #fff;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.master-nav-links {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.master-nav-links a {
  color: #cbd5e1;
  text-decoration: none;
  font-size: 13px;
  font-weight: 600;
  padding: 7px 13px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(255, 255, 255, 0.05);
  transition: all 0.15s ease;
}

.master-nav-links a:hover, .master-nav-links a.active {
  background: #22c55e;
  color: #091310;
  border-color: #22c55e;
}

.master-nav-links a.btn-lab2:hover { background: #38bdf8; color: #02122b; border-color: #38bdf8; }
.master-nav-links a.btn-lab3:hover { background: #4ade80; color: #052e16; border-color: #4ade80; }
.master-nav-links a.btn-lab4:hover { background: #c084fc; color: #2e1065; border-color: #c084fc; }
.master-nav-links a.btn-matrix:hover { background: #fbbf24; color: #451a03; border-color: #fbbf24; }

.master-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.master-actions button {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.25);
  color: #f1f5f9;
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}

.master-actions button:hover {
  background: #f8fafc;
  color: #0f172a;
}

/* Master Executive Banner */
.master-hero {
  width: 100%;
  background: linear-gradient(135deg, #091a13 0%, #123225 55%, #1b4b38 100%);
  color: #ffffff;
  padding: clamp(32px, 5vw, 64px) clamp(20px, 4vw, 56px);
  border-bottom: 3px solid #22c55e;
}

.master-hero .eyebrow {
  color: #86efac;
  font-size: 13px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 12px;
}

.master-hero h1 {
  font-size: clamp(28px, 4vw, 48px);
  line-height: 1.2;
  margin: 0 0 16px 0;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.master-hero p.lead {
  font-size: clamp(16px, 1.8vw, 20px);
  max-width: 1200px;
  color: #d1fae5;
  margin: 0 0 24px 0;
  line-height: 1.5;
}

.master-hero .kpi-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-top: 24px;
}

.master-hero .kpi-card {
  background: rgba(0, 0, 0, 0.35);
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-left: 4px solid #22c55e;
  padding: 16px 20px;
}

.master-hero .kpi-card .kpi-val {
  font-size: 26px;
  font-weight: 800;
  color: #4ade80;
  line-height: 1.2;
}

.master-hero .kpi-card .kpi-title {
  font-size: 14px;
  font-weight: 700;
  color: #ffffff;
  margin: 4px 0 2px 0;
}

.master-hero .kpi-card .kpi-sub {
  font-size: 12px;
  color: #cbd5e1;
}

/* Lab Section Full-Width Containers */
.lab-section {
  width: 100%;
  margin: 0;
  padding: 0;
  border-bottom: 8px solid #0f172a;
}

.lab-section-header {
  width: 100%;
  padding: clamp(28px, 4vw, 48px) clamp(20px, 4vw, 56px);
  color: #ffffff;
  border-bottom: 2px solid rgba(255,255,255,0.15);
}

.lab-section-header.lab02-header {
  background: linear-gradient(135deg, #0c1e47 0%, #1e3a8a 70%, #0369a1 100%);
  border-bottom-color: #38bdf8;
}

.lab-section-header.lab03-header {
  background: linear-gradient(135deg, #0b221a 0%, #183f35 70%, #15803d 100%);
  border-bottom-color: #4ade80;
}

.lab-section-header.lab04-header {
  background: linear-gradient(135deg, #240e4f 0%, #4c1d95 70%, #4338ca 100%);
  border-bottom-color: #c084fc;
}

.lab-section-header .eyebrow {
  font-size: 12px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-weight: 700;
  opacity: 0.85;
  margin-bottom: 8px;
}

.lab-section-header h2 {
  font-size: clamp(24px, 3.2vw, 40px);
  margin: 0 0 10px 0;
  font-weight: 800;
}

.lab-section-header p {
  font-size: 16px;
  max-width: 1000px;
  margin: 0 0 16px 0;
  opacity: 0.92;
}

/* Sub-nav for each lab */
nav.lab-subnav {
  position: sticky;
  top: 50px;
  z-index: 900;
  width: 100%;
  background: #ffffff;
  border-bottom: 1px solid var(--border);
  padding: 10px clamp(20px, 4vw, 56px);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  box-shadow: 0 2px 4px rgba(0,0,0,0.04);
}

nav.lab-subnav a {
  font-size: 13px;
  color: #334155;
  text-decoration: none;
  padding: 5px 11px;
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  font-weight: 600;
  transition: all 0.1s ease;
}

nav.lab-subnav a:hover, nav.lab-subnav a[aria-current="location"] {
  background: #0f172a;
  color: #ffffff;
  border-color: #0f172a;
}

nav.lab-subnav button {
  margin-left: auto;
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  padding: 5px 12px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

/* Flow Map Container */
.flow-map {
  width: 100%;
  background: #ffffff;
  border-bottom: 1px solid var(--border);
  padding: 28px clamp(20px, 4vw, 56px);
  margin-bottom: 0;
}

.flow-map .eyebrow {
  font-size: 11px;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: #678273;
  font-weight: 700;
}

.flow-map h2 {
  font-size: 24px;
  margin: 6px 0 12px 0;
  color: var(--text-main);
}

.flow-map p {
  color: var(--text-muted);
  max-width: 1000px;
  margin-bottom: 20px;
}

.map-scroll {
  overflow-x: auto;
  width: 100%;
  padding-bottom: 8px;
}

.map-svg, .experiment-map {
  display: block;
  width: 100%;
  min-width: 800px;
  max-width: 1200px;
  margin: 0 auto;
}

/* Content Stages Full Bleed */
.lab-body {
  width: 100%;
  padding: 32px clamp(20px, 4vw, 56px);
  background: var(--bg-page);
}

.stage, .flow-stage {
  border-left: 3px solid #94a3b8;
  margin: 0 0 48px 20px;
  padding: 0 0 32px 30px;
}

.stage-head, .stage-heading {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.number, .stage-number {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  margin-left: -53px;
  background: #0f172a;
  color: #ffffff;
  border: 3px solid var(--bg-page);
  font-weight: 700;
  font-size: 15px;
}

.stage h2, .flow-stage h2 {
  font-size: 24px;
  line-height: 1.3;
  margin: 4px 0;
  color: var(--text-main);
}

.mini-flow {
  padding: 12px 18px;
  background: #e2e8f0;
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
  border-left: 4px solid #0284c7;
  margin: 12px 0 18px 0;
}

.intro, .stage-intro {
  color: var(--text-muted);
  font-size: 15px;
  margin: 12px 0 20px 0;
  max-width: 100%;
}

/* Image Cards & Gallery */
.image-card, .gallery-item {
  background: #ffffff;
  border: 1px solid var(--border);
  overflow: hidden;
  margin: 20px 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.image-card img, .gallery-item img {
  display: block;
  max-width: 100%;
  width: 100%;
  max-height: 900px;
  object-fit: contain;
  cursor: zoom-in;
  background: #ffffff;
}

.image-card figcaption, .gallery-item .caption {
  padding: 14px 20px;
  font-size: 14px;
  color: #334155;
  border-top: 1px solid #e2e8f0;
  background: #f8fafc;
}

.image-card figcaption b, .gallery-item .caption strong {
  display: block;
  color: var(--text-main);
  font-size: 15px;
  margin-bottom: 4px;
}

.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 20px;
  margin-top: 16px;
}

.takeaway, .callout {
  padding: 16px 20px;
  background: #f0fdf4;
  border-left: 4px solid #16a34a;
  font-size: 14px;
  color: #14532d;
  margin: 20px 0;
}

.callout.warn {
  background: #fffbeb;
  border-left-color: #f59e0b;
  color: #78350f;
}

details {
  background: #ffffff;
  border: 1px solid var(--border);
  margin: 16px 0;
}

summary {
  padding: 14px 20px;
  cursor: pointer;
  font-weight: 700;
  color: #0f172a;
  background: #f8fafc;
  border-bottom: 1px solid transparent;
  user-select: none;
}

details[open] summary {
  border-bottom-color: var(--border);
}

.detail-body {
  padding: 20px;
  overflow-wrap: anywhere;
}

.code-note {
  font-size: 13px;
  color: #64748b;
  margin: 8px 0;
}

pre {
  padding: 16px 20px;
  background: #0f172a;
  color: #e2e8f0;
  overflow-x: auto;
  line-height: 1.6;
  font-size: 13px;
  font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace;
  margin: 12px 0;
}

code {
  font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace;
  background: #e2e8f0;
  color: #0f172a;
  padding: 2px 5px;
  font-size: 0.9em;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

.support {
  margin: 48px 0 24px 0;
  padding-top: 24px;
  border-top: 2px solid var(--border);
}

.support h2 {
  font-size: 26px;
  margin-bottom: 16px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
  font-size: 14px;
}

th, td {
  padding: 10px 14px;
  border: 1px solid var(--border);
  text-align: left;
}

th {
  background: #f1f5f9;
  font-weight: 700;
}

/* Master Comparison Matrix Block */
.matrix-block {
  width: 100%;
  background: #ffffff;
  border-bottom: 8px solid #0f172a;
  padding: clamp(32px, 5vw, 64px) clamp(20px, 4vw, 56px);
}

.matrix-block h2 {
  font-size: clamp(24px, 3vw, 36px);
  margin-bottom: 16px;
  font-weight: 800;
}

.matrix-block p {
  color: var(--text-muted);
  max-width: 1000px;
  margin-bottom: 24px;
  font-size: 16px;
}

/* Lightbox Modal */
dialog#viewer {
  border: 1px solid #475569;
  padding: 0;
  max-width: 98vw;
  max-height: 96vh;
  background: #090d16;
}

dialog#viewer::backdrop {
  background: rgba(0, 0, 0, 0.85);
  backdrop-filter: blur(4px);
}

dialog#viewer .dialog-header {
  display: flex;
  justify-content: flex-end;
  padding: 8px 14px;
  background: #0f172a;
  border-bottom: 1px solid #334155;
}

dialog#viewer button {
  background: #e2e8f0;
  color: #0f172a;
  border: none;
  padding: 6px 14px;
  font-weight: 700;
  cursor: pointer;
}

dialog#viewer img {
  display: block;
  max-width: 96vw;
  max-height: 88vh;
  margin: 0 auto;
  object-fit: contain;
}

/* Footer */
footer.master-footer {
  width: 100%;
  background: #090d16;
  color: #94a3b8;
  padding: 32px clamp(20px, 4vw, 56px);
  text-align: center;
  font-size: 14px;
  border-top: 1px solid #1e293b;
}

/* Print Styles */
@media print {
  .master-topbar, nav.lab-subnav, dialog#viewer, .master-actions {
    display: none !important;
  }
  body, main {
    background: #ffffff;
    color: #000000;
  }
  .stage, .flow-stage {
    border-left: none;
    margin-left: 0;
    padding-left: 0;
  }
  .image-card, .gallery-item {
    break-inside: avoid;
  }
}
"""

# Global JavaScript for Sticky Navigation, Lightbox, and Detail Toggles
UNIFIED_JS = """
document.addEventListener('DOMContentLoaded', () => {
  // 1. Global Image Lightbox Modal
  const viewer = document.querySelector('#viewer');
  const viewerImg = viewer.querySelector('img');
  document.querySelectorAll('.image-card img, .gallery-item img, figure img').forEach(image => {
    image.tabIndex = 0;
    const openModal = () => {
      viewerImg.src = image.src;
      viewerImg.alt = image.alt || 'Chi tiết hình ảnh thực nghiệm';
      viewer.showModal();
    };
    image.addEventListener('click', openModal);
    image.addEventListener('keydown', event => {
      if (event.key === 'Enter') openModal();
    });
  });
  const closeBtn = viewer.querySelector('button');
  if (closeBtn) closeBtn.addEventListener('click', () => viewer.close());
  viewer.addEventListener('click', event => {
    if (event.target === viewer) viewer.close();
  });

  // 2. Master Active Nav Tracker via IntersectionObserver
  if ('IntersectionObserver' in window) {
    const masterLinks = [...document.querySelectorAll('.master-nav-links a')];
    const labSections = document.querySelectorAll('.lab-section, #overview, #matrix');
    const masterObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          masterLinks.forEach(link => {
            if (link.getAttribute('href') === '#' + entry.target.id) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }
      });
    }, { rootMargin: '-20% 0px -60% 0px' });
    labSections.forEach(section => masterObserver.observe(section));

    // Local subnav links
    const subnavLinks = [...document.querySelectorAll('nav.lab-subnav a')];
    const stages = document.querySelectorAll('.stage, .flow-stage, .flow-map, .support');
    const stageObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          subnavLinks.forEach(link => {
            if (link.getAttribute('href') === '#' + entry.target.id) {
              link.setAttribute('aria-current', 'location');
            } else {
              link.removeAttribute('aria-current');
            }
          });
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px' });
    stages.forEach(stage => stageObserver.observe(stage));
  }

  // 3. Print handling: Expand details before print, restore after
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('details').forEach(detail => {
      detail.dataset.previous = detail.open;
      detail.open = true;
    });
  });
  window.addEventListener('afterprint', () => {
    document.querySelectorAll('details').forEach(detail => {
      detail.open = detail.dataset.previous === 'true';
    });
  });
});

// Toggle all code snippets across the entire master report
let codeOpen = false;
function toggleAllCode() {
  codeOpen = !codeOpen;
  document.querySelectorAll('details').forEach(d => d.open = codeOpen);
  const btn = document.querySelector('#btn-toggle-code');
  if (btn) btn.textContent = codeOpen ? 'Thu gọn code' : 'Mở tất cả code';
}
"""


def render_unified_full_html() -> str:
    """Generate edge-to-edge standalone HTML report combining Lab 02, 03, and 04 in full."""
    txt2 = (LABS_DIR / "lab02" / "LAB_02_REPORT.html").read_text(encoding="utf-8")
    txt3 = (LABS_DIR / "lab03" / "LAB_03_REPORT.html").read_text(encoding="utf-8")
    txt4 = (LABS_DIR / "lab04" / "LAB_04_REPORT.html").read_text(encoding="utf-8")

    # 1. Process Lab 02
    l2_nav_match = re.search(r'(<nav aria-label="Mục lục">.*?</nav>)', txt2, re.S)
    l2_nav = l2_nav_match.group(1) if l2_nav_match else ""
    l2_nav = l2_nav.replace('<nav aria-label="Mục lục">', '<nav class="lab-subnav" aria-label="Mục lục Lab 02">')

    l2_map_match = re.search(r'(<section class="flow-map"[^>]*>.*?</section>)', txt2, re.S)
    l2_map = l2_map_match.group(1) if l2_map_match else ""

    l2_res_match = re.search(r'(<div id="results">.*?)(<dialog|<footer>|<script)', txt2, re.S)
    l2_results = l2_res_match.group(1).rstrip() if l2_res_match else ""
    if l2_results.endswith("</div></div>"):
        l2_results = l2_results[:-6]  # Clean trailing outer container closing tag

    l2_content = f"""
    <div class="lab-section-header lab02-header" id="lab02">
      <div class="eyebrow">CHUYÊN ĐỀ 01 · 11 YÊU CẦU THỰC NGHIỆM</div>
      <h2>Lab 02 — Toán Tử Điểm Ảnh &amp; Bộ Lọc Không Gian</h2>
      <p>Toàn bộ 11 yêu cầu thực nghiệm: Biến đổi cường độ sáng (&beta;=&plusmn;50), tương phản (&alpha;={0.5, 1.8}), âm bản, phân ngưỡng histogram; bộ lọc tuyến tính Mean vs Gaussian; làm nét Sharpening (&Sigma;=1.000) vs Laplacian; đạo hàm biên Sobel vs Prewitt; quy tắc tổng kernel và lọc phi tuyến Median khử 100% cực trị muối tiêu.</p>
      <div style="display:flex;gap:14px;margin-top:12px;flex-wrap:wrap;">
        <a href="../../notebooks/labs/lab_02.ipynb" style="color:#7dd3fc;text-decoration:none;font-weight:700;">Notebook lab_02.ipynb &nearr;</a>
        <a href="lab02/LAB_02_REPORT.md" style="color:#7dd3fc;text-decoration:none;font-weight:700;">Báo Cáo Lab 02 Markdown &nearr;</a>
      </div>
    </div>
    {l2_nav}
    {l2_map}
    <div class="lab-body">
      {l2_results}
    </div>
    """
    for id_name in ['flow', 'prepare', 'point', 'linear', 'edges', 'kernel', 'nonlinear', 'comparison', 'principles']:
        l2_content = re.sub(rf'id="{id_name}"', f'id="lab02-{id_name}"', l2_content)
        l2_content = re.sub(rf'href="#{id_name}"', f'href="#lab02-{id_name}"', l2_content)

    # 2. Process Lab 03
    l3_nav_match = re.search(r'(<nav aria-label="Lộ trình">.*?</nav>)', txt3, re.S)
    l3_nav = l3_nav_match.group(1) if l3_nav_match else ""
    l3_nav = l3_nav.replace('<nav aria-label="Lộ trình">', '<nav class="lab-subnav" aria-label="Lộ trình Lab 03">')

    l3_map_match = re.search(r'(<section class="flow-map"[^>]*>.*?</section>)', txt3, re.S)
    l3_map = l3_map_match.group(1) if l3_map_match else ""

    l3_stages = "".join(re.findall(r'(<section class="(?:stage|support)"[^>]*>.*?</section>)', txt3, re.S))

    l3_content = f"""
    <div class="lab-section-header lab03-header" id="lab03">
      <div class="eyebrow">CHUYÊN ĐỀ 02 · 4 YÊU CẦU &amp; CÂU HỎI MỞ RỘNG</div>
      <h2>Lab 03 — Tách Biên Cải Tiến Canny &amp; Nhận Dạng Hình Học</h2>
      <p>Toàn bộ thí nghiệm tách biên vi sai: Đối sánh thư viện OpenCV (30,980 px) vs Skimage (7,554 px); quét tham số sigma (&sigma;=0.5 &rarr; 3.5) và ngưỡng kép; lọc nhiễu Gauss, muối tiêu, vân gạch brick; cân bằng histogram cứu ảnh mờ (0 &rarr; 13,897 px); phân đoạn vùng mask kín 41,111 px và HoughCircles bắt trọn vẹn 24/24 đồng xu.</p>
      <div style="display:flex;gap:14px;margin-top:12px;flex-wrap:wrap;">
        <a href="../../notebooks/labs/lab_03.ipynb" style="color:#86efac;text-decoration:none;font-weight:700;">Notebook lab_03.ipynb &nearr;</a>
        <a href="lab03/LAB_03_REPORT.md" style="color:#86efac;text-decoration:none;font-weight:700;">Báo Cáo Lab 03 Markdown &nearr;</a>
      </div>
    </div>
    {l3_nav}
    {l3_map}
    <div class="lab-body">
      {l3_stages}
    </div>
    """
    for id_name in ['flow', 'input', 'libraries', 'parameters', 'preprocess', 'regions', 'shapes', 'support']:
        l3_content = re.sub(rf'id="{id_name}"', f'id="lab03-{id_name}"', l3_content)
        l3_content = re.sub(rf'href="#{id_name}"', f'href="#lab03-{id_name}"', l3_content)

    # 3. Process Lab 04
    l4_nav_match = re.search(r'(<nav aria-label="Lộ trình">.*?</nav>)', txt4, re.S)
    l4_nav = l4_nav_match.group(1) if l4_nav_match else ""
    l4_nav = l4_nav.replace('<nav aria-label="Lộ trình">', '<nav class="lab-subnav" aria-label="Lộ trình Lab 04">')

    l4_map_match = re.search(r'(<section class="flow-map"[^>]*>.*?</section>)', txt4, re.S)
    l4_map = l4_map_match.group(1) if l4_map_match else ""

    l4_stages = "".join(re.findall(r'(<section class="(?:stage|support)"[^>]*>.*?</section>)', txt4, re.S))

    l4_content = f"""
    <div class="lab-section-header lab04-header" id="lab04">
      <div class="eyebrow">CHUYÊN ĐỀ 03 · 147 CẶP ẢNH &amp; WAVELET HASH 192-BIT</div>
      <h2>Lab 04 — Mã Băm Nhận Thức Wavelet Hash RGB 192-bit</h2>
      <p>Toàn bộ thí nghiệm băm sóng nhỏ 2D DWT db4: Trích xuất vector băm 192-bit từ 3 kênh RGB; ma trận Hamming 26&times;26; kiểm định độ bền 7 biến đổi; tối ưu ngưỡng Youden &tau;=18 bits trên Calibration (100% Acc); đánh giá Evaluation 98.77% Acc (ROC AUC=1.0000); so sánh 4 cấu hình Wavelet và truy vấn Top-3.</p>
      <div style="display:flex;gap:14px;margin-top:12px;flex-wrap:wrap;">
        <a href="../../notebooks/labs/lab_04.ipynb" style="color:#d8b4fe;text-decoration:none;font-weight:700;">Notebook lab_04.ipynb &nearr;</a>
        <a href="lab04/LAB_04_REPORT.md" style="color:#d8b4fe;text-decoration:none;font-weight:700;">Báo Cáo Lab 04 Markdown &nearr;</a>
      </div>
    </div>
    {l4_nav}
    {l4_map}
    <div class="lab-body">
      {l4_stages}
    </div>
    """
    for id_name in ['flow', 'dataset', 'hash', 'hamming', 'threshold', 'robustness', 'methods', 'retrieval', 'support']:
        l4_content = re.sub(rf'id="{id_name}"', f'id="lab04-{id_name}"', l4_content)
        l4_content = re.sub(rf'href="#{id_name}"', f'href="#lab04-{id_name}"', l4_content)

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Báo Cáo Toàn Diện Thực Hành Thị Giác Máy Tính: Lab 02 · 03 · 04 | UTH</title>
  <style>{UNIFIED_CSS}</style>
</head>
<body>
  <!-- Master Sticky Navigation Bar -->
  <header class="master-topbar">
    <a href="#overview" class="master-brand">
      <span class="tag">UTH CV</span>
      <span>Báo Cáo Tổng Hợp Labs 02 · 03 · 04</span>
    </a>
    <nav class="master-nav-links" aria-label="Điều hướng chính">
      <a href="#overview" class="btn-overview active">🚀 Tổng Quan</a>
      <a href="#lab02" class="btn-lab2">⚡ Lab 02: Lọc Không Gian</a>
      <a href="#lab03" class="btn-lab3">🎯 Lab 03: Tách Biên Canny</a>
      <a href="#lab04" class="btn-lab4">🌊 Lab 04: Wavelet Hash RGB</a>
      <a href="#matrix" class="btn-matrix">📊 Ma Trận Nghiệm Thu</a>
    </nav>
    <div class="master-actions">
      <button type="button" id="btn-toggle-code" onclick="toggleAllCode()">Mở tất cả code</button>
      <button type="button" onclick="window.print()">In / PDF</button>
    </div>
  </header>

  <main>
    <!-- Master Executive Hero Banner -->
    <section class="master-hero" id="overview">
      <div class="eyebrow">TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) — KHOA CÔNG NGHỆ THÔNG TIN</div>
      <h1>BÁO CÁO TOÀN DIỆN THỰC HÀNH THỊ GIÁC MÁY TÍNH<br/>CHUYÊN ĐỀ LIÊN HOÀN: LAB 02 – LAB 03 – LAB 04</h1>
      <p class="lead">Hệ thống thực nghiệm kết nối xuyên suốt 3 tầng kiến trúc thị giác máy tính: từ xử lý mức xám &amp; tích chập không gian (Lab 02), tối ưu vi sai tách biên đa hướng &amp; trích chọn hình học (Lab 03), đến biểu diễn nhận thức sóng nhỏ tần số không gian Wavelet Hash RGB 192-bit (Lab 04).</p>

      <div class="kpi-strip">
        <div class="kpi-card" style="border-left-color: #38bdf8;">
          <div class="kpi-val">11 / 11 YC</div>
          <div class="kpi-title">Lab 02: Lọc Không Gian</div>
          <div class="kpi-sub">Pearson r &approx; 0.9961, &Sigma;=1.000, 100% khử muối tiêu</div>
        </div>
        <div class="kpi-card" style="border-left-color: #4ade80;">
          <div class="kpi-val">24 / 24 Xu</div>
          <div class="kpi-title">Lab 03: Tách Biên Canny</div>
          <div class="kpi-sub">100% phát hiện hình tròn, phục hồi ảnh mờ 0 &rarr; 13,897 px</div>
        </div>
        <div class="kpi-card" style="border-left-color: #c084fc;">
          <div class="kpi-val">98.77% Acc</div>
          <div class="kpi-title">Lab 04: Wavelet Hash RGB</div>
          <div class="kpi-sub">ROC AUC 1.0000, 192-bit Hex, Youden &tau; = 18 bits</div>
        </div>
        <div class="kpi-card" style="border-left-color: #fbbf24;">
          <div class="kpi-val">58 / 58 PASS</div>
          <div class="kpi-title">Kiểm Thử Hồi Quy</div>
          <div class="kpi-sub">100% Unit Tests vượt qua (22 Lab 02 + 11 Lab 03 + 25 Lab 04)</div>
        </div>
      </div>
    </section>

    <!-- Section 1: Lab 02 -->
    <section class="lab-section" id="lab02-section">
      {l2_content}
    </section>

    <!-- Section 2: Lab 03 -->
    <section class="lab-section" id="lab03-section">
      {l3_content}
    </section>

    <!-- Section 3: Lab 04 -->
    <section class="lab-section" id="lab04-section">
      {l4_content}
    </section>

    <!-- Section 4: Master Cross-Lab Comparison Matrix -->
    <section class="matrix-block" id="matrix">
      <div class="eyebrow" style="font-size:12px;letter-spacing:0.15em;text-transform:uppercase;color:#64748b;font-weight:700;">TỔNG KẾT ĐỐI SÁNH KỸ THUẬT</div>
      <h2>Ma Trận Nghiệm Thu &amp; So Sánh Chéo 3 Chuyên Đề</h2>
      <p>Đối chiếu toàn diện các khía cạnh toán học, kích thước đặc trưng, độ phức tạp thuật toán và giá trị kỹ thuật thực tiễn được kiểm chứng qua 58 bài kiểm thử tự động.</p>

      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th style="min-width: 180px;">Tiêu Chí Đánh Giá</th>
              <th>Lab 02: Lọc Không Gian &amp; Điểm Ảnh</th>
              <th>Lab 03: Tách Biên Canny &amp; Hình Học</th>
              <th>Lab 04: Wavelet Hash RGB 192-bit</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Bản chất toán học</strong></td>
              <td>Ánh xạ mức xám điểm ảnh &amp; Tích chập 2D không gian</td>
              <td>Tối ưu hóa hàm tiêu chuẩn John Canny &amp; Không gian tích lũy Hough</td>
              <td>Biến đổi sóng nhỏ 2D DWT Daubechies 4 (`db4`) &amp; Khoảng cách Hamming</td>
            </tr>
            <tr>
              <td><strong>Kích thước đặc trưng</strong></td>
              <td>Giữ nguyên kích thước ma trận ảnh gốc (1920&times;1080 uint8)</td>
              <td>Bản đồ nhị phân cạnh 1-pixel &amp; Vector tọa độ hình học (x, y, r)</td>
              <td>Vector nhị phân cô đọng <strong>192-bit</strong> (48 ký tự Hex, 24 bytes)</td>
            </tr>
            <tr>
              <td><strong>Độ phức tạp thuật toán</strong></td>
              <td>&Omicron;(H &middot; W &middot; k<sup>2</sup>)</td>
              <td>&Omicron;(H &middot; W) + &Omicron;(H &middot; W &middot; R)</td>
              <td>&Omicron;(H &middot; W) trích xuất + &Omicron;(1) so khớp bit bằng POPCNT</td>
            </tr>
            <tr>
              <td><strong>Kết quả then chốt</strong></td>
              <td>
                &bull; Sobel vs Prewitt tương quan Pearson r &approx; 0.9961<br>
                &bull; Sharpening bảo toàn năng lượng sáng &Sigma;=1.000<br>
                &bull; Median khử triệt để 100% điểm cực trị muối tiêu 3%
              </td>
              <td>
                &bull; OpenCV 30,980 px vs Skimage 7,554 px<br>
                &bull; Cân bằng tương phản phục hồi ảnh mù: 0 &rarr; 13,897 px<br>
                &bull; HoughCircles phát hiện chính xác <strong>24 / 24 đồng xu (100%)</strong>
              </td>
              <td>
                &bull; Ngưỡng tối ưu Youden &tau; = 18 bits (Acc 100% trên Calibration)<br>
                &bull; Evaluation: <strong>80 / 81 đúng (98.77% Acc)</strong>, ROC AUC = <strong>1.0000</strong><br>
                &bull; Top-1 Retrieval đạt 100% chính xác (D<sub>H</sub> = 2)
              </td>
            </tr>
            <tr>
              <td><strong>Kiểm thử hồi quy</strong></td>
              <td><span style="color:#16a34a;font-weight:700;">22 / 22 PASS</span> (<code>tests/test_filters.py</code>)</td>
              <td><span style="color:#16a34a;font-weight:700;">11 / 11 PASS</span> (<code>tests/test_canny.py</code>)</td>
              <td><span style="color:#16a34a;font-weight:700;">25 / 25 PASS</span> (<code>test_wavelet_hash</code>, <code>pairs</code>)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style="margin-top:28px;padding:20px;background:#f8fafc;border:1px solid #cbd5e1;">
        <h4 style="margin-top:0;font-size:16px;color:#0f172a;">Lệnh Tái Lập Thí Nghiệm &amp; Kiểm Thử Toàn Cục</h4>
        <pre><code class="language-bash"># Chạy toàn bộ 58 bài kiểm thử tự động cho cả 3 chuyên đề
.venv/bin/pytest tests/test_filters.py tests/test_canny.py tests/test_wavelet_hash.py tests/test_lab04_baseline.py tests/test_lab04_pairs.py

# Kết xuất lại báo cáo HTML toàn diện full màn hình
python scripts/render_unified_report.py</code></pre>
      </div>
    </section>
  </main>

  <footer class="master-footer">
    UTH Computer Vision Workspace &middot; Trường Đại học Giao thông Vận tải TP.HCM &middot; Khoa Công Nghệ Thông Tin &middot; 2026
  </footer>

  <!-- Lightbox Viewer Modal -->
  <dialog id="viewer" aria-label="Xem ảnh phóng to">
    <div class="dialog-header">
      <button type="button" aria-label="Đóng cửa sổ">Đóng &times;</button>
    </div>
    <img src="" alt="Chi tiết hình ảnh phóng to">
  </dialog>

  <script>{UNIFIED_JS}</script>
</body>
</html>"""


def render_unified_markdown_report() -> str:
    r"""Generate Comprehensive Unified Markdown Report for Labs 02-04."""
    return r"""# BÁO CÁO TỔNG HỢP CHUYÊN ĐỀ THỊ GIÁC MÁY TÍNH: LAB 02 – LAB 03 – LAB 04
## TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) — KHOA CÔNG NGHỆ THÔNG TIN

- **Môi trường thực nghiệm**: Python 3.14.8 | OpenCV 5.0.0 | PyWavelets 1.8.0 | Scikit-Image 0.26.0 | NumPy 2.2.6
- **Báo cáo Slide Trình Chiếu HTML tương tác 100vh**: [LABS_02_03_04_REPORT.html](LABS_02_03_04_REPORT.html)
- **Kiểm thử hồi quy tự động**: `58/58 passed` (100% tests cho 3 lab tại `tests/test_filters.py`, `tests/test_canny.py`, `tests/test_wavelet_hash.py`, `tests/test_lab04_baseline.py`, `tests/test_lab04_pairs.py`)

---

## 🚀 1. TỔNG QUAN ĐIỀU HÀNH & TIẾN TRÌNH LIÊN HOÀN 3 CHUYÊN ĐỀ

Chuỗi thực nghiệm kết nối xuyên suốt 3 tầng kiến trúc thị giác máy tính: từ **xử lý cường độ mức xám & tích chập không gian (Lab 02)** lên **tách biên tối ưu vi sai & nhận dạng hình học (Lab 03)** và đạt tới **trích xuất đặc trưng nhận thức sóng nhỏ Wavelet Hash RGB 192-bit (Lab 04)**.

```mermaid
flowchart TD
    subgraph LAB02 ["1. LAB 02: TOÁN TỬ KHÔNG GIAN & ĐIỂM ẢNH"]
        direction TB
        L2_IN["Ảnh Mẫu: image-2.jpg (1920×1080 BGR/Gray)"]
        L2_P1["Phần I: Point Ops (g = clip(α·f + β))<br/>β=±50, α={0.5, 1.8}, 255-f, T={80, 127, 180}"]
        L2_P2["Phần II: Linear Filters (Σ = 1.000)<br/>Mean (7×7, 21×21) vs Gaussian (k ≈ 6σ + 1) & Sharpen"]
        L2_P3["Phần III: Derivatives & Non-linear<br/>Sobel vs Prewitt (r≈0.996) & Median khử 100% cực trị muối tiêu"]
        L2_IN --> L2_P1 --> L2_P2 --> L2_P3
    end

    subgraph LAB03 ["2. LAB 03: TÁCH BIÊN CẢI TIẾN CANNY & HÌNH HỌC"]
        direction TB
        L3_IN["Ảnh Mẫu skimage: camera, brick (512×512), coins (303×384)"]
        L3_ST["5 Bước Toán Học Canny<br/>Gauss mượt → Sobel G,θ → NMS 1px → Ngưỡng kép → Hysteresis"]
        L3_EXP["Thực Nghiệm Chuyên Sâu<br/>• YC1: OpenCV (30,980 px) vs Skimage (7,554 px)<br/>• YC2: Quét σ (20,012 → 793 px) & 4 cặp ngưỡng<br/>• YC3: Cân bằng equalizeHist phục hồi ảnh mù (0 → 13,897 px)<br/>• YC4: Closing + fill holes (Mask 41,111 px) & HoughCircles (24/24 xu)"]
        L3_IN --> L3_ST --> L3_EXP
    end

    subgraph LAB04 ["3. LAB 04: WAVELET HASH RGB 192-BIT"]
        direction TB
        L4_IN["Tập Dữ Liệu: 26 Nguồn → 130 Ảnh (147 Cặp)<br/>Split không rò rỉ: Calib 66 cặp / Eval 81 cặp"]
        L4_WT["2D DWT db4 Level 1 (LL1 131×131 → 8×8)<br/>Lượng tử trung vị 64 bit/kênh → Vector [R|G|B] 192-bit"]
        L4_EV["Đánh Giá & Ứng Dụng<br/>• Ngưỡng Youden τ = 18 bits (Calib Acc = 100%)<br/>• Evaluation: 80/81 đúng (98.77% Acc, ROC AUC = 1.0000)<br/>• Top-3 Nearest Neighbor: Top-1 đạt chính xác 100% (D_H = 2)"]
        L4_IN --> L4_WT --> L4_EV
    end

    LAB02 --> LAB03 --> LAB04
```

### Bảng Ma Trận Nghiệm Thu Toàn Diện 3 Bài Lab

| Tiêu Chí So Sánh | Lab 02: Lọc Không Gian | Lab 03: Canny & Hough Circles | Lab 04: Wavelet Hash RGB 192-bit |
| :--- | :--- | :--- | :--- |
| **Ảnh mẫu thực nghiệm** | `image-2.jpg` (1920×1080 uint8) | `camera`, `brick` (512×512), `coins` (303×384) | 26 nguồn (Chihuahua-Muffin, Mèo-Kem) → 130 ảnh |
| **Bản chất toán học** | Tích chập 2D không gian & ánh xạ điểm | Tối ưu hóa 3 tiêu chí John Canny & tích lũy Hough | Biến đổi tần số không gian 2D DWT db4 & khoảng cách Hamming |
| **Độ dài / Chiều đặc trưng** | Giữ nguyên kích thước ảnh gốc | Bản đồ biên nhị phân 1-pixel & danh sách tọa độ (x,y,r) | Vector nhị phân cô đọng **192-bit** (48 ký tự Hex) |
| **Độ phức tạp tính toán** | $\mathcal{O}(H \cdot W \cdot k^2)$ | $\mathcal{O}(H \cdot W) + \mathcal{O}(H \cdot W \cdot R)$ | $\mathcal{O}(H \cdot W) + \mathcal{O}(1)$ so khớp bit bằng POPCNT |
| **Chỉ số kỹ thuật đột phá** | • Sobel vs Prewitt $r \approx 0.996$<br/>• Sharpen bảo toàn độ sáng $\Sigma = 1.000$<br/>• Median triệt tiêu 100% cực trị muối tiêu 3% | • OpenCV 30,980 px vs Skimage 7,554 px<br/>• equalizeHist cứu ảnh mờ: $0 \to 13,897$ px<br/>• HoughCircles bắt trọn vẹn **24/24 đồng xu (100%)** | • Ngưỡng tối ưu Youden $\tau = 18$ bits<br/>• Evaluation **80/81 đúng (98.77% Acc)**<br/>• ROC AUC = **1.0000** hoàn hảo, Top-1 Precision 100% |
| **Kiểm thử tự động** | **22 / 22 PASS** (`tests/test_filters.py`) | **11 / 11 PASS** (`tests/test_canny.py`) | **25 / 25 PASS** (`test_wavelet_hash`, `baseline`, `pairs`) |

---

## 📌 2. CHI TIẾT KỸ THUẬT CHUYÊN ĐỀ LAB 02: TOÁN TỬ ĐIỂM & LỌC KHÔNG GIAN

### 2.1. Phần I — Toán Tử Điểm Ảnh (YC1–YC4)
- **YC1 (Độ sáng)**: $g(x,y) = \text{clip}(f(x,y) + \beta, 0, 255)$. Với $\beta = +50$, dải mức xám tịnh tiến sang $[50, 255]$; với $\beta = -50$, dải dịch sang $[0, 197]$. Phép cộng không làm thay đổi hình dáng lược đồ xám.
- **YC2 (Tương phản)**: $g(x,y) = \text{clip}(\alpha \cdot f(x,y), 0, 255)$. Với $\alpha = 0.5$, co dải động về $[0, 127]$ (ảnh bạc màu mờ đục); với $\alpha = 1.8$, kéo giãn dải mức xám, tăng độ tương phản rõ rệt nhưng bão hòa trắng tại 255.
- **YC3 (Âm bản)**: $g(x,y) = 255 - f(x,y)$. Đảo ngược toàn bộ thang sáng tối, giúp mắt người nhận diện rõ các chi tiết bị chìm trong vùng tối.
- **YC4 (Cắt ngưỡng & Histogram)**: Phân ngưỡng $g(x,y) = 255$ nếu $f(x,y) \ge T$ ngược lại bằng $0$. Với $T=127$, chia đều phân bố; lược đồ histogram sau cắt ngưỡng quy tụ về đúng 2 vạch rời rạc $\{0, 255\}$.

### 2.2. Phần II — Bộ Lọc Tuyến Tính (YC5–YC7)
- **YC5 & YC6 (Mean vs Gaussian Blur)**:
  - Mean filter gán trọng số đồng nhất $1/k^2$, tạo hiện tượng lem biên dạng ô vuông (box artifact).
  - Gaussian filter áp dụng hàm chuông chuẩn 2 chiều đối xứng tâm $G(x,y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$.
  - **Quy tắc kích thước kernel**: $k \approx 6\sigma + 1$ (bao trọn dải $\pm 3\sigma$) giúp tích hợp $99.73\%$ năng lượng, ngăn ngừa hiện tượng cắt cụt biên (truncation artifact).
- **YC7 (Làm nét Sharpening)**:
  - Bản chất toán học: $\text{Sharpen} = \text{Identity} - \text{Laplacian}$.
  - Đặc điểm nhận dạng: Kernel làm nét bắt buộc có tổng trọng số $\Sigma w_{ij} = 1.000$ (bảo toàn độ sáng trung bình) VÀ có hệ số âm ở lân cận để khuếch đại vi sai biên.
  - Phân biệt: Laplacian ($\Sigma = 0$) là toán tử tách biên, không phải làm nét.

### 2.3. Phần III — Đạo Hàm Biên Gradient, Tổng Kernel & Lọc Phi Tuyến (YC8–YC11)
- **YC8 (Sobel vs Prewitt)**: Đạo hàm bậc một theo hai trục trực giao $Gx, Gy$. Sobel gán trọng số trung tâm bằng 2 ($[1, 2, 1]$), tích hợp sẵn bộ lọc Gauss 1D khử nhiễu. Hệ số tương quan Pearson giữa hai toán tử đạt $r \approx 0.9961$; biên độ trung bình của Sobel lớn hơn Prewitt $\sim 1.37\times$ ($39.5$ vs $28.9$).
- **YC9 (Quy tắc tổng trọng số kernel)**: Kiểm chứng trên vùng phẳng $c = 100$:
  - $\Sigma = 0.5 \implies \text{output} = 50$ (tối đi một nửa năng lượng).
  - $\Sigma = 1.0 \implies \text{output} = 100$ (bảo toàn nguyên vẹn độ sáng).
  - $\Sigma = 2.0 \implies \text{output} = 200$ (cháy sáng).
  - $\Sigma = 0.0 \implies \text{output} = 0$ (vùng phẳng biến mất, chỉ giữ lại đường biên).
- **YC10 & YC11 (Lọc phi tuyến trên 3% nhiễu muối tiêu)**: Median Filter áp dụng thống kê thứ tự trung vị, loại bỏ triệt để $100\%$ điểm cực trị $0$ và $255$ mà không làm nhòe ảnh; Bilateral Filter làm mịn vùng phẳng giữ cạnh sắc nét nhưng không triệt tiêu hết hạt muối tiêu vì vẫn bản chất là trung bình có trọng số.

---

## 📌 3. CHI TIẾT KỸ THUẬT CHUYÊN ĐỀ LAB 03: TÁCH BIÊN CẢI TIẾN CANNY & HÌNH HỌC

### 3.1. 5 Bước Toán Học Canny Kinh Điển
1. **Gaussian Smoothing**: Làm mịn ảnh với bán kính $k = 2\lceil 3\sigma \rceil + 1$ để triệt tiêu nhiễu tần số cao.
2. **Sobel Gradient**: Tính biên độ $G = \sqrt{Gx^2 + Gy^2}$ và góc pháp tuyến $\theta = \text{atan2}(Gy, Gx)$.
3. **Non-Maximum Suppression (NMS)**: Lượng tử hóa góc gradient về 4 hướng chính ($0^\circ, 45^\circ, 90^\circ, 135^\circ$), chỉ giữ lại điểm cực đại cục bộ, gọt mỏng đường biên về đúng $1$ pixel.
4. **Double Thresholding**: Phân loại ứng viên biên thành 3 nhóm: Strong ($G \ge T_{\text{high}}$), Weak ($T_{\text{low}} \le G < T_{\text{high}}$), và Non-edge ($G < T_{\text{low}}$).
5. **Hysteresis Tracking**: Duyệt 8-lân cận, giữ lại các pixel Weak có liên kết trực tiếp với Strong; loại bỏ các điểm yếu cô lập.

### 3.2. Kết Quả Thực Nghiệm Trọng Tâm (YC1–YC4)
- **YC1 (OpenCV vs Scikit-image)**: Trên ảnh `camera` (512×512) với cùng ngưỡng 50/150, `cv2.Canny` giữ **30,980** pixel biên vs `skimage.feature.canny` giữ **7,554** pixel biên. Khác biệt do skimage tự động làm mịn Gauss nội bộ ($\sigma=1$) và chuẩn hóa dải float $[0, 1]$.
- **YC2 (Quét tham số $\sigma$ và ngưỡng kép)**:
  - Quét $\sigma$ cố định $T=50/150$: $\sigma=0.5$ (20,012 px) $\to \sigma=1.0$ (7,937 px mốc chuẩn) $\to \sigma=2.0$ (4,740 px) $\to \sigma=3.5$ (793 px, xóa sạch chi tiết nhỏ).
  - Quét cặp ngưỡng cố định $\sigma=1.0$: $30/90$ (15,972 px nhạy cao), $50/150$ (7,937 px tỉ lệ vàng 1:3), $100/200$ (6,025 px khắt khe), $20/180$ (8,483 px khoảng hysteresis rộng).
- **YC3 (Tiền xử lý ảnh thách thức)**:
  - Nhiễu Gauss ($\text{std}=20$): Lọc Bilateral ($d=9, 50, 5$) giảm biên nhiễu từ $9,003 \to 5,938$ px.
  - Nhiễu muối tiêu ($3\%$): Lọc Median 3×3 giảm biên giả từ $13,201 \to 6,963$ px.
  - Ảnh nhiều vân `brick` (512×512): Lọc Bilateral giảm biên vân rác bề mặt từ $18,750 \to 9,319$ px mà vẫn bảo tồn trọn vẹn đường ron gạch.
  - Ảnh tương phản thấp ($[100, 163]$): Canny trực tiếp mù hoàn toàn (**0 pixel**); áp dụng `cv2.equalizeHist` phục hồi đầy đủ **13,897 pixel** biên.
- **YC4 (Phân đoạn vùng & Hough Circles trên `coins` 303×384)**:
  - Phân đoạn vùng: Canny (50/150) $\to$ Closing Morphology (Ellipse 3×3) nối khe hở $\to$ `scipy.ndimage.binary_fill_holes` tạo mask kín **41,111 pixel trắng**.
  - Nhận dạng hình học: Tiền xử lý Gaussian 9×9 ($\sigma=2.0$) $\to$ `cv2.HoughCircles` phát hiện chính xác trọn vẹn **24 / 24 đồng xu** ($100\%$ phát hiện, không sót, không báo nhầm).

---

## 📌 4. CHI TIẾT KỸ THUẬT CHUYÊN ĐỀ LAB 04: WAVELET HASH RGB 192-BIT

### 4.1. Thiết Kế Tập Dữ Liệu & Quy Tắc Nhãn Không Rò Rỉ
- 26 ảnh nguồn chia thành 2 chủ đề dễ nhầm hình thức: `chihua_muffin` (7 chó, 7 bánh muffin) và `meo_kem` (6 mèo, 6 kem).
- Mỗi ảnh nguồn tạo 4 biến thể: `bright ×1.3`, `dark ×0.7`, `noise σ=10`, `jpeg Q=50` $\to$ 104 biến thể (tổng 130 ảnh).
- Quy tắc gán nhãn:
  - Nhãn **1** (Cùng nguồn): Ảnh gốc ghép với biến thể của chính nó.
  - Nhãn **0** (Khác đối tượng): Hai ảnh gốc khác loại trong cùng chủ đề (chó ↔ muffin, mèo ↔ kem).
- **Chia tập theo nguồn (Seed 42)**:
  - Tập Calibration: 12 nguồn $\to$ **66 cặp** (48 dương, 18 âm).
  - Tập Evaluation: 14 nguồn $\to$ **81 cặp** (56 dương, 25 âm).
  - Đảm bảo không có bất kỳ ảnh nguồn nào xuất hiện đồng thời ở cả hai tập (chống Data Leakage tuyệt đối).

### 4.2. Baseline wHash RGB 192-bit & Độ Bền Vững
- **Pipeline trích xuất**:
  1. Chuẩn hóa kích thước: Resize về $256 \times 256$ (INTER_AREA) và tách độc lập 3 kênh $R, G, B$.
  2. Biến đổi sóng nhỏ: 2D DWT sử dụng sóng mẹ Daubechies 4 (`db4`, Level 1) trích xuất dải xấp xỉ thấp tần $LL_1$ kích thước $131 \times 131$ chứa trên $93\%$ năng lượng hình thái.
  3. Thu nhỏ & Lượng tử hóa trung vị: Thu nhỏ $LL_1$ về $8 \times 8$ (64 hệ số); gán $\text{bit} = 1$ nếu $LL_{8 \times 8} > \text{median}$ ngược lại bằng $0$ (tạo 32 bit 0 và 32 bit 1, đạt Entropy thông tin cực đại $1.0$).
  4. Ghép vector: Nối 3 chuỗi 64-bit thành vector **192-bit** ($[R_{64} \mid G_{64} \mid B_{64}]$, tương đương 48 ký tự Hex).
- **Khảo sát độ bền vững (Robustness)**:
  - Rất bền với biến đổi cường độ và nén lossy ($D_H \le 18$ bits): `dark ×0.7` (median $0$), `noise σ=10` (median $2$), `jpeg Q=50` (median $3$), `bright ×1.3` (median $7$).
  - Nhạy với biến đổi hình học (làm lệch ma trận tọa độ không gian 8×8): `rotate 5°` (median $15$), `crop 80%` (median $40$), `flip ngang` (median $77$).

### 4.3. Tối Ưu Hóa Ngưỡng Youden & Đánh Giá Độc Lập
- Tối ưu chỉ số Youden $J = \text{Recall} + \text{Specificity} - 1$ trên tập Calibration: Chốt ngưỡng tối ưu **$\tau = 18$ bits** (Accuracy = $100\%$, 48 TP, 18 TN, 0 FP, 0 FN). Khoảng cách Hamming giữa cặp dương $[0, 18]$ và cặp âm $[50, 98]$ cách biệt an toàn 32 bits.
- **Đánh giá trên tập Evaluation (Cố định $\tau = 18$)**:
  - Đúng **80 / 81 cặp** $\implies$ **Accuracy = 98.77%**.
  - **Precision = 100.0%**, **Specificity = 100.0%** (25/25 cặp âm đều có $D_H \ge 70$, không có báo nhầm FP = 0).
  - **Recall = 98.21%**, **F1-Score = 99.10%**, **ROC AUC = 1.0000** hoàn hảo.
  - Phân tích 1 ca lỗi False Negative duy nhất: `meo_kem_doi_tuong_12` vs bản `bright` có $D_H = 20 > 18$ do ảnh bị clip bão hòa 255 làm méo mó phân bố trung vị cục bộ ở vùng kem trắng.
- **Nâng cao 1 (So sánh 4 cấu hình Wavelet)**:
  - `RGB db4 L1` (192-bit): Acc 98.77%, ROC AUC 1.0000 (Tối ưu nhất).
  - `RGB haar L1` (192-bit): Acc 98.77%, ROC AUC 1.0000 (Tương đương db4).
  - `RGB db4 L2` (192-bit): Acc 97.53%, ROC AUC 1.0000 (Mất chi tiết tần số trung bình, 2 FN).
  - `Gray Step L3` (Slide 17, 71,542 bits): Acc 69.14%, Recall 55.36%, ROC AUC 0.7157 (25 FN $\implies$ chứng minh số lượng bit nhiều không đồng nghĩa với hiệu năng tốt hơn).
- **Nâng cao 2 (Ứng dụng tìm kiếm Top-3)**: Truy vấn ảnh JPEG tìm ra chính xác $100\%$ ảnh gốc ở vị trí Top-1 với khoảng cách Hamming $D_H = 2 \le 18$. Các ảnh vị trí Top-2 và Top-3 có $D_H > 80$.

---

## 🛠️ 5. HƯỚNG DẪN TÁI LẬP & TRÌNH CHIẾU (REPRODUCIBILITY)

1. **Khởi chạy Báo Cáo HTML Toàn Diện**:
   - Mở trực tiếp file [LABS_02_03_04_REPORT.html](LABS_02_03_04_REPORT.html) trong bất kỳ trình duyệt web nào (Chrome, Firefox, Safari, Edge).
   - Thanh điều hướng trên cùng cho phép nhảy nhanh giữa Tổng quan, Lab 02, Lab 03, Lab 04 và Ma trận so sánh.
   - Nhấp vào bất kỳ hình ảnh nào để mở cửa sổ phóng to **Lightbox Modal** kiểm tra chi tiết.

2. **Chạy lại kiểm thử hồi quy**:
   ```bash
   .venv/bin/pytest tests/test_filters.py tests/test_canny.py tests/test_wavelet_hash.py tests/test_lab04_baseline.py tests/test_lab04_pairs.py
   # Kết quả: 58 passed in ~1.8s
   ```

3. **Tái tạo báo cáo tổng hợp**:
   ```bash
   python scripts/render_unified_report.py
   ```
"""


def main() -> None:
    """Generate full-bleed edge-to-edge unified HTML and Markdown reports for Labs 02-04."""
    LABS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Full Edge-to-Edge HTML Report
    html_content = render_unified_full_html()
    out_unified_html = LABS_DIR / "LABS_02_03_04_REPORT.html"
    out_unified_html.write_text(html_content, encoding="utf-8")
    print(f"✔ Generated Full-Bleed Unified Report: {out_unified_html} ({out_unified_html.stat().st_size // 1024} KB)")

    # 2. Generate Unified Markdown Technical Report
    md_content = render_unified_markdown_report()
    out_unified_md = LABS_DIR / "LABS_02_03_04_REPORT.md"
    out_unified_md.write_text(md_content, encoding="utf-8")
    print(f"✔ Generated Unified Markdown Report: {out_unified_md} ({out_unified_md.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
