#!/usr/bin/env python3
"""Unified Interactive Visual Dashboard Generator for Labs 02, 03, and 04.

Generates a sleek, edge-to-edge, zero-border-radius interactive HTML dashboard report
(LABS_02_03_04_REPORT.html) with:
1. No bulky sidebar, no bulky multi-row headers - straight to the point ("thẳng vào việc luôn").
2. Single slim 48px top control bar with:
   - Primary Lab Tabs: [🚀 Tổng Quan] [⚡ Lab 02] [🎯 Lab 03] [🌊 Lab 04] [📊 Ma Trận]
   - Stepper Navigation: ◀ [Quick Jump Dropdown (28 steps)] ▶ [Step counter 01/28]
   - Utility controls: [Cuộn liên tục / Từng bước], [Thu gọn chi tiết], [In / PDF]
3. Stepper Mode (DEFAULT): View each experiment step-by-step without scrolling ("không phải lướt tay"),
   controlled via keyboard arrows (← / → / Space), the stepper buttons, or the dropdown list.
4. 100% Full Content Preservation: Every single image (36 base64 images),
   notebook code block (44 pre blocks), mathematical explanation table,
   takeaway, and research extension from Labs 02, 03, and 04 is fully retained
   and visually displayed (<details open> by default).
5. Strictly HTML only (no .md generated).
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


# Sleek, Modern, Straight-To-Point CSS (Zero border-radius, No bulky sidebar)
DASHBOARD_CSS = """
/* Universal Reset & Full Bleed Layout (Zero Border-Radius) */
*, *::before, *::after {
  box-sizing: border-box;
  border-radius: 0 !important;
}

:root {
  --primary-2: #0284c7;
  --primary-3: #16a34a;
  --primary-4: #7c3aed;
  --bg-page: #f8fafc;
  --bg-card: #ffffff;
  --border: #cbd5e1;
  --border-dark: #334155;
  --text-main: #0f172a;
  --text-muted: #475569;
  --topbar-bg: #090d16;
}

html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  max-width: 100%;
  background-color: var(--bg-page);
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.65;
  scroll-behavior: smooth;
}

/* Master Top Sticky Bar (Sleek, Compact: 48px) */
header.master-topbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  width: 100%;
  height: 48px;
  background: var(--topbar-bg);
  border-bottom: 2px solid #22c55e;
  padding: 0 clamp(10px, 2vw, 32px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

.topbar-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #ffffff;
  font-weight: 800;
  font-size: 14px;
  text-decoration: none;
  white-space: nowrap;
}

.brand-tag {
  background: #16a34a;
  color: #fff;
  padding: 2px 7px;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

/* Master Lab Tabs */
.topbar-tabs {
  display: flex;
  align-items: center;
  gap: 4px;
}

.tab-btn {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.15);
  color: #cbd5e1;
  padding: 5px 12px;
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s ease;
}

.tab-btn:hover, .tab-btn.active {
  background: #22c55e;
  color: #091310;
  border-color: #22c55e;
}

.tab-btn.tab-l2.active { background: #38bdf8; color: #02122b; border-color: #38bdf8; }
.tab-btn.tab-l3.active { background: #4ade80; color: #052e16; border-color: #4ade80; }
.tab-btn.tab-l4.active { background: #c084fc; color: #2e1065; border-color: #c084fc; }
.tab-btn.tab-mat.active { background: #fbbf24; color: #451a03; border-color: #fbbf24; }

/* Stepper Controller in Topbar */
.topbar-stepper {
  display: flex;
  align-items: center;
  gap: 6px;
}

.btn-step {
  background: #1e293b;
  border: 1px solid #334155;
  color: #f8fafc;
  width: 32px;
  height: 32px;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 800;
  cursor: pointer;
  transition: all 0.15s ease;
}

.btn-step:hover {
  background: #38bdf8;
  color: #02122b;
  border-color: #38bdf8;
}

.step-select {
  background: #0f172a;
  border: 1px solid #334155;
  color: #f8fafc;
  padding: 5px 10px;
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
  max-width: 380px;
  outline: none;
}

.step-select optgroup {
  background: #090d16;
  color: #38bdf8;
  font-weight: 800;
}

.step-select option {
  background: #0f172a;
  color: #f8fafc;
  padding: 4px;
}

.step-counter {
  font-size: 12px;
  font-weight: 800;
  color: #38bdf8;
  white-space: nowrap;
}

/* Action Controls in Topbar */
.topbar-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

.btn-action {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #f8fafc;
  padding: 5px 11px;
  font-size: 11.5px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s ease;
}

.btn-action:hover {
  background: #ffffff;
  color: #0f172a;
}

/* Stepper Mode (Default: Zero Scrolling Needed) */
body.mode-stepper .content-panel {
  display: none;
}

body.mode-stepper .content-panel.active {
  display: block;
  animation: panelIn 0.12s ease-in;
}

@keyframes panelIn {
  from { opacity: 0.7; }
  to { opacity: 1; }
}

/* Continuous Scroll Mode */
body.mode-scroll .content-panel {
  display: block;
  border-bottom: 8px solid #cbd5e1;
}

/* Content Panel Full Bleed */
.content-panel {
  width: 100%;
  padding: 28px clamp(16px, 4vw, 56px);
  background: #ffffff;
}

.panel-header-banner {
  width: 100%;
  padding: 20px 28px;
  color: #ffffff;
  margin-bottom: 20px;
  border-bottom: 3px solid rgba(255,255,255,0.2);
}

.panel-header-banner.banner-overview { background: linear-gradient(135deg, #091a13 0%, #15803d 100%); border-bottom-color: #22c55e; }
.panel-header-banner.banner-l2 { background: linear-gradient(135deg, #0c1e47 0%, #1e3a8a 100%); border-bottom-color: #38bdf8; }
.panel-header-banner.banner-l3 { background: linear-gradient(135deg, #0b221a 0%, #183f35 100%); border-bottom-color: #4ade80; }
.panel-header-banner.banner-l4 { background: linear-gradient(135deg, #240e4f 0%, #4c1d95 100%); border-bottom-color: #c084fc; }
.panel-header-banner.banner-mat { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-bottom-color: #fbbf24; }

.panel-header-banner .eyebrow {
  font-size: 11px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-weight: 800;
  opacity: 0.85;
  margin-bottom: 4px;
}

.panel-header-banner h2 {
  font-size: clamp(20px, 2.6vw, 32px);
  margin: 0 0 6px 0;
  font-weight: 800;
}

.panel-header-banner p {
  font-size: 14.5px;
  margin: 0;
  max-width: 1100px;
  opacity: 0.92;
}

/* Stage elements */
.flow-map {
  width: 100%;
  background: #ffffff;
  padding: 12px 0 20px 0;
  border-bottom: 1px solid var(--border);
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

.stage, .flow-stage {
  border-left: 3px solid #94a3b8;
  margin: 0 0 24px 16px;
  padding: 0 0 16px 24px;
}

.stage-head, .stage-heading {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
}

.number, .stage-number {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: 42px;
  height: 42px;
  margin-left: -46px;
  background: #0f172a;
  color: #ffffff;
  border: 3px solid var(--bg-page);
  font-weight: 700;
  font-size: 14px;
}

.stage h2, .flow-stage h2 {
  font-size: 22px;
  line-height: 1.3;
  margin: 4px 0;
  color: var(--text-main);
}

.mini-flow {
  padding: 10px 16px;
  background: #e2e8f0;
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
  border-left: 4px solid #0284c7;
  margin: 10px 0 16px 0;
}

.intro, .stage-intro {
  color: var(--text-muted);
  font-size: 14.5px;
  margin: 10px 0 18px 0;
}

/* Images & Visual Galleries */
.image-card, .gallery-item {
  background: #ffffff;
  border: 1px solid var(--border);
  overflow: hidden;
  margin: 18px 0;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.image-card img, .gallery-item img {
  display: block;
  max-width: 100%;
  width: 100%;
  max-height: 850px;
  object-fit: contain;
  cursor: zoom-in;
  background: #ffffff;
}

.image-card figcaption, .gallery-item .caption {
  padding: 12px 18px;
  font-size: 13.5px;
  color: #334155;
  border-top: 1px solid #e2e8f0;
  background: #f8fafc;
}

.image-card figcaption b, .gallery-item .caption strong {
  display: block;
  color: var(--text-main);
  font-size: 14.5px;
  margin-bottom: 3px;
}

.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px;
  margin-top: 14px;
}

.takeaway, .callout {
  padding: 14px 18px;
  background: #f0fdf4;
  border-left: 4px solid #16a34a;
  font-size: 14px;
  color: #14532d;
  margin: 18px 0;
}

/* Visual Details & Code Blocks (OPEN by default) */
details {
  background: #ffffff;
  border: 1px solid var(--border);
  margin: 14px 0;
}

summary {
  padding: 10px 16px;
  cursor: pointer;
  font-weight: 700;
  font-size: 13.5px;
  color: #0f172a;
  background: #f1f5f9;
  border-bottom: 1px solid var(--border);
  user-select: none;
}

.detail-body {
  padding: 16px 18px;
  overflow-wrap: anywhere;
}

.code-note {
  font-size: 12.5px;
  color: #64748b;
  margin: 4px 0;
}

pre {
  padding: 14px 18px;
  background: #0f172a;
  color: #e2e8f0;
  overflow-x: auto;
  line-height: 1.55;
  font-size: 13px;
  font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace;
  margin: 8px 0;
}

code {
  font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace;
  background: #e2e8f0;
  color: #0f172a;
  padding: 2px 5px;
  font-size: 0.88em;
}

pre code {
  background: transparent;
  color: inherit;
  padding: 0;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0;
  font-size: 13.5px;
}

th, td {
  padding: 8px 12px;
  border: 1px solid var(--border);
  text-align: left;
}

th {
  background: #f1f5f9;
  font-weight: 700;
}

/* KPI Cards */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin: 20px 0;
}

.kpi-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-left: 4px solid #16a34a;
  padding: 16px 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.kpi-card .kpi-val {
  font-size: 26px;
  font-weight: 800;
  color: #0f172a;
  line-height: 1.2;
}

.kpi-card .kpi-title {
  font-size: 14px;
  font-weight: 700;
  color: #334155;
  margin: 4px 0 2px 0;
}

.kpi-card .kpi-sub {
  font-size: 12px;
  color: #64748b;
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
  padding: 6px 12px;
  background: #0f172a;
  border-bottom: 1px solid #334155;
}

dialog#viewer button {
  background: #e2e8f0;
  color: #0f172a;
  border: none;
  padding: 5px 12px;
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

/* Responsive Media Queries */
@media (max-width: 1050px) {
  .topbar-brand { display: none; }
  .step-select { max-width: 260px; }
}

@media (max-width: 768px) {
  .topbar-tabs { display: none; }
  .step-select { max-width: 200px; }
  header.master-topbar { padding: 0 10px; }
}

/* Print Styles */
@media print {
  header.master-topbar, dialog#viewer {
    display: none !important;
  }
  body.mode-stepper .content-panel {
    display: block !important;
  }
  .content-panel {
    break-inside: avoid;
    padding: 16px 0;
  }
}
"""

# Interactive JavaScript for Stepper, Quick Dropdown, and Keyboard Controls
DASHBOARD_JS = """
// 28 Sequential Panels
const panelList = [
  { id: 'overview', lab: 'overview', title: '🚀 Toàn Cảnh & KPIs 3 Chuyên Đề' },
  { id: 'lab02-flow', lab: 'lab02', title: 'Lab 02 · 🗺️ Sơ Đồ 5 Nhánh Thí Nghiệm' },
  { id: 'lab02-prepare', lab: 'lab02', title: 'Lab 02 · 01 Chuẩn Bị Ảnh (BGR / Gray)' },
  { id: 'lab02-point', lab: 'lab02', title: 'Lab 02 · 02 Điểm Ảnh YC1–4 (Sáng, Tương Phản, Âm Bản, Ngưỡng)' },
  { id: 'lab02-linear', lab: 'lab02', title: 'Lab 02 · 03 Lọc Tuyến Tính YC5–7 (Mean, Gauss, Làm Nét)' },
  { id: 'lab02-edges', lab: 'lab02', title: 'Lab 02 · 04 Đạo Hàm Biên YC8 (Sobel vs Prewitt)' },
  { id: 'lab02-kernel', lab: 'lab02', title: 'Lab 02 · 05 Tổng Kernel YC9 (Vùng Phẳng & Ảnh Màu)' },
  { id: 'lab02-nonlinear', lab: 'lab02', title: 'Lab 02 · 06 Lọc Phi Tuyến YC10 (Median vs Bilateral)' },
  { id: 'lab02-comparison', lab: 'lab02', title: 'Lab 02 · 07 So Sánh Tổng Hợp YC11 (Master 12 Bộ Lọc)' },
  { id: 'lab02-principles', lab: 'lab02', title: 'Lab 02 · 08 Nguyên Lý & Xác Thực 22 Tests' },
  { id: 'lab03-flow', lab: 'lab03', title: 'Lab 03 · 🗺️ Sơ Đồ 5 Bước Canny & 2 Hướng' },
  { id: 'lab03-input', lab: 'lab03', title: 'Lab 03 · 01 Dữ Liệu Đầu Vào (Camera, Brick, Coins)' },
  { id: 'lab03-libraries', lab: 'lab03', title: 'Lab 03 · 02 YC1 Hai Thư Viện (OpenCV vs Skimage)' },
  { id: 'lab03-parameters', lab: 'lab03', title: 'Lab 03 · 03 YC2 Quét Tham Số (Sigma & Ngưỡng Kép)' },
  { id: 'lab03-preprocess', lab: 'lab03', title: 'Lab 03 · 04 YC3 Tiền Xử Lý Ảnh Khó (Nhiễu, Vân Gạch, Tương Phản)' },
  { id: 'lab03-regions', lab: 'lab03', title: 'Lab 03 · 05 YC4.1 Phân Đoạn Vùng (Closing + Fill Holes)' },
  { id: 'lab03-shapes', lab: 'lab03', title: 'Lab 03 · 06 YC4.2 Nhận Dạng Hình Tròn (Hough 24/24 Xu)' },
  { id: 'lab03-support', lab: 'lab03', title: 'Lab 03 · 07 Câu Hỏi Nghiên Cứu Mở Rộng' },
  { id: 'lab04-flow', lab: 'lab04', title: 'Lab 04 · 🗺️ Sơ Đồ 2D DWT db4 & Luồng Quyết Định' },
  { id: 'lab04-dataset', lab: 'lab04', title: 'Lab 04 · 01 Dữ Liệu & Gán Nhãn (26 Nguồn, 147 Cặp Không Rò Rỉ)' },
  { id: 'lab04-hash', lab: 'lab04', title: 'Lab 04 · 02 YC1 Trích Xuất wHash RGB 192-bit' },
  { id: 'lab04-hamming', lab: 'lab04', title: 'Lab 04 · 03 YC2 Ma Trận Hamming 26x26 & 3 Kênh R,G,B' },
  { id: 'lab04-threshold', lab: 'lab04', title: 'Lab 04 · 04 YC4 Ngưỡng Youden (tau=18) & Đánh Giá 98.77% Acc' },
  { id: 'lab04-robustness', lab: 'lab04', title: 'Lab 04 · 05 YC3 Khảo Sát Độ Bền Vững (7 Biến Đổi)' },
  { id: 'lab04-methods', lab: 'lab04', title: 'Lab 04 · 06 Nâng Cao 1 So Sánh 4 Cấu Hình Wavelet' },
  { id: 'lab04-retrieval', lab: 'lab04', title: 'Lab 04 · 07 Nâng Cao 2 Truy Vấn Top-3 (Top-1 100%)' },
  { id: 'lab04-support', lab: 'lab04', title: 'Lab 04 · 08 Mở Rộng Cấu Trúc Tìm Kiếm (BK-Tree, MIH, DTCWT)' },
  { id: 'matrix', lab: 'matrix', title: '📊 Ma Trận Nghiệm Thu & So Sánh Chéo 3 Lab (58/58 Tests)' }
];

let currentIndex = 0;
let isStepperMode = true;

function activatePanel(index, scrollToTop = true) {
  if (index < 0 || index >= panelList.length) return;
  currentIndex = index;
  const current = panelList[currentIndex];

  // Update Master Lab Tabs (Top Bar)
  document.querySelectorAll('.tab-btn').forEach(tab => {
    if (tab.getAttribute('data-lab') === current.lab) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Update Select Dropdown
  const selectEl = document.querySelector('#step-selector');
  if (selectEl) selectEl.value = current.id;

  // Update Counter
  const counterEl = document.querySelector('#step-counter');
  if (counterEl) counterEl.textContent = `${currentIndex + 1} / ${panelList.length}`;

  // Activate panel in DOM
  if (isStepperMode) {
    document.querySelectorAll('.content-panel').forEach(p => {
      if (p.id === current.id) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });
    if (scrollToTop) window.scrollTo({ top: 0, behavior: 'instant' });
  } else {
    const targetEl = document.getElementById(current.id);
    if (targetEl && scrollToTop) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // Highlight SVG node
  document.querySelectorAll('svg a').forEach(a => {
    if (a.getAttribute('href') === '#' + current.id) {
      a.querySelector('rect')?.setAttribute('stroke-width', '3');
    } else {
      a.querySelector('rect')?.setAttribute('stroke-width', '1');
    }
  });
}

function nextPanel() {
  if (currentIndex < panelList.length - 1) activatePanel(currentIndex + 1);
}

function prevPanel() {
  if (currentIndex > 0) activatePanel(currentIndex - 1);
}

function jumpToPanel(panelId) {
  const idx = panelList.findIndex(p => p.id === panelId);
  if (idx !== -1) activatePanel(idx, true);
}

function toggleMode(targetMode) {
  if (targetMode === undefined) {
    isStepperMode = !isStepperMode;
  } else {
    isStepperMode = targetMode === 'stepper';
  }

  const btnMode = document.querySelector('#btn-toggle-mode');
  if (isStepperMode) {
    document.body.classList.remove('mode-scroll');
    document.body.classList.add('mode-stepper');
    if (btnMode) btnMode.textContent = 'Cuộn liên tục';
    activatePanel(currentIndex, true);
  } else {
    document.body.classList.remove('mode-stepper');
    document.body.classList.add('mode-scroll');
    if (btnMode) btnMode.textContent = 'Từng bước (Không lướt)';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  toggleMode('stepper');
  activatePanel(0, false);

  // Tab buttons click
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const lab = btn.getAttribute('data-lab');
      const idx = panelList.findIndex(p => p.lab === lab);
      if (idx !== -1) activatePanel(idx, true);
    });
  });

  // SVG link clicks
  document.querySelectorAll('svg a').forEach(a => {
    a.addEventListener('click', (e) => {
      const href = a.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        jumpToPanel(href.substring(1));
      }
    });
  });

  // Lightbox viewer
  const viewer = document.querySelector('#viewer');
  const viewerImg = viewer.querySelector('img');
  document.querySelectorAll('.image-card img, .gallery-item img, figure img').forEach(image => {
    image.tabIndex = 0;
    const openModal = () => {
      viewerImg.src = image.src;
      viewerImg.alt = image.alt || 'Hình ảnh thực nghiệm';
      viewer.showModal();
    };
    image.addEventListener('click', openModal);
    image.addEventListener('keydown', e => { if (e.key === 'Enter') openModal(); });
  });
  const closeBtn = viewer.querySelector('button');
  if (closeBtn) closeBtn.addEventListener('click', () => viewer.close());
  viewer.addEventListener('click', e => { if (e.target === viewer) viewer.close(); });

  // Keyboard navigation: ArrowRight, ArrowLeft, Space, PageDown, PageUp
  document.addEventListener('keydown', (e) => {
    if (viewer && viewer.open) {
      if (e.key === 'Escape') viewer.close();
      return;
    }
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      e.preventDefault();
      nextPanel();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      prevPanel();
    } else if (e.key === 'Home') {
      e.preventDefault();
      activatePanel(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      activatePanel(panelList.length - 1);
    }
  });
});

let allOpen = true;
function toggleAllDetails() {
  allOpen = !allOpen;
  document.querySelectorAll('details').forEach(d => d.open = allOpen);
  const btn = document.querySelector('#btn-toggle-details');
  if (btn) btn.textContent = allOpen ? 'Thu gọn chi tiết' : 'Mở tất cả chi tiết';
}
"""


def render_unified_full_html() -> str:
    """Generate edge-to-edge standalone HTML report combining Lab 02, 03, and 04 in full."""
    txt2 = (LABS_DIR / "lab02" / "LAB_02_REPORT.html").read_text(encoding="utf-8")
    txt3 = (LABS_DIR / "lab03" / "LAB_03_REPORT.html").read_text(encoding="utf-8")
    txt4 = (LABS_DIR / "lab04" / "LAB_04_REPORT.html").read_text(encoding="utf-8")

    # 1. Process Lab 02
    l2_map_match = re.search(r'(<section class="flow-map"[^>]*>.*?</section>)', txt2, re.S)
    l2_map = l2_map_match.group(1) if l2_map_match else ""

    l2_res_match = re.search(r'(<div id="results">.*?)(<dialog|<footer>|<script)', txt2, re.S)
    l2_results = l2_res_match.group(1).rstrip() if l2_res_match else ""
    if l2_results.endswith("</div></div>"):
        l2_results = l2_results[:-6]

    l2_content = f"""
    <section class="content-panel" id="lab02-flow">
      <div class="panel-header-banner banner-l2">
        <div class="eyebrow">LAB 02 · TOÁN TỬ KHÔNG GIAN</div>
        <h2>Sơ Đồ 5 Nhánh Thí Nghiệm Lab 02</h2>
        <p>Một ảnh đầu vào, năm nhánh khảo sát độc lập: Toán tử điểm ảnh YC1–4, Bộ lọc tuyến tính YC5–7, Phát hiện biên YC8, Tổng kernel YC9, Lọc phi tuyến YC10, và So sánh tổng hợp YC11.</p>
      </div>
      {l2_map}
    </section>
    <div class="lab02-panels">
      {l2_results}
    </div>
    """
    for id_name in ['flow', 'prepare', 'point', 'linear', 'edges', 'kernel', 'nonlinear', 'comparison', 'principles']:
        l2_content = re.sub(rf'id="{id_name}"', f'id="lab02-{id_name}"', l2_content)
        l2_content = re.sub(rf'href="#{id_name}"', f'href="#lab02-{id_name}"', l2_content)

    # 2. Process Lab 03
    l3_map_match = re.search(r'(<section class="flow-map"[^>]*>.*?</section>)', txt3, re.S)
    l3_map = l3_map_match.group(1) if l3_map_match else ""

    l3_stages = "".join(re.findall(r'(<section class="(?:stage|support)"[^>]*>.*?</section>)', txt3, re.S))

    l3_content = f"""
    <section class="content-panel" id="lab03-flow">
      <div class="panel-header-banner banner-l3">
        <div class="eyebrow">LAB 03 · TÁCH BIÊN CẢI TIẾN CANNY</div>
        <h2>Sơ Đồ 5 Bước Canny &amp; 2 Hướng Ứng Dụng</h2>
        <p>Các thí nghiệm độc lập: YC1 đối sánh hai thư viện, YC2–3 khảo sát tham số và tiền xử lý ảnh khó; YC4 tách hai nhánh ứng dụng độc lập tìm vùng và hình tròn đồng xu.</p>
      </div>
      {l3_map}
    </section>
    <div class="lab03-panels">
      {l3_stages}
    </div>
    """
    for id_name in ['flow', 'input', 'libraries', 'parameters', 'preprocess', 'regions', 'shapes', 'support']:
        l3_content = re.sub(rf'id="{id_name}"', f'id="lab03-{id_name}"', l3_content)
        l3_content = re.sub(rf'href="#{id_name}"', f'href="#lab03-{id_name}"', l3_content)

    # 3. Process Lab 04
    l4_map_match = re.search(r'(<section class="flow-map"[^>]*>.*?</section>)', txt4, re.S)
    l4_map = l4_map_match.group(1) if l4_map_match else ""

    l4_stages = "".join(re.findall(r'(<section class="(?:stage|support)"[^>]*>.*?</section>)', txt4, re.S))

    l4_content = f"""
    <section class="content-panel" id="lab04-flow">
      <div class="panel-header-banner banner-l4">
        <div class="eyebrow">LAB 04 · WAVELET HASH RGB 192-BIT</div>
        <h2>Sơ Đồ Biến Đổi Sóng Nhỏ &amp; Luồng Quyết Định</h2>
        <p>Luồng chính: Dữ liệu không rò rỉ &rarr; 2D DWT db4 Level 1 &rarr; Hamming &rarr; Ngưỡng Youden Calibration &rarr; Đánh giá Evaluation. Khảo sát độ bền vững và hai chuyên đề nâng cao.</p>
      </div>
      {l4_map}
    </section>
    <div class="lab04-panels">
      {l4_stages}
    </div>
    """
    for id_name in ['flow', 'dataset', 'hash', 'hamming', 'threshold', 'robustness', 'methods', 'retrieval', 'support']:
        l4_content = re.sub(rf'id="{id_name}"', f'id="lab04-{id_name}"', l4_content)
        l4_content = re.sub(rf'href="#{id_name}"', f'href="#lab04-{id_name}"', l4_content)

    # Wrap each section into a .content-panel
    def mark_panel(text: str, sid: str) -> str:
        def _repl(m: re.Match[str]) -> str:
            tag = m.group(0)
            if 'class="' in tag:
                return re.sub(r'class="([^"]*)"', r'class="content-panel \1"', tag)
            return tag.replace("<section", '<section class="content-panel"', 1)

        return re.sub(rf'<section\b[^>]*\bid="{sid}"[^>]*>', _repl, text)

    for sec_id in [
        "lab02-prepare",
        "lab02-point",
        "lab02-linear",
        "lab02-edges",
        "lab02-kernel",
        "lab02-nonlinear",
        "lab02-comparison",
        "lab02-principles",
        "lab03-input",
        "lab03-libraries",
        "lab03-parameters",
        "lab03-preprocess",
        "lab03-regions",
        "lab03-shapes",
        "lab03-support",
        "lab04-dataset",
        "lab04-hash",
        "lab04-hamming",
        "lab04-threshold",
        "lab04-robustness",
        "lab04-methods",
        "lab04-retrieval",
        "lab04-support",
    ]:
        l2_content = mark_panel(l2_content, sec_id)
        l3_content = mark_panel(l3_content, sec_id)
        l4_content = mark_panel(l4_content, sec_id)

    # Force all details to have open attribute by default for high visual impact
    l2_content = re.sub(r'<details(?![^>]*\bopen\b)', '<details open', l2_content)
    l3_content = re.sub(r'<details(?![^>]*\bopen\b)', '<details open', l3_content)
    l4_content = re.sub(r'<details(?![^>]*\bopen\b)', '<details open', l4_content)

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Báo Cáo Toàn Diện Thị Giác Máy Tính: Lab 02 · 03 · 04 | UTH</title>
  <style>{DASHBOARD_CSS}</style>
</head>
<body class="mode-stepper">
  <!-- Master Top Sticky Control Bar (Slim 48px: Zero Clutter, Straight to Point) -->
  <header class="master-topbar">
    <div class="topbar-brand">
      <span class="brand-tag">UTH CV</span>
      <span>BÁO CÁO LABS 02–04</span>
    </div>

    <!-- Master Lab Tabs -->
    <nav class="topbar-tabs" aria-label="Chọn chuyên đề">
      <button type="button" class="tab-btn active" data-lab="overview">🚀 Tổng Quan</button>
      <button type="button" class="tab-btn tab-l2" data-lab="lab02">⚡ Lab 02: Lọc Không Gian</button>
      <button type="button" class="tab-btn tab-l3" data-lab="lab03">🎯 Lab 03: Tách Biên Canny</button>
      <button type="button" class="tab-btn tab-l4" data-lab="lab04">🌊 Lab 04: Wavelet Hash RGB</button>
      <button type="button" class="tab-btn tab-mat" data-lab="matrix">📊 Ma Trận Nghiệm Thu</button>
    </nav>

    <!-- Stepper Navigation & Quick Jump Dropdown -->
    <div class="topbar-stepper">
      <button type="button" class="btn-step" onclick="prevPanel()" title="Phím tắt: &larr; (Mũi tên trái)">◀</button>
      <select id="step-selector" class="step-select" onchange="jumpToPanel(this.value)">
        <optgroup label="🚀 TỔNG QUAN">
          <option value="overview">Toàn Cảnh &amp; KPIs 3 Chuyên Đề</option>
        </optgroup>
        <optgroup label="⚡ LAB 02: LỌC KHÔNG GIAN &amp; ĐIỂM ẢNH">
          <option value="lab02-flow">Lab 02 · 🗺️ Sơ Đồ 5 Nhánh Thí Nghiệm</option>
          <option value="lab02-prepare">Lab 02 · 01 Chuẩn Bị Ảnh (BGR / Gray)</option>
          <option value="lab02-point">Lab 02 · 02 Điểm Ảnh YC1–4 (Sáng, Tương Phản, Âm Bản, Ngưỡng)</option>
          <option value="lab02-linear">Lab 02 · 03 Lọc Tuyến Tính YC5–7 (Mean, Gauss, Làm Nét)</option>
          <option value="lab02-edges">Lab 02 · 04 Đạo Hàm Biên YC8 (Sobel vs Prewitt)</option>
          <option value="lab02-kernel">Lab 02 · 05 Tổng Kernel YC9 (Vùng Phẳng &amp; Ảnh Màu)</option>
          <option value="lab02-nonlinear">Lab 02 · 06 Lọc Phi Tuyến YC10 (Median vs Bilateral)</option>
          <option value="lab02-comparison">Lab 02 · 07 So Sánh Tổng Hợp YC11 (Master 12 Bộ Lọc)</option>
          <option value="lab02-principles">Lab 02 · 08 Nguyên Lý &amp; Xác Thực 22 Tests</option>
        </optgroup>
        <optgroup label="🎯 LAB 03: TÁCH BIÊN CẢI TIẾN CANNY &amp; HÌNH HỌC">
          <option value="lab03-flow">Lab 03 · 🗺️ Sơ Đồ 5 Bước Canny &amp; 2 Hướng</option>
          <option value="lab03-input">Lab 03 · 01 Dữ Liệu Đầu Vào (Camera, Brick, Coins)</option>
          <option value="lab03-libraries">Lab 03 · 02 YC1 Hai Thư Viện (OpenCV vs Skimage)</option>
          <option value="lab03-parameters">Lab 03 · 03 YC2 Quét Tham Số (Sigma &amp; Ngưỡng Kép)</option>
          <option value="lab03-preprocess">Lab 03 · 04 YC3 Tiền Xử Lý Ảnh Khó (Nhiễu, Vân Gạch, Tương Phản)</option>
          <option value="lab03-regions">Lab 03 · 05 YC4.1 Phân Đoạn Vùng (Closing + Fill Holes)</option>
          <option value="lab03-shapes">Lab 03 · 06 YC4.2 Nhận Dạng Hình Tròn (Hough 24/24 Xu)</option>
          <option value="lab03-support">Lab 03 · 07 Câu Hỏi Nghiên Cứu Mở Rộng</option>
        </optgroup>
        <optgroup label="🌊 LAB 04: WAVELET HASH RGB 192-BIT">
          <option value="lab04-flow">Lab 04 · 🗺️ Sơ Đồ 2D DWT db4 &amp; Luồng Quyết Định</option>
          <option value="lab04-dataset">Lab 04 · 01 Dữ Liệu &amp; Gán Nhãn (26 Nguồn, 147 Cặp)</option>
          <option value="lab04-hash">Lab 04 · 02 YC1 Trích Xuất wHash RGB 192-bit</option>
          <option value="lab04-hamming">Lab 04 · 03 YC2 Ma Trận Hamming 26&times;26</option>
          <option value="lab04-threshold">Lab 04 · 04 YC4 Ngưỡng Youden (&tau;=18) &amp; Đánh Giá 98.77%</option>
          <option value="lab04-robustness">Lab 04 · 05 YC3 Khảo Sát Độ Bền Vững (7 Biến Đổi)</option>
          <option value="lab04-methods">Lab 04 · 06 Nâng Cao 1 So Sánh 4 Cấu Hình Wavelet</option>
          <option value="lab04-retrieval">Lab 04 · 07 Nâng Cao 2 Truy Vấn Top-3 (Top-1 100%)</option>
          <option value="lab04-support">Lab 04 · 08 Mở Rộng Cấu Trúc Tìm Kiếm (BK-Tree, MIH)</option>
        </optgroup>
        <optgroup label="📊 TỔNG KẾT &amp; ĐỐI SÁNH">
          <option value="matrix">Ma Trận Nghiệm Thu &amp; So Sánh Chéo 3 Lab</option>
        </optgroup>
      </select>
      <button type="button" class="btn-step" onclick="nextPanel()" title="Phím tắt: &rarr; hoặc Space">▶</button>
      <span class="step-counter" id="step-counter">01 / 28</span>
    </div>

    <!-- Actions -->
    <div class="topbar-actions">
      <button type="button" class="btn-action" id="btn-toggle-mode" onclick="toggleMode()">Cuộn liên tục</button>
      <button type="button" class="btn-action" id="btn-toggle-details" onclick="toggleAllDetails()">Thu gọn chi tiết</button>
      <button type="button" class="btn-action" onclick="window.print()">In / PDF</button>
    </div>
  </header>

  <main>
    <!-- Panel 01: Overview -->
    <section class="content-panel active" id="overview">
      <div class="panel-header-banner banner-overview">
        <div class="eyebrow">TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) — KHOA CÔNG NGHỆ THÔNG TIN</div>
        <h2>BÁO CÁO TOÀN DIỆN THỰC HÀNH THỊ GIÁC MÁY TÍNH<br/>CHUYÊN ĐỀ LIÊN HOÀN: LAB 02 – LAB 03 – LAB 04</h2>
        <p>Hệ thống thực nghiệm kết nối xuyên suốt 3 tầng kiến trúc thị giác máy tính: từ xử lý mức xám &amp; tích chập không gian (Lab 02), tối ưu vi sai tách biên đa hướng &amp; trích chọn hình học (Lab 03), đến biểu diễn nhận thức sóng nhỏ tần số không gian Wavelet Hash RGB 192-bit (Lab 04).</p>
      </div>

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

      <div class="takeaway">
        <strong>HƯỚNG DẪN XEM BÁO CÁO NHANH CHÓNG (KHÔNG CẦN LƯỚT):</strong>
        <ul style="margin:8px 0 0 20px;padding:0;">
          <li>Dùng <strong>Hộp Chọn Thí Nghiệm (trên thanh đầu trang)</strong> để nhảy thẳng đến bất kỳ bước nào trong 1 click.</li>
          <li>Dùng phím mũi tên <strong>&rarr; (Phải) / Phím Cách (Space)</strong> để tiến bước kế tiếp; phím <strong>&larr; (Trái)</strong> để lùi bước trước.</li>
          <li>Toàn bộ mã nguồn thực thi, bảng số liệu thực nghiệm và hình ảnh đều được <strong>mở trực quan đầy đủ</strong>.</li>
          <li>Bấm vào bất kỳ hình ảnh nào để <strong>phóng to xem chi tiết</strong> ở độ phân giải gốc trong Lightbox modal.</li>
        </ul>
      </div>

      <div style="margin-top:24px;display:flex;gap:12px;flex-wrap:wrap;">
        <button type="button" class="btn-step" style="width:auto;padding:8px 20px;font-size:14px;background:#0f172a;color:#fff;" onclick="nextPanel()">Bắt Đầu Xem Lab 02 ▶</button>
        <a href="../../notebooks/labs/lab_02.ipynb" class="btn-step" style="text-decoration:none;width:auto;padding:8px 16px;background:#1e293b;color:#f8fafc;">Notebook Lab 02 ↗</a>
        <a href="../../notebooks/labs/lab_03.ipynb" class="btn-step" style="text-decoration:none;width:auto;padding:8px 16px;background:#1e293b;color:#f8fafc;">Notebook Lab 03 ↗</a>
        <a href="../../notebooks/labs/lab_04.ipynb" class="btn-step" style="text-decoration:none;width:auto;padding:8px 16px;background:#1e293b;color:#f8fafc;">Notebook Lab 04 ↗</a>
      </div>
    </section>

    <!-- Lab 02 Panels -->
    {l2_content}

    <!-- Lab 03 Panels -->
    {l3_content}

    <!-- Lab 04 Panels -->
    {l4_content}

    <!-- Panel 28: Master Comparison Matrix -->
    <section class="content-panel" id="matrix">
      <div class="panel-header-banner banner-mat">
        <div class="eyebrow">TỔNG KẾT ĐỐI SÁNH KỸ THUẬT</div>
        <h2>Ma Trận Nghiệm Thu &amp; So Sánh Chéo 3 Chuyên Đề</h2>
        <p>Đối chiếu toàn diện các khía cạnh toán học, kích thước đặc trưng, độ phức tạp thuật toán và giá trị kỹ thuật thực tiễn được kiểm chứng qua 58 bài kiểm thử tự động.</p>
      </div>

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

  <!-- Lightbox Viewer Modal -->
  <dialog id="viewer" aria-label="Xem ảnh phóng to">
    <div class="dialog-header">
      <button type="button" aria-label="Đóng cửa sổ">Đóng &times;</button>
    </div>
    <img src="" alt="Chi tiết hình ảnh phóng to">
  </dialog>

  <script>{DASHBOARD_JS}</script>
</body>
</html>"""


def main() -> None:
    """Generate edge-to-edge interactive dashboard HTML report for Labs 02-04."""
    LABS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate Interactive Dashboard HTML Report (HTML ONLY)
    html_content = render_unified_full_html()
    out_unified_html = LABS_DIR / "LABS_02_03_04_REPORT.html"
    out_unified_html.write_text(html_content, encoding="utf-8")
    print(f"✔ Generated Master Visual HTML Report: {out_unified_html} ({out_unified_html.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
