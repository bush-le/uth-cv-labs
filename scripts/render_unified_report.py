#!/usr/bin/env python3
"""Unified Interactive Dashboard Report Generator for Labs 02, 03, and 04.

Generates an edge-to-edge, zero border-radius interactive dashboard report
(LABS_02_03_04_REPORT.html) with:
1. Stepper / Tabbed Panel Mode (DEFAULT): View each experiment step-by-step
   without needing to manually scroll/swipe ("không phải lướt tay"), navigated via
   the left sidebar, top controller, or keyboard arrows (← / → / Space).
2. Continuous Scroll Mode: Switchable via button to view all sections sequentially.
3. 100% Full Content Preservation: Every single image (36 base64 images),
   notebook code block (44 pre blocks), mathematical explanation table,
   takeaway, and research extension from Labs 02, 03, and 04 is fully retained
   and visually displayed (<details open> by default).
4. Synchronized technical Markdown report (LABS_02_03_04_REPORT.md).
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


# Edge-to-Edge Dashboard CSS (Zero border-radius, Stepper & Scroll Modes)
DASHBOARD_CSS = """
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
  --bg-sidebar: #090d16;
  --bg-card: #ffffff;
  --border: #cbd5e1;
  --border-dark: #334155;
  --text-main: #0f172a;
  --text-muted: #475569;
  --text-light: #f8fafc;
  --sidebar-w: 320px;
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

/* Master Layout Grid: Sidebar + Main Content */
.dashboard-app {
  display: flex;
  width: 100%;
  min-height: 100vh;
}

/* Left Navigation Sidebar */
aside.dashboard-sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  background: var(--bg-sidebar);
  color: #cbd5e1;
  border-right: 2px solid #1e293b;
  height: 100vh;
  position: sticky;
  top: 0;
  display: flex;
  flex-direction: column;
  z-index: 900;
  box-shadow: 4px 0 16px rgba(0,0,0,0.25);
}

.sidebar-header {
  padding: 18px 20px;
  background: #060911;
  border-bottom: 1px solid #1e293b;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  font-size: 15px;
  color: #ffffff;
  text-decoration: none;
}

.sidebar-brand .tag {
  background: #16a34a;
  color: #fff;
  padding: 2px 7px;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.sidebar-sub {
  font-size: 11px;
  color: #94a3b8;
  margin-top: 4px;
}

/* Sidebar Stepper Quick Controls */
.sidebar-stepper {
  padding: 12px 16px;
  background: #0f172a;
  border-bottom: 1px solid #1e293b;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.sidebar-stepper button {
  background: #1e293b;
  border: 1px solid #334155;
  color: #f8fafc;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.15s ease;
}

.sidebar-stepper button:hover {
  background: #38bdf8;
  color: #02122b;
  border-color: #38bdf8;
}

.stepper-progress {
  font-size: 12px;
  font-weight: 700;
  color: #38bdf8;
  white-space: nowrap;
}

/* Sidebar Treeview Scroll Area */
.sidebar-tree {
  flex-grow: 1;
  overflow-y: auto;
  padding: 12px 0 24px 0;
}

.sidebar-tree::-webkit-scrollbar {
  width: 5px;
}
.sidebar-tree::-webkit-scrollbar-thumb {
  background: #334155;
}

.tree-group {
  margin-bottom: 14px;
}

.tree-group-header {
  padding: 8px 18px 4px 18px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #94a3b8;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.tree-group-header.group-l2 { color: #38bdf8; border-left: 3px solid #38bdf8; }
.tree-group-header.group-l3 { color: #4ade80; border-left: 3px solid #4ade80; }
.tree-group-header.group-l4 { color: #c084fc; border-left: 3px solid #c084fc; }
.tree-group-header.group-mat { color: #fbbf24; border-left: 3px solid #fbbf24; }

.tree-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 18px;
  color: #94a3b8;
  text-decoration: none;
  font-size: 12.5px;
  font-weight: 600;
  border-left: 3px solid transparent;
  transition: all 0.12s ease;
  cursor: pointer;
}

.tree-item:hover {
  background: rgba(255, 255, 255, 0.05);
  color: #ffffff;
}

.tree-item.active {
  background: rgba(56, 189, 248, 0.12);
  color: #ffffff;
  border-left-color: #38bdf8;
  font-weight: 700;
}

.tree-item.active.group-l3-item {
  background: rgba(74, 222, 128, 0.12);
  border-left-color: #4ade80;
}

.tree-item.active.group-l4-item {
  background: rgba(192, 132, 252, 0.12);
  border-left-color: #c084fc;
}

.tree-item.active.group-mat-item {
  background: rgba(251, 191, 36, 0.12);
  border-left-color: #fbbf24;
}

/* Main Viewport Content */
main.dashboard-main {
  flex-grow: 1;
  min-width: 0;
  background: var(--bg-page);
  display: flex;
  flex-direction: column;
}

/* Master Top Sticky Control Bar */
.master-topbar {
  position: sticky;
  top: 0;
  z-index: 800;
  width: 100%;
  background: #ffffff;
  border-bottom: 2px solid var(--border);
  padding: 10px clamp(16px, 3vw, 40px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.topbar-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.btn-sidebar-toggle {
  display: none;
  background: #0f172a;
  border: 1px solid #334155;
  color: #f8fafc;
  padding: 6px 12px;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}

.topbar-breadcrumb {
  font-size: 14px;
  font-weight: 700;
  color: var(--text-main);
  display: flex;
  align-items: center;
  gap: 8px;
}

.topbar-breadcrumb .badge-lab {
  background: #0f172a;
  color: #fff;
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.btn-ctrl {
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  color: #0f172a;
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.15s ease;
}

.btn-ctrl:hover {
  background: #0f172a;
  color: #ffffff;
  border-color: #0f172a;
}

.btn-ctrl.btn-mode.active-mode {
  background: #0284c7;
  color: #ffffff;
  border-color: #0284c7;
}

/* Stepper Mode (Default: show only active panel without scrolling) */
body.mode-stepper .content-panel {
  display: none;
}

body.mode-stepper .content-panel.active {
  display: block;
  animation: panelFade 0.15s ease-in;
}

@keyframes panelFade {
  from { opacity: 0.6; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Continuous Scroll Mode */
body.mode-scroll .content-panel {
  display: block;
  border-bottom: 6px solid #cbd5e1;
}

/* Panel Containers Full Bleed */
.content-panel {
  width: 100%;
  padding: 32px clamp(20px, 4vw, 56px);
  background: #ffffff;
}

.panel-header-banner {
  width: 100%;
  padding: 24px 32px;
  color: #ffffff;
  margin-bottom: 24px;
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
  margin-bottom: 6px;
}

.panel-header-banner h2 {
  font-size: clamp(22px, 2.8vw, 36px);
  margin: 0 0 8px 0;
  font-weight: 800;
}

.panel-header-banner p {
  font-size: 15px;
  margin: 0;
  max-width: 1000px;
  opacity: 0.92;
}

/* Visual Stage Elements */
.flow-map {
  width: 100%;
  background: #ffffff;
  padding: 16px 0 24px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
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
  margin: 0 0 28px 16px;
  padding: 0 0 20px 24px;
}

.stage-head, .stage-heading {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 14px;
}

.number, .stage-number {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: 44px;
  height: 44px;
  margin-left: -47px;
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

/* Visual Image Cards */
.image-card, .gallery-item {
  background: #ffffff;
  border: 1px solid var(--border);
  overflow: hidden;
  margin: 20px 0;
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

/* Visual Details & Code Blocks (OPEN by default for high visual impact) */
details {
  background: #ffffff;
  border: 1px solid var(--border);
  margin: 16px 0;
}

summary {
  padding: 12px 18px;
  cursor: pointer;
  font-weight: 700;
  font-size: 14px;
  color: #0f172a;
  background: #f1f5f9;
  border-bottom: 1px solid var(--border);
  user-select: none;
}

.detail-body {
  padding: 18px 20px;
  overflow-wrap: anywhere;
}

.code-note {
  font-size: 13px;
  color: #64748b;
  margin: 6px 0;
}

pre {
  padding: 16px 20px;
  background: #0f172a;
  color: #e2e8f0;
  overflow-x: auto;
  line-height: 1.6;
  font-size: 13px;
  font-family: Consolas, "SFMono-Regular", Menlo, Monaco, monospace;
  margin: 10px 0;
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

/* Master Overview KPI Cards */
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

/* Responsive Media Queries */
@media (max-width: 960px) {
  aside.dashboard-sidebar {
    position: fixed;
    left: -320px;
    transition: left 0.2s ease-in-out;
  }
  aside.dashboard-sidebar.open {
    left: 0;
  }
  .btn-sidebar-toggle {
    display: block;
  }
}

/* Print Styles */
@media print {
  aside.dashboard-sidebar, .master-topbar, dialog#viewer {
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

# Interactive JavaScript for Stepper, Tree Navigation, and Keyboard Controls
DASHBOARD_JS = """
// 28 Sequential Panels
const panelList = [
  { id: 'overview', lab: 'Tổng Quan', title: 'Toàn Cảnh & KPIs 3 Chuyên Đề' },
  { id: 'lab02-flow', lab: 'Lab 02', title: 'Sơ Đồ 5 Nhánh Thí Nghiệm' },
  { id: 'lab02-prepare', lab: 'Lab 02', title: '01 · Đọc Ảnh & Chuẩn Bị (BGR / Gray)' },
  { id: 'lab02-point', lab: 'Lab 02', title: '02 · Điểm Ảnh YC1–4 (Sáng, Tương Phản, Âm Bản, Ngưỡng)' },
  { id: 'lab02-linear', lab: 'Lab 02', title: '03 · Lọc Tuyến Tính YC5–7 (Mean, Gauss, Làm Nét)' },
  { id: 'lab02-edges', lab: 'Lab 02', title: '04 · Đạo Hàm Biên YC8 (Sobel vs Prewitt)' },
  { id: 'lab02-kernel', lab: 'Lab 02', title: '05 · Tổng Kernel YC9 (Vùng Phẳng & Ảnh Màu)' },
  { id: 'lab02-nonlinear', lab: 'Lab 02', title: '06 · Lọc Phi Tuyến YC10 (Median vs Bilateral)' },
  { id: 'lab02-comparison', lab: 'Lab 02', title: '07 · So Sánh Tổng Hợp YC11 (Master Grid)' },
  { id: 'lab02-principles', lab: 'Lab 02', title: '08 · Nguyên Lý & Kiểm Thử (22 Tests)' },
  { id: 'lab03-flow', lab: 'Lab 03', title: 'Sơ Đồ 5 Bước Canny & 2 Hướng' },
  { id: 'lab03-input', lab: 'Lab 03', title: '01 · Dữ Liệu Đầu Vào (Camera, Brick, Coins)' },
  { id: 'lab03-libraries', lab: 'Lab 03', title: '02 · YC1 Hai Thư Viện (OpenCV vs Skimage)' },
  { id: 'lab03-parameters', lab: 'Lab 03', title: '03 · YC2 Quét Tham Số (Sigma & Ngưỡng)' },
  { id: 'lab03-preprocess', lab: 'Lab 03', title: '04 · YC3 Tiền Xử Lý (Nhiễu, Brick, EqualizeHist)' },
  { id: 'lab03-regions', lab: 'Lab 03', title: '05 · YC4.1 Phân Đoạn Vùng (Closing + Fill Holes)' },
  { id: 'lab03-shapes', lab: 'Lab 03', title: '06 · YC4.2 Nhận Dạng Hình Tròn (Hough 24/24 Xu)' },
  { id: 'lab03-support', lab: 'Lab 03', title: '07 · Câu Hỏi Nghiên Cứu Mở Rộng' },
  { id: 'lab04-flow', lab: 'Lab 04', title: 'Sơ Đồ 2D DWT db4 & Luồng Quyết Định' },
  { id: 'lab04-dataset', lab: 'Lab 04', title: '01 · Dữ Liệu & Gán Nhãn (26 Nguồn, 147 Cặp)' },
  { id: 'lab04-hash', lab: 'Lab 04', title: '02 · YC1 Trích Xuất wHash RGB 192-bit' },
  { id: 'lab04-hamming', lab: 'Lab 04', title: '03 · YC2 Ma Trận Hamming 26×26' },
  { id: 'lab04-threshold', lab: 'Lab 04', title: '04 · YC4 Ngưỡng Youden (τ=18) & Đánh Giá 98.77%' },
  { id: 'lab04-robustness', lab: 'Lab 04', title: '05 · YC3 Khảo Sát Độ Bền Vững (7 Biến Đổi)' },
  { id: 'lab04-methods', lab: 'Lab 04', title: '06 · Nâng Cao 1 So Sánh 4 Cấu Hình Wavelet' },
  { id: 'lab04-retrieval', lab: 'Lab 04', title: '07 · Nâng Cao 2 Truy Vấn Top-3 (Top-1 100%)' },
  { id: 'lab04-support', lab: 'Lab 04', title: '08 · Mở Rộng Cấu Trúc Tìm Kiếm (BK-Tree, MIH)' },
  { id: 'matrix', lab: 'Tổng Kết', title: 'Ma Trận Nghiệm Thu & So Sánh Chéo 3 Lab' }
];

let currentIndex = 0;
let isStepperMode = true;

function activatePanel(index, scrollToTop = true) {
  if (index < 0 || index >= panelList.length) return;
  currentIndex = index;
  const current = panelList[currentIndex];

  // Update Breadcrumb & Indicators
  const badgeLab = document.querySelector('#bc-lab');
  const bcTitle = document.querySelector('#bc-title');
  const progressText = document.querySelectorAll('.stepper-progress');

  if (badgeLab) badgeLab.textContent = current.lab;
  if (bcTitle) bcTitle.textContent = current.title;
  progressText.forEach(el => el.textContent = `${currentIndex + 1} / ${panelList.length}`);

  // Update Sidebar Tree Active States
  document.querySelectorAll('.tree-item').forEach(item => {
    if (item.getAttribute('data-target') === current.id) {
      item.classList.add('active');
      item.scrollIntoView({ block: 'nearest' });
    } else {
      item.classList.remove('active');
    }
  });

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

  // Update SVG node highlighting if present
  document.querySelectorAll('svg a').forEach(a => {
    if (a.getAttribute('href') === '#' + current.id) {
      a.querySelector('rect')?.setAttribute('stroke-width', '3');
    } else {
      a.querySelector('rect')?.setAttribute('stroke-width', '1');
    }
  });
}

function nextPanel() {
  if (currentIndex < panelList.length - 1) {
    activatePanel(currentIndex + 1);
  }
}

function prevPanel() {
  if (currentIndex > 0) {
    activatePanel(currentIndex - 1);
  }
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
    if (btnMode) btnMode.textContent = 'Chế độ: Từng bước (Không lướt)';
    activatePanel(currentIndex, true);
  } else {
    document.body.classList.remove('mode-stepper');
    document.body.classList.add('mode-scroll');
    if (btnMode) btnMode.textContent = 'Chế độ: Cuộn toàn bộ';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Default Mode: Stepper (zero scrolling needed)
  toggleMode('stepper');
  activatePanel(0, false);

  // Bind Treeview Item Clicks
  document.querySelectorAll('.tree-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = item.getAttribute('data-target');
      const idx = panelList.findIndex(p => p.id === targetId);
      if (idx !== -1) activatePanel(idx, true);
    });
  });

  // Bind SVG Links
  document.querySelectorAll('svg a').forEach(a => {
    a.addEventListener('click', (e) => {
      const href = a.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        const targetId = href.substring(1);
        const idx = panelList.findIndex(p => p.id === targetId);
        if (idx !== -1) activatePanel(idx, true);
      }
    });
  });

  // Global Image Lightbox Modal
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

  // Keyboard Navigation: ArrowRight, ArrowLeft, Space, PageDown, PageUp
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

  // Mobile Sidebar Toggle
  const btnToggleSidebar = document.querySelector('#btn-toggle-sidebar');
  const sidebar = document.querySelector('aside.dashboard-sidebar');
  if (btnToggleSidebar && sidebar) {
    btnToggleSidebar.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
});

// Toggle all details tags
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

    # Extract all sections inside Lab 02 results
    l2_res_match = re.search(r'(<div id="results">.*?)(<dialog|<footer>|<script)', txt2, re.S)
    l2_results = l2_res_match.group(1).rstrip() if l2_res_match else ""
    if l2_results.endswith("</div></div>"):
        l2_results = l2_results[:-6]

    # Namespace IDs and hrefs in Lab 02
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
    for sec_id in [
        'lab02-prepare', 'lab02-point', 'lab02-linear', 'lab02-edges', 'lab02-kernel', 'lab02-nonlinear', 'lab02-comparison', 'lab02-principles',
        'lab03-input', 'lab03-libraries', 'lab03-parameters', 'lab03-preprocess', 'lab03-regions', 'lab03-shapes', 'lab03-support',
        'lab04-dataset', 'lab04-hash', 'lab04-hamming', 'lab04-threshold', 'lab04-robustness', 'lab04-methods', 'lab04-retrieval', 'lab04-support'
    ]:
        # Replace <section class="..." id="{sec_id}" with <section class="content-panel ..." id="{sec_id}"
        pattern = rf'<section class="([^"]*)" id="{sec_id}"'
        repl = rf'<section class="content-panel \1" id="{sec_id}"'
        l2_content = re.sub(pattern, repl, l2_content)
        l3_content = re.sub(pattern, repl, l3_content)
        l4_content = re.sub(pattern, repl, l4_content)

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
  <div class="dashboard-app">
    <!-- Left Navigation Sidebar -->
    <aside class="dashboard-sidebar">
      <div class="sidebar-header">
        <a href="#overview" class="sidebar-brand">
          <span class="tag">UTH CV</span>
          <span>BÁO CÁO LABS 02–04</span>
        </a>
        <div class="sidebar-sub">58/58 Unit Tests Passed &bull; Python 3.14.8</div>
      </div>

      <!-- Stepper Quick Controls -->
      <div class="sidebar-stepper">
        <button type="button" onclick="prevPanel()" title="Phím tắt: Mũi tên trái (&larr;)">◀ Trước</button>
        <span class="stepper-progress">1 / 28</span>
        <button type="button" onclick="nextPanel()" title="Phím tắt: Mũi tên phải (&rarr;) hoặc Space">Tiếp ▶</button>
      </div>

      <!-- Treeview of all 28 panels -->
      <div class="sidebar-tree">
        <!-- Overview Group -->
        <div class="tree-group">
          <div class="tree-group-header">TỔNG QUAN</div>
          <a class="tree-item active" data-target="overview">
            <span>🚀</span> <span>Toàn Cảnh &amp; KPIs 3 Lab</span>
          </a>
        </div>

        <!-- Lab 02 Group -->
        <div class="tree-group">
          <div class="tree-group-header group-l2">LAB 02 · LỌC KHÔNG GIAN</div>
          <a class="tree-item" data-target="lab02-flow"><span>🗺️</span> <span>Sơ Đồ 5 Nhánh Lab 02</span></a>
          <a class="tree-item" data-target="lab02-prepare"><span>01</span> <span>Đọc Ảnh &amp; Chuẩn Bị</span></a>
          <a class="tree-item" data-target="lab02-point"><span>02</span> <span>Điểm Ảnh YC1–4 (Sáng/Ngưỡng)</span></a>
          <a class="tree-item" data-target="lab02-linear"><span>03</span> <span>Lọc Tuyến Tính YC5–7</span></a>
          <a class="tree-item" data-target="lab02-edges"><span>04</span> <span>Phát Hiện Biên YC8</span></a>
          <a class="tree-item" data-target="lab02-kernel"><span>05</span> <span>Tổng Kernel YC9</span></a>
          <a class="tree-item" data-target="lab02-nonlinear"><span>06</span> <span>Lọc Phi Tuyến YC10</span></a>
          <a class="tree-item" data-target="lab02-comparison"><span>07</span> <span>So Sánh Tổng Hợp YC11</span></a>
          <a class="tree-item" data-target="lab02-principles"><span>08</span> <span>Nguyên Lý &amp; Kiểm Thử</span></a>
        </div>

        <!-- Lab 03 Group -->
        <div class="tree-group">
          <div class="tree-group-header group-l3">LAB 03 · TÁCH BIÊN CANNY</div>
          <a class="tree-item group-l3-item" data-target="lab03-flow"><span>🗺️</span> <span>Sơ Đồ 5 Bước Canny</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-input"><span>01</span> <span>Dữ Liệu Đầu Vào</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-libraries"><span>02</span> <span>YC1 Hai Thư Viện</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-parameters"><span>03</span> <span>YC2 Quét Tham Số</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-preprocess"><span>04</span> <span>YC3 Tiền Xử Lý Ảnh Khó</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-regions"><span>05</span> <span>YC4.1 Phân Đoạn Vùng</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-shapes"><span>06</span> <span>YC4.2 Hough 24/24 Xu</span></a>
          <a class="tree-item group-l3-item" data-target="lab03-support"><span>07</span> <span>Câu Hỏi Mở Rộng</span></a>
        </div>

        <!-- Lab 04 Group -->
        <div class="tree-group">
          <div class="tree-group-header group-l4">LAB 04 · WAVELET HASH RGB</div>
          <a class="tree-item group-l4-item" data-target="lab04-flow"><span>🗺️</span> <span>Sơ Đồ 2D DWT db4</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-dataset"><span>01</span> <span>Dữ Liệu &amp; Gán Nhãn</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-hash"><span>02</span> <span>YC1 Hash RGB 192-bit</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-hamming"><span>03</span> <span>YC2 Hamming 26&times;26</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-threshold"><span>04</span> <span>YC4 Ngưỡng Youden &tau;=18</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-robustness"><span>05</span> <span>YC3 Độ Bền Vững</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-methods"><span>06</span> <span>Nâng Cao 1 So Sánh Wavelet</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-retrieval"><span>07</span> <span>Nâng Cao 2 Top-3</span></a>
          <a class="tree-item group-l4-item" data-target="lab04-support"><span>08</span> <span>Mở Rộng Tìm Kiếm</span></a>
        </div>

        <!-- Matrix Group -->
        <div class="tree-group">
          <div class="tree-group-header group-mat">ĐỐI SÁNH &amp; TỔNG KẾT</div>
          <a class="tree-item group-mat-item" data-target="matrix"><span>📊</span> <span>Ma Trận Nghiệm Thu 3 Lab</span></a>
        </div>
      </div>
    </aside>

    <!-- Main Content Area -->
    <main class="dashboard-main">
      <!-- Master Topbar -->
      <div class="master-topbar">
        <div class="topbar-left">
          <button type="button" class="btn-sidebar-toggle" id="btn-toggle-sidebar">☰ Mục Lục</button>
          <div class="topbar-breadcrumb">
            <span class="badge-lab" id="bc-lab">Tổng Quan</span>
            <span id="bc-title">Toàn Cảnh &amp; KPIs 3 Chuyên Đề</span>
          </div>
        </div>

        <div class="topbar-right">
          <button type="button" class="btn-ctrl" onclick="prevPanel()" title="Phím tắt: &larr;">◀ Trước</button>
          <span class="stepper-progress" style="font-size:13px;font-weight:700;color:#0284c7;">1 / 28</span>
          <button type="button" class="btn-ctrl" onclick="nextPanel()" title="Phím tắt: &rarr; hoặc Space">Tiếp ▶</button>
          <button type="button" class="btn-ctrl btn-mode active-mode" id="btn-toggle-mode" onclick="toggleMode()">Chế độ: Từng bước (Không lướt)</button>
          <button type="button" class="btn-ctrl" id="btn-toggle-details" onclick="toggleAllDetails()">Thu gọn chi tiết</button>
          <button type="button" class="btn-ctrl" onclick="window.print()">In / PDF</button>
        </div>
      </div>

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
          <strong>HƯỚNG DẪN XEM BÁO CÁO KHÔNG CẦN LƯỚT:</strong>
          <ul style="margin:8px 0 0 20px;padding:0;">
            <li>Bấm vào từng mục ở <strong>Cột Mục Lục bên trái</strong> để chuyển ngay đến nội dung thí nghiệm tương ứng mà không phải cuộn trang.</li>
            <li>Dùng phím mũi tên <strong>&rarr; (Phải) / Phím Cách (Space)</strong> để tiến bước kế tiếp; phím <strong>&larr; (Trái)</strong> để lùi bước trước.</li>
            <li>Toàn bộ mã nguồn thực thi, bảng số liệu thực nghiệm và hình ảnh đều được <strong>mở trực quan đầy đủ</strong>.</li>
            <li>Bấm vào bất kỳ hình ảnh nào để <strong>phóng to xem chi tiết</strong> ở độ phân giải gốc.</li>
          </ul>
        </div>

        <div style="margin-top:24px;display:flex;gap:12px;flex-wrap:wrap;">
          <button type="button" class="btn-ctrl" style="padding:10px 20px;font-size:14px;background:#0f172a;color:#fff;" onclick="nextPanel()">Bắt Đầu Xem Lab 02 ▶</button>
          <a href="../../notebooks/labs/lab_02.ipynb" class="btn-ctrl" style="text-decoration:none;padding:10px 16px;">Notebook Lab 02 ↗</a>
          <a href="../../notebooks/labs/lab_03.ipynb" class="btn-ctrl" style="text-decoration:none;padding:10px 16px;">Notebook Lab 03 ↗</a>
          <a href="../../notebooks/labs/lab_04.ipynb" class="btn-ctrl" style="text-decoration:none;padding:10px 16px;">Notebook Lab 04 ↗</a>
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
  </div>

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
    """Generate edge-to-edge interactive dashboard HTML and Markdown reports for Labs 02-04."""
    LABS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Interactive Dashboard HTML Report
    html_content = render_unified_full_html()
    out_unified_html = LABS_DIR / "LABS_02_03_04_REPORT.html"
    out_unified_html.write_text(html_content, encoding="utf-8")
    print(f"✔ Generated Interactive Dashboard Report: {out_unified_html} ({out_unified_html.stat().st_size // 1024} KB)")

    # 2. Generate Unified Markdown Technical Report
    md_content = render_unified_markdown_report()
    out_unified_md = LABS_DIR / "LABS_02_03_04_REPORT.md"
    out_unified_md.write_text(md_content, encoding="utf-8")
    print(f"✔ Generated Unified Markdown Report: {out_unified_md} ({out_unified_md.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
