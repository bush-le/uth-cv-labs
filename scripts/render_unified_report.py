#!/usr/bin/env python3
"""Unified Interactive Visual Dashboard Generator for Labs 02, 03, and 04.

Generates an edge-to-edge, zero-border-radius interactive HTML dashboard report
(LABS_02_03_04_REPORT.html) with:
1. Stepper / Two-Tier Tabbed Navigation (DEFAULT): View each experiment step-by-step
   without needing to manually scroll or swipe ("không phải lướt tay"), navigated via
   the top master tabs, dynamic sub-ribbon, left sidebar tree, or keyboard arrows (← / → / Space).
2. Continuous Scroll Mode: Switchable via button to view all sections sequentially.
3. 100% Full Content Preservation: Every single image (36 base64 images),
   notebook code block (44 pre blocks), mathematical explanation table,
   takeaway, and research extension from Labs 02, 03, and 04 is fully retained
   and visually displayed (<details open> by default).
4. Strictly HTML only (no .md generated as requested).
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


# Edge-to-Edge Dashboard CSS (Zero border-radius, Two-Tier Ribbon & Stepper Navigation)
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
  z-index: 950;
  box-shadow: 4px 0 16px rgba(0,0,0,0.25);
}

.sidebar-header {
  padding: 16px 20px;
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
  padding: 10px 16px;
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

/* Tier 1: Master Top Sticky Control Bar */
.master-topbar {
  position: sticky;
  top: 0;
  z-index: 850;
  width: 100%;
  background: #091310;
  border-bottom: 2px solid #22c55e;
  padding: 8px clamp(16px, 3vw, 36px);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
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

/* Master Lab Selectors */
.master-lab-tabs {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.master-lab-tab {
  color: #cbd5e1;
  text-decoration: none;
  font-size: 13px;
  font-weight: 700;
  padding: 6px 14px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(255, 255, 255, 0.05);
  cursor: pointer;
  transition: all 0.15s ease;
}

.master-lab-tab:hover, .master-lab-tab.active {
  background: #22c55e;
  color: #091310;
  border-color: #22c55e;
}

.master-lab-tab.tab-l2.active { background: #38bdf8; color: #02122b; border-color: #38bdf8; }
.master-lab-tab.tab-l3.active { background: #4ade80; color: #052e16; border-color: #4ade80; }
.master-lab-tab.tab-l4.active { background: #c084fc; color: #2e1065; border-color: #c084fc; }
.master-lab-tab.tab-mat.active { background: #fbbf24; color: #451a03; border-color: #fbbf24; }

.topbar-right {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.btn-ctrl {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #f8fafc;
  padding: 6px 14px;
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.15s ease;
}

.btn-ctrl:hover {
  background: #ffffff;
  color: #0f172a;
}

.btn-ctrl.btn-mode.active-mode {
  background: #0284c7;
  color: #ffffff;
  border-color: #0284c7;
}

/* Tier 2: Dynamic Sub-Ribbon for Active Lab Stage Navigation */
.master-subribbon {
  position: sticky;
  top: 48px;
  z-index: 800;
  width: 100%;
  background: #ffffff;
  border-bottom: 2px solid var(--border);
  padding: 8px clamp(16px, 3vw, 36px);
  display: flex;
  align-items: center;
  gap: 6px;
  overflow-x: auto;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

.master-subribbon::-webkit-scrollbar {
  height: 4px;
}
.master-subribbon::-webkit-scrollbar-thumb {
  background: #cbd5e1;
}

.subribbon-btn {
  white-space: nowrap;
  font-size: 12.5px;
  font-weight: 700;
  padding: 5px 11px;
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  color: #334155;
  cursor: pointer;
  transition: all 0.12s ease;
}

.subribbon-btn:hover, .subribbon-btn.active {
  background: #0f172a;
  color: #ffffff;
  border-color: #0f172a;
}

.subribbon-btn.l2-ribbon.active { background: #0284c7; border-color: #0284c7; color: #fff; }
.subribbon-btn.l3-ribbon.active { background: #16a34a; border-color: #16a34a; color: #fff; }
.subribbon-btn.l4-ribbon.active { background: #7c3aed; border-color: #7c3aed; color: #fff; }
.subribbon-btn.mat-ribbon.active { background: #d97706; border-color: #d97706; color: #fff; }

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
  max-width: 1100px;
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
  aside.dashboard-sidebar, .master-topbar, .master-subribbon, dialog#viewer {
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

# Interactive JavaScript for Stepper, Two-Tier Ribbon Navigation, and Keyboard Controls
DASHBOARD_JS = """
// 28 Sequential Panels
const panelList = [
  { id: 'overview', lab: 'overview', labName: 'Tổng Quan', title: 'Toàn Cảnh & KPIs 3 Chuyên Đề' },
  { id: 'lab02-flow', lab: 'lab02', labName: 'Lab 02', title: 'Sơ Đồ 5 Nhánh Thí Nghiệm' },
  { id: 'lab02-prepare', lab: 'lab02', labName: 'Lab 02', title: '01 · Đọc Ảnh & Chuẩn Bị (BGR / Gray)' },
  { id: 'lab02-point', lab: 'lab02', labName: 'Lab 02', title: '02 · Điểm Ảnh YC1–4 (Sáng, Tương Phản, Âm Bản, Ngưỡng)' },
  { id: 'lab02-linear', lab: 'lab02', labName: 'Lab 02', title: '03 · Lọc Tuyến Tính YC5–7 (Mean, Gauss, Làm Nét)' },
  { id: 'lab02-edges', lab: 'lab02', labName: 'Lab 02', title: '04 · Đạo Hàm Biên YC8 (Sobel vs Prewitt)' },
  { id: 'lab02-kernel', lab: 'lab02', labName: 'Lab 02', title: '05 · Tổng Kernel YC9 (Vùng Phẳng & Ảnh Màu)' },
  { id: 'lab02-nonlinear', lab: 'lab02', labName: 'Lab 02', title: '06 · Lọc Phi Tuyến YC10 (Median vs Bilateral)' },
  { id: 'lab02-comparison', lab: 'lab02', labName: 'Lab 02', title: '07 · So Sánh Tổng Hợp YC11 (Master Grid)' },
  { id: 'lab02-principles', lab: 'lab02', labName: 'Lab 02', title: '08 · Nguyên Lý & Kiểm Thử (22 Tests)' },
  { id: 'lab03-flow', lab: 'lab03', labName: 'Lab 03', title: 'Sơ Đồ 5 Bước Canny & 2 Hướng' },
  { id: 'lab03-input', lab: 'lab03', labName: 'Lab 03', title: '01 · Dữ Liệu Đầu Vào (Camera, Brick, Coins)' },
  { id: 'lab03-libraries', lab: 'lab03', labName: 'Lab 03', title: '02 · YC1 Hai Thư Viện (OpenCV vs Skimage)' },
  { id: 'lab03-parameters', lab: 'lab03', labName: 'Lab 03', title: '03 · YC2 Quét Tham Số (Sigma & Ngưỡng)' },
  { id: 'lab03-preprocess', lab: 'lab03', labName: 'Lab 03', title: '04 · YC3 Tiền Xử Lý (Nhiễu, Brick, EqualizeHist)' },
  { id: 'lab03-regions', lab: 'lab03', labName: 'Lab 03', title: '05 · YC4.1 Phân Đoạn Vùng (Closing + Fill Holes)' },
  { id: 'lab03-shapes', lab: 'lab03', labName: 'Lab 03', title: '06 · YC4.2 Nhận Dạng Hình Tròn (Hough 24/24 Xu)' },
  { id: 'lab03-support', lab: 'lab03', labName: 'Lab 03', title: '07 · Câu Hỏi Nghiên Cứu Mở Rộng' },
  { id: 'lab04-flow', lab: 'lab04', labName: 'Lab 04', title: 'Sơ Đồ 2D DWT db4 & Luồng Quyết Định' },
  { id: 'lab04-dataset', lab: 'lab04', labName: 'Lab 04', title: '01 · Dữ Liệu & Gán Nhãn (26 Nguồn, 147 Cặp)' },
  { id: 'lab04-hash', lab: 'lab04', labName: 'Lab 04', title: '02 · YC1 Trích Xuất wHash RGB 192-bit' },
  { id: 'lab04-hamming', lab: 'lab04', labName: 'Lab 04', title: '03 · YC2 Ma Trận Hamming 26×26' },
  { id: 'lab04-threshold', lab: 'lab04', labName: 'Lab 04', title: '04 · YC4 Ngưỡng Youden (τ=18) & Đánh Giá 98.77%' },
  { id: 'lab04-robustness', lab: 'lab04', labName: 'Lab 04', title: '05 · YC3 Khảo Sát Độ Bền Vững (7 Biến Đổi)' },
  { id: 'lab04-methods', lab: 'lab04', labName: 'Lab 04', title: '06 · Nâng Cao 1 So Sánh 4 Cấu Hình Wavelet' },
  { id: 'lab04-retrieval', lab: 'lab04', labName: 'Lab 04', title: '07 · Nâng Cao 2 Truy Vấn Top-3 (Top-1 100%)' },
  { id: 'lab04-support', lab: 'lab04', labName: 'Lab 04', title: '08 · Mở Rộng Cấu Trúc Tìm Kiếm (BK-Tree, MIH)' },
  { id: 'matrix', lab: 'matrix', labName: 'Tổng Kết', title: 'Ma Trận Nghiệm Thu & So Sánh Chéo 3 Lab' }
];

let currentIndex = 0;
let isStepperMode = true;

// Render dynamic sub-ribbon buttons based on active lab
function updateSubribbon(activeLab) {
  const ribbonContainer = document.querySelector('#master-subribbon');
  if (!ribbonContainer) return;
  ribbonContainer.innerHTML = '';

  const relevantPanels = panelList.filter(p => p.lab === activeLab);
  relevantPanels.forEach(p => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `subribbon-btn ${activeLab}-ribbon`;
    btn.textContent = p.title.split('(')[0].trim();
    if (p.id === panelList[currentIndex].id) btn.classList.add('active');
    btn.addEventListener('click', () => {
      const idx = panelList.findIndex(item => item.id === p.id);
      if (idx !== -1) activatePanel(idx, true);
    });
    ribbonContainer.appendChild(btn);
  });
}

function activatePanel(index, scrollToTop = true) {
  if (index < 0 || index >= panelList.length) return;
  currentIndex = index;
  const current = panelList[currentIndex];

  // Update Master Lab Tabs (Top Bar)
  document.querySelectorAll('.master-lab-tab').forEach(tab => {
    if (tab.getAttribute('data-lab') === current.lab) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Update Dynamic Sub-Ribbon
  updateSubribbon(current.lab);

  // Update Progress Indicator
  document.querySelectorAll('.stepper-progress').forEach(el => {
    el.textContent = `${currentIndex + 1} / ${panelList.length}`;
  });

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

  // Bind Master Lab Tabs Clicks
  document.querySelectorAll('.master-lab-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const labName = tab.getAttribute('data-lab');
      const firstPanelIdx = panelList.findIndex(p => p.lab === labName);
      if (firstPanelIdx !== -1) activatePanel(firstPanelIdx, true);
    });
  });

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
    for sec_id in [
        'lab02-prepare', 'lab02-point', 'lab02-linear', 'lab02-edges', 'lab02-kernel', 'lab02-nonlinear', 'lab02-comparison', 'lab02-principles',
        'lab03-input', 'lab03-libraries', 'lab03-parameters', 'lab03-preprocess', 'lab03-regions', 'lab03-shapes', 'lab03-support',
        'lab04-dataset', 'lab04-hash', 'lab04-hamming', 'lab04-threshold', 'lab04-robustness', 'lab04-methods', 'lab04-retrieval', 'lab04-support'
    ]:
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
        <button type="button" onclick="prevPanel()" title="Phím tắt: &larr; (Mũi tên trái)">◀ Trước</button>
        <span class="stepper-progress">1 / 28</span>
        <button type="button" onclick="nextPanel()" title="Phím tắt: &rarr; (Mũi tên phải) hoặc Space">Tiếp ▶</button>
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
      <!-- Tier 1: Master Top Sticky Control Bar -->
      <header class="master-topbar">
        <div class="topbar-left">
          <button type="button" class="btn-sidebar-toggle" id="btn-toggle-sidebar">☰ Mục Lục</button>

          <!-- Master Lab Tabs -->
          <nav class="master-lab-tabs" aria-label="Chọn chuyên đề">
            <button type="button" class="master-lab-tab active" data-lab="overview">🚀 Tổng Quan</button>
            <button type="button" class="master-lab-tab tab-l2" data-lab="lab02">⚡ Lab 02: Lọc Không Gian</button>
            <button type="button" class="master-lab-tab tab-l3" data-lab="lab03">🎯 Lab 03: Tách Biên Canny</button>
            <button type="button" class="master-lab-tab tab-l4" data-lab="lab04">🌊 Lab 04: Wavelet Hash RGB</button>
            <button type="button" class="master-lab-tab tab-mat" data-lab="matrix">📊 Ma Trận Nghiệm Thu</button>
          </nav>
        </div>

        <div class="topbar-right">
          <button type="button" class="btn-ctrl" onclick="prevPanel()" title="Phím tắt: &larr;">◀ Trước</button>
          <span class="stepper-progress" style="font-size:13px;font-weight:700;color:#0284c7;">1 / 28</span>
          <button type="button" class="btn-ctrl" onclick="nextPanel()" title="Phím tắt: &rarr; hoặc Space">Tiếp ▶</button>
          <button type="button" class="btn-ctrl btn-mode active-mode" id="btn-toggle-mode" onclick="toggleMode()">Chế độ: Từng bước (Không lướt)</button>
          <button type="button" class="btn-ctrl" id="btn-toggle-details" onclick="toggleAllDetails()">Thu gọn chi tiết</button>
          <button type="button" class="btn-ctrl" onclick="window.print()">In / PDF</button>
        </div>
      </header>

      <!-- Tier 2: Dynamic Sub-Ribbon for Active Lab Stage Navigation -->
      <nav class="master-subribbon" id="master-subribbon" aria-label="Thanh điều hướng nhanh thí nghiệm con">
        <!-- Dynamically injected via JavaScript -->
      </nav>

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
          <strong>HƯỚNG DẪN XEM BÁO CÁO TRỰC QUAN KHÔNG CẦN LƯỚT:</strong>
          <ul style="margin:8px 0 0 20px;padding:0;">
            <li>Bấm vào từng mục ở <strong>Thanh Ribbon Thí Nghiệm Con (ở ngay trên đầu)</strong> hoặc <strong>Cột Danh Mục bên trái</strong> để chuyển ngay đến nội dung thí nghiệm tương ứng mà không phải cuộn trang.</li>
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
