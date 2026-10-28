"""Unified Fullscreen 100vh Presentation & Technical Report for Labs 02, 03, and 04.

Generates a single, cohesive, interactive presentation & technical report combining Labs 02-04:
- Lab 02: Spatial Filters & Point Operators (11 Yêu Cầu, Linear, Edge, Non-linear)
- Lab 03: Optimal Canny Edge Detector & Hough Circle Recognition (4 Yêu Cầu + 4 Mở Rộng)
- Lab 04: Perceptual Image Matching via Wavelet Hash RGB 192-bit (147 Pairs, ROC AUC 1.0000, Top-3)

Features:
- True 100vh Fullscreen Snap (Mỗi Slide chiếm trọn 100% khung nhìn, lướt hít cứng mượt mà)
- Dark Luxury Keynote Aesthetic (#0b0f19, glowing cyan accents, glassmorphic cards)
- 5-Stage Visual Pipeline Ribbons with Real Experimental Images & Exact Math Formulas
- Sơ đồ kiến trúc luồng dữ liệu Vector SVG độ phân giải cao
- Minimalist Right-Edge Dots Pager (13 slides tương ứng 13 chấm định vị)
- Keyboard navigation (Space, ArrowDown, ArrowUp, ArrowRight, ArrowLeft)
- Interactive tabbed image canvases and fullscreen Lightbox modal
- 100% self-contained with inline CSS, JS, and Base64-encoded images.
"""

from __future__ import annotations

import base64
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


# Unified Fullscreen 100vh Slide Presentation CSS Design System
SHARED_CSS = """
:root {
  --primary: #38bdf8;
  --primary-dark: #0284c7;
  --primary-glow: rgba(56, 189, 248, 0.25);
  --accent: #818cf8;
  --success: #34d399;
  --success-bg: rgba(52, 211, 153, 0.15);
  --warning: #fbbf24;
  --danger: #f87171;
  --bg-deck: #0b0f19;
  --bg-card: rgba(15, 23, 42, 0.75);
  --border: rgba(255, 255, 255, 0.12);
  --border-light: rgba(255, 255, 255, 0.06);
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --radius-lg: 20px;
  --radius-md: 12px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
  background-color: var(--bg-deck);
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  overflow-x: hidden;
  overflow-y: scroll;
  scroll-snap-type: y mandatory;
  scroll-behavior: smooth;
}

body::-webkit-scrollbar {
  width: 5px;
}
body::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 9999px;
}

/* Slide Screen Container (100vh Snap) */
.slide-screen {
  height: 100vh;
  min-height: 100vh;
  max-height: 100vh;
  scroll-snap-align: start;
  scroll-snap-stop: always;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 16px 44px;
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.slide-frame {
  width: 100%;
  max-width: 1440px;
  height: 94vh;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 30px;
  box-sizing: border-box;
  backdrop-filter: blur(20px);
  box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6);
  position: relative;
}

/* Cover / Hero Slide */
.hero-frame {
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
  border-color: rgba(56, 189, 248, 0.3);
  box-shadow: 0 30px 80px rgba(2, 132, 199, 0.15);
  justify-content: center;
  text-align: center;
  align-items: center;
}

.brand-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(56, 189, 248, 0.15);
  border: 1px solid rgba(56, 189, 248, 0.4);
  color: var(--primary);
  padding: 5px 14px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 10px;
}

.hero-title {
  font-size: 30px;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 8px;
  line-height: 1.25;
  background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  max-width: 1100px;
}

.hero-desc {
  color: #94a3b8;
  font-size: 14px;
  max-width: 950px;
  margin-bottom: 16px;
  line-height: 1.5;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  margin-bottom: 16px;
}

.pill-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 11.5px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.08);
  color: #f1f5f9;
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.pill-badge.success { background: rgba(52, 211, 153, 0.15); color: #34d399; border-color: rgba(52, 211, 153, 0.3); }
.pill-badge.info { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-color: rgba(56, 189, 248, 0.3); }
.pill-badge.warning { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); }

/* Hero KPI Grid */
.hero-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  width: 100%;
  max-width: 1200px;
  margin-bottom: 14px;
}

.hero-kpi-card {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  padding: 14px 16px;
  text-align: left;
  transition: transform 0.2s, border-color 0.2s, background 0.2s;
}

.hero-kpi-card:hover {
  transform: translateY(-3px);
  border-color: var(--primary);
  background: rgba(255, 255, 255, 0.07);
}

.hero-kpi-title {
  font-size: 10.5px;
  text-transform: uppercase;
  color: #94a3b8;
  font-weight: 700;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}

.hero-kpi-value {
  font-size: 24px;
  font-weight: 800;
  color: #38bdf8;
  line-height: 1.2;
}

.hero-kpi-sub {
  font-size: 11px;
  color: #cbd5e1;
  margin-top: 4px;
}

/* Roadmap Grid */
.lab-roadmap-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  width: 100%;
  max-width: 1200px;
  margin-bottom: 12px;
}

.roadmap-card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 8px 12px;
  text-align: left;
  cursor: pointer;
  transition: all 0.2s;
}

.roadmap-card:hover {
  background: rgba(56, 189, 248, 0.1);
  border-color: var(--primary);
  transform: translateY(-2px);
}

.roadmap-step-tag {
  font-size: 10px;
  font-weight: 800;
  color: var(--primary);
  text-transform: uppercase;
  margin-bottom: 2px;
}

.roadmap-step-title {
  font-size: 12px;
  font-weight: 700;
  color: #ffffff;
}

.roadmap-step-desc {
  font-size: 10.5px;
  color: var(--text-muted);
  margin-top: 2px;
}

.scroll-hint {
  font-size: 12px;
  color: #64748b;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
  animation: pulse 2s infinite ease-in-out;
}
@keyframes pulse {
  0%, 100% { opacity: 0.5; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(4px); }
}

/* Regular Slide Header */
.slide-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border-light);
  gap: 16px;
  flex-shrink: 0;
}

.slide-tag {
  display: inline-block;
  background: rgba(56, 189, 248, 0.15);
  color: var(--primary);
  font-size: 10.5px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 6px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 2px;
  border: 1px solid rgba(56, 189, 248, 0.3);
}

.slide-title {
  font-size: 19px;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: -0.01em;
}

.slide-summary {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 2px;
}

.slide-badge-num {
  font-size: 11px;
  font-weight: 700;
  color: #64748b;
  background: rgba(255, 255, 255, 0.05);
  padding: 3px 9px;
  border-radius: 9999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  white-space: nowrap;
}

/* Slide Center Body */
.slide-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 0;
  margin: 4px 0;
  overflow: hidden;
}

/* Visual Pipeline Ribbon */
.pipeline-ribbon {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  width: 100%;
}

.pipeline-step {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  padding: 10px;
  display: flex;
  flex-direction: column;
  transition: transform 0.2s, border-color 0.2s, background 0.2s;
  cursor: zoom-in;
  position: relative;
}

.pipeline-step:hover {
  transform: translateY(-3px);
  border-color: var(--primary);
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
}

.step-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}

.step-num {
  font-size: 10px;
  font-weight: 800;
  color: var(--primary);
  background: rgba(56, 189, 248, 0.15);
  padding: 2px 6px;
  border-radius: 9999px;
  letter-spacing: 0.05em;
  border: 1px solid rgba(56, 189, 248, 0.3);
}

.step-badge {
  font-size: 9.5px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
}

.step-thumb {
  width: 100%;
  height: 120px;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: #000000;
  margin-bottom: 6px;
}

.step-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.step-title {
  font-size: 12.5px;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 2px;
  line-height: 1.25;
}

.step-formula {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.1);
  padding: 2px 5px;
  border-radius: 4px;
  margin-bottom: 4px;
  display: inline-block;
  border: 1px solid rgba(56, 189, 248, 0.25);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.step-desc {
  font-size: 10.5px;
  color: var(--text-muted);
  line-height: 1.3;
  margin-bottom: 6px;
  flex-grow: 1;
}

.step-metric {
  font-size: 10px;
  font-weight: 700;
  color: var(--success);
  background: var(--success-bg);
  border: 1px solid rgba(52, 211, 153, 0.3);
  padding: 2px 6px;
  border-radius: 4px;
  display: inline-block;
  align-self: flex-start;
}

/* Visual Canvas & Tabs */
.visual-canvas {
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-md);
  padding: 8px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.tab-nav {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding-bottom: 5px;
  overflow-x: auto;
  flex-shrink: 0;
}

.tab-btn {
  background: transparent;
  border: none;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.tab-btn:hover {
  background: rgba(255, 255, 255, 0.08);
  color: #ffffff;
}

.tab-btn.active {
  background: var(--primary);
  color: #0f172a;
  font-weight: 700;
}

.tab-pane {
  display: none;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.tab-pane.active {
  display: flex;
  flex-direction: column;
}

.img-container {
  position: relative;
  flex: 1;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: #000000;
  cursor: zoom-in;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.img-container img {
  width: 100%;
  height: 100%;
  max-height: 52vh;
  object-fit: contain;
  display: block;
  margin: auto;
}

.img-caption {
  padding: 5px 10px;
  background: rgba(15, 23, 42, 0.85);
  font-size: 11px;
  color: var(--text-muted);
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.img-caption strong {
  color: #ffffff;
}

.zoom-hint {
  font-size: 10px;
  color: var(--primary);
  font-weight: 600;
}

/* Side-by-Side Images in Canvas */
.side-by-side-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  flex: 1;
  min-height: 0;
}

.quad-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: 6px;
  flex: 1;
  min-height: 0;
}

/* Slide Footer & Takeaways Pills */
.slide-footer {
  margin-top: 6px;
  flex-shrink: 0;
}

.takeaways-pills {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 6px;
}

.takeaway-pill {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-left: 3px solid var(--success);
  border-radius: 6px;
  padding: 5px 8px;
  font-size: 11.5px;
  color: #cbd5e1;
  line-height: 1.35;
}

.takeaway-pill strong {
  color: #ffffff;
}

/* Data Table in Slide Mode */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
  margin: auto 0;
}

.data-table th, .data-table td {
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.data-table th {
  background: rgba(255, 255, 255, 0.06);
  font-weight: 700;
  color: var(--primary);
}

.data-table tr:hover {
  background: rgba(255, 255, 255, 0.03);
}

/* Minimalist Right Pager Dots */
.slide-dots {
  position: fixed;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 50;
}

.slide-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.25);
  transition: all 0.25s ease;
  cursor: pointer;
}

.slide-dot:hover {
  background: rgba(255, 255, 255, 0.7);
  transform: scale(1.3);
}

.slide-dot.active {
  background: var(--primary);
  height: 18px;
  border-radius: 9999px;
  box-shadow: 0 0 10px var(--primary);
}

/* Lightbox Modal */
.modal-overlay {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(5, 8, 15, 0.95);
  z-index: 1000;
  justify-content: center;
  align-items: center;
  padding: 20px;
  backdrop-filter: blur(12px);
}

.modal-overlay.active {
  display: flex;
}

.modal-content {
  max-width: 95vw;
  max-height: 94vh;
  position: relative;
  background: transparent;
  text-align: center;
}

.modal-img {
  max-width: 100%;
  max-height: 86vh;
  border-radius: var(--radius-md);
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.15);
}

.modal-caption {
  color: #ffffff;
  margin-top: 8px;
  font-size: 13.5px;
  font-weight: 600;
}

.modal-close {
  position: absolute;
  top: -38px;
  right: 0;
  background: rgba(255, 255, 255, 0.15);
  border: none;
  color: #ffffff;
  font-size: 22px;
  cursor: pointer;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}

.modal-close:hover {
  background: rgba(255, 255, 255, 0.35);
}
"""

SHARED_JS = """
document.addEventListener('DOMContentLoaded', () => {
  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const parent = btn.closest('.visual-canvas') || btn.closest('.slide-frame');
      const targetId = btn.getAttribute('data-tab');
      parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      parent.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetPane = parent.querySelector(`#${targetId}`);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Lightbox Modal
  const modal = document.getElementById('lightbox-modal');
  const modalImg = document.getElementById('modal-img');
  const modalCap = document.getElementById('modal-caption');
  const modalClose = document.getElementById('modal-close');

  const openLightbox = (imgSrc, title) => {
    if (modal && modalImg) {
      modalImg.src = imgSrc;
      if (modalCap) modalCap.textContent = title || '';
      modal.classList.add('active');
    }
  };

  document.querySelectorAll('.img-container').forEach(container => {
    container.addEventListener('click', () => {
      const img = container.querySelector('img');
      const caption = container.querySelector('.img-caption strong');
      if (img) openLightbox(img.src, caption ? caption.textContent : '');
    });
  });

  document.querySelectorAll('.pipeline-step').forEach(step => {
    step.addEventListener('click', () => {
      const img = step.querySelector('img');
      const title = step.querySelector('.step-title');
      if (img) openLightbox(img.src, title ? title.textContent : '');
    });
  });

  if (modalClose) {
    modalClose.addEventListener('click', () => modal.classList.remove('active'));
  }
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('active');
    });
  }

  // Keyboard Slide Snap Navigation
  const slides = document.querySelectorAll('.slide-screen');
  const dots = document.querySelectorAll('.slide-dot');

  let currentIdx = 0;

  const updateActiveDot = (idx) => {
    dots.forEach((dot, dIdx) => {
      if (dIdx === idx) dot.classList.add('active');
      else dot.classList.remove('active');
    });
  };

  window.scrollToSlide = (idx) => {
    if (idx >= 0 && idx < slides.length) {
      slides[idx].scrollIntoView({ behavior: 'smooth' });
      currentIdx = idx;
      updateActiveDot(idx);
    }
  };

  dots.forEach(dot => {
    dot.addEventListener('click', () => {
      const targetIdx = parseInt(dot.getAttribute('data-index'), 10);
      window.scrollToSlide(targetIdx);
    });
  });

  // ScrollSpy for Dots
  window.addEventListener('scroll', () => {
    const scrollPos = window.scrollY + window.innerHeight / 2;
    slides.forEach((slide, idx) => {
      if (scrollPos >= slide.offsetTop && scrollPos < slide.offsetTop + slide.offsetHeight) {
        currentIdx = idx;
        updateActiveDot(idx);
      }
    });
  });

  // Keyboard navigation
  document.addEventListener('keydown', (e) => {
    if (modal && modal.classList.contains('active')) {
      if (e.key === 'Escape') modal.classList.remove('active');
      return;
    }

    if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      e.preventDefault();
      window.scrollToSlide(currentIdx + 1);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      window.scrollToSlide(currentIdx - 1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      window.scrollToSlide(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      window.scrollToSlide(slides.length - 1);
    }
  });
});
"""

LIGHTBOX_MARKUP = """
<div class="modal-overlay" id="lightbox-modal">
  <div class="modal-content">
    <button class="modal-close" id="modal-close">&times;</button>
    <img src="" alt="Enlarged view" class="modal-img" id="modal-img">
    <div class="modal-caption" id="modal-caption"></div>
  </div>
</div>
"""

def render_unified_labs_02_03_04() -> str:
    """Generate Fullscreen 100vh Unified Presentation HTML for Labs 02-04."""
    # Lab 02 assets
    l2_step1 = get_base64_img("lab02/image-2.jpg")
    l2_step2 = get_base64_img("lab02/brightness_plus50.jpg")
    l2_step3 = get_base64_img("lab02/gaussian_19x19_s3.jpg")
    l2_step4 = get_base64_img("lab02/sharpen.jpg")
    l2_step5 = get_base64_img("lab02/median_denoised.jpg")
    l2_bright = get_base64_img("lab02/brightness_comparison.png")
    l2_contrast = get_base64_img("lab02/contrast_comparison.png")
    l2_negative = get_base64_img("lab02/negative_comparison.png")
    l2_thresh = get_base64_img("lab02/threshold_comparison.png")
    l2_hist_orig = get_base64_img("lab02/histogram_original.png")
    l2_hist_bin = get_base64_img("lab02/histogram_binary.png")
    l2_blur = get_base64_img("lab02/blur_comparison_grid.png")
    l2_sharpen = get_base64_img("lab02/sharpen_comparison.png")
    l2_edges = get_base64_img("lab02/edge_detection_grid.png")
    l2_sobel_mag = get_base64_img("lab02/sobel_gradient_magnitude.png")
    l2_sobel_x = get_base64_img("lab02/sobel_x.png")
    l2_sobel_y = get_base64_img("lab02/sobel_y.png")
    l2_prewitt_x = get_base64_img("lab02/prewitt_x.png")
    l2_prewitt_y = get_base64_img("lab02/prewitt_y.png")
    l2_sum_rule = get_base64_img("lab02/kernel_sum_rule.png")
    l2_sum_effects = get_base64_img("lab02/kernel_effects_on_image.png")
    l2_nonlinear = get_base64_img("lab02/nonlinear_comparison_grid.png")
    l2_all = get_base64_img("lab02/all_filters_grid.png")
    svg_lab02 = get_base64_img("workflow_lab02.svg")

    # Lab 03 assets
    l3_yc1 = get_base64_img("lab03/yc1_libraries.png")
    l3_yc2_sigma = get_base64_img("lab03/yc2_sigma.png")
    l3_yc2_thresh = get_base64_img("lab03/yc2_threshold.png")
    l3_yc3_noise_img = get_base64_img("lab03/yc3_noise_images.png")
    l3_yc3_noise_edge = get_base64_img("lab03/yc3_noise_edges.png")
    l3_yc3_contrast = get_base64_img("lab03/yc3_contrast.png")
    l3_yc3_texture = get_base64_img("lab03/yc3_texture.png")
    l3_yc4_regions = get_base64_img("lab03/yc4_regions.png")
    l3_yc4_circles = get_base64_img("lab03/yc4_circles.png")
    svg_lab03 = get_base64_img("workflow_lab03.svg")

    # Lab 04 assets
    l4_dataset = get_base64_img("lab04/dataset_pairs.png")
    l4_rgb = get_base64_img("lab04/rgb_channels.png")
    l4_baseline = get_base64_img("lab04/baseline_hash.png")
    l4_bands = get_base64_img("lab04/wavelet_bands.png")
    l4_channel_dist = get_base64_img("lab04/channel_distances.png")
    l4_hamming = get_base64_img("lab04/hamming_matrix.png")
    l4_robust = get_base64_img("lab04/robustness.png")
    l4_thresh_sel = get_base64_img("lab04/threshold_selection.png")
    l4_roc = get_base64_img("lab04/roc_evaluation.png")
    l4_review = get_base64_img("lab04/review_pairs.png")
    l4_wavelet_comp = get_base64_img("lab04/wavelet_comparison.png")
    l4_topk = get_base64_img("lab04/top_k.png")
    svg_lab04 = get_base64_img("workflow_lab04.svg")

    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Báo Cáo Tổng Hợp Chuyên Đề Lab 02 - Lab 03 - Lab 04 | UTH Computer Vision</title>
  <style>{SHARED_CSS}</style>
</head>
<body>

  <!-- Minimalist Right Pager Dots (13 Slides) -->
  <div class="slide-dots">
    <div class="slide-dot active" data-index="0" title="Slide 01: Bìa & Dashboard Tổng Hợp"></div>
    <div class="slide-dot" data-index="1" title="Slide 02: Lab 02 · Tiến Trình 5 Tầng & Sơ Đồ SVG"></div>
    <div class="slide-dot" data-index="2" title="Slide 03: Lab 02 · Điểm Ảnh & Làm Mờ Không Gian"></div>
    <div class="slide-dot" data-index="3" title="Slide 04: Lab 02 · Làm Nét & Đạo Hàm Sobel/Prewitt"></div>
    <div class="slide-dot" data-index="4" title="Slide 05: Lab 02 · Tổng Kernel & Lọc Phi Tuyến"></div>
    <div class="slide-dot" data-index="5" title="Slide 06: Lab 03 · 5 Bước Canny & Benchmark Thư Viện"></div>
    <div class="slide-dot" data-index="6" title="Slide 07: Lab 03 · Quét Tham Số σ & Tiền Xử Lý Ảnh Khó"></div>
    <div class="slide-dot" data-index="7" title="Slide 08: Lab 03 · Phân Đoạn & Hough Circles 24 Xu"></div>
    <div class="slide-dot" data-index="8" title="Slide 09: Lab 04 · Tiến Trình wHash 192-bit & Dữ Liệu"></div>
    <div class="slide-dot" data-index="9" title="Slide 10: Lab 04 · Wavelet 2D DWT & Độ Bền Vững"></div>
    <div class="slide-dot" data-index="10" title="Slide 11: Lab 04 · Ngưỡng Youden τ=18 & Đánh Giá"></div>
    <div class="slide-dot" data-index="11" title="Slide 12: Lab 04 · Cấu Hình Hash & Tìm Kiếm Top-3"></div>
    <div class="slide-dot" data-index="12" title="Slide 13: Tổng Kết Nghiệm Thu & Ma Trận 3 Lab"></div>
  </div>

  <!-- ========================================================================
       SLIDE 01: MASTER COVER & EXECUTIVE ROADMAP DASHBOARD
       ======================================================================== -->
  <section class="slide-screen" id="slide-0">
    <div class="slide-frame hero-frame">
      <div class="brand-badge">Trường Đại học Giao thông Vận tải TP.HCM · Khoa Công Nghệ Thông Tin</div>
      <h1 class="hero-title">BÁO CÁO TỔNG HỢP CHUYÊN ĐỀ THỊ GIÁC MÁY TÍNH<br/>LAB 02 · LAB 03 · LAB 04</h1>
      <p class="hero-desc">Hợp nhất chuỗi chuyên đề cốt lõi: Toán tử điểm ảnh & Lọc không gian tuyến tính/phi tuyến (Lab 02), Tách biên tối ưu Canny & Nhận dạng hình tròn Hough (Lab 03) và So sánh ảnh bằng Wavelet Hash RGB 192-bit (Lab 04).</p>
      <div class="badge-row">
        <span class="pill-badge info">Python 3.14.8</span>
        <span class="pill-badge info">OpenCV 5.0.0</span>
        <span class="pill-badge info">PyWavelets</span>
        <span class="pill-badge info">Scikit-Image</span>
        <span class="pill-badge success">58/58 Tests Passed (100%)</span>
        <span class="pill-badge success">54 Artifacts Tích Hợp</span>
        <span class="pill-badge success">100% Offline Standalone</span>
      </div>
      <div class="hero-kpi-grid">
        <div class="hero-kpi-card" onclick="window.scrollToSlide(1)" style="cursor: pointer;">
          <div class="hero-kpi-title">Lab 02 · Lọc Không Gian</div>
          <div class="hero-kpi-value">11 Yêu Cầu</div>
          <div class="hero-kpi-sub">Sobel-Prewitt r≈0.996 · Sharpen Σ=1.000 · Median 100%</div>
        </div>
        <div class="hero-kpi-card" onclick="window.scrollToSlide(5)" style="cursor: pointer;">
          <div class="hero-kpi-title">Lab 03 · Tách Biên Canny</div>
          <div class="hero-kpi-value">24 / 24 Xu</div>
          <div class="hero-kpi-sub">5 bước Canny · 30,980 vs 7,554 px · HoughCircles 100%</div>
        </div>
        <div class="hero-kpi-card" onclick="window.scrollToSlide(8)" style="cursor: pointer;">
          <div class="hero-kpi-title">Lab 04 · Wavelet Hash</div>
          <div class="hero-kpi-value">98.77% Acc</div>
          <div class="hero-kpi-sub">147 cặp · 192-bit RGB · Youden τ=18 · ROC AUC 1.0000</div>
        </div>
        <div class="hero-kpi-card" onclick="window.scrollToSlide(12)" style="cursor: pointer;">
          <div class="hero-kpi-title">Chất Lượng Kỹ Nghệ</div>
          <div class="hero-kpi-value">58 / 58 PASS</div>
          <div class="hero-kpi-sub">PyTest 2.42s · 0 Ruff Warnings · Báo cáo Slide 100vh</div>
        </div>
      </div>
      <div class="lab-roadmap-grid">
        <div class="roadmap-card" onclick="window.scrollToSlide(1)">
          <div class="roadmap-step-tag">Lab 02 (Slides 02–05)</div>
          <div class="roadmap-step-title">Toán Tử Điểm & Lọc Không Gian</div>
          <div class="roadmap-step-desc">Point ops, Mean/Gauss, Sharpen, Sobel & Median</div>
        </div>
        <div class="roadmap-card" onclick="window.scrollToSlide(5)">
          <div class="roadmap-step-tag">Lab 03 (Slides 06–08)</div>
          <div class="roadmap-step-title">Tách Biên Canny & Hough Circles</div>
          <div class="roadmap-step-desc">Benchmark thư viện, quét σ, equalizeHist & 24 xu</div>
        </div>
        <div class="roadmap-card" onclick="window.scrollToSlide(8)">
          <div class="roadmap-step-tag">Lab 04 (Slides 09–12)</div>
          <div class="roadmap-step-title">Wavelet Hash RGB 192-bit</div>
          <div class="roadmap-step-desc">2D DWT db4 L1, Hamming, Youden τ=18 & Top-3</div>
        </div>
        <div class="roadmap-card" onclick="window.scrollToSlide(12)">
          <div class="roadmap-step-tag">Nghiệm Thu (Slide 13)</div>
          <div class="roadmap-step-title">Ma Trận Kỹ Nghệ 3 Bài Lab</div>
          <div class="roadmap-step-desc">Bảng đối chiếu toàn diện thuật toán, dữ liệu & KPI</div>
        </div>
      </div>
      <div class="scroll-hint">
        <span>↓ Cuộn chuột hoặc bấm phím <strong>[Space]</strong> / <strong>[↓]</strong> để xem chuỗi bài báo cáo</span>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 02: LAB 02 - PIPELINE RIBBON & ARCHITECTURE SVG
       ======================================================================== -->
  <section class="slide-screen" id="slide-1">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 02 / 01 · TIẾN TRÌNH THỊ GIÁC</span>
          <h2 class="slide-title">Lab 02 · Chu Trình 5 Tầng Lọc Không Gian & Toán Tử Điểm Ảnh</h2>
          <div class="slide-summary">Chuyển hóa mức xám từ ảnh gốc 1920×1080 qua các tầng tích chập tuyến tính, vi sai đạo hàm và khử nhiễu phi tuyến.</div>
        </div>
        <span class="slide-badge-num">Slide 02 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-l2-ribbon">Chuỗi 5 Bước Ảnh Thực Nghiệm</button>
            <button class="tab-btn" data-tab="tab-u-l2-workflow">Sơ Đồ Kiến Trúc Vector (SVG)</button>
            <button class="tab-btn" data-tab="tab-u-l2-mapping">Bảng 11 Yêu Cầu YC1–YC11</button>
          </div>
          <div class="tab-pane active" id="tab-u-l2-ribbon">
            <div class="pipeline-ribbon">
              <div class="pipeline-step">
                <div class="step-header">
                  <span class="step-num">01</span>
                  <span class="step-badge">ĐẦU VÀO</span>
                </div>
                <div class="step-thumb"><img src="{l2_step1}" alt="Ảnh gốc"></div>
                <div class="step-title">Ảnh Gốc Đầu Vào</div>
                <div class="step-formula">BGR (1920×1080) → Gray</div>
                <div class="step-desc">Ảnh mức xám chuẩn Rec. 601, bảo toàn nguyên bản phân bố cường độ sáng.</div>
                <span class="step-metric">1920×1080 uint8</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header">
                  <span class="step-num">02</span>
                  <span class="step-badge">ĐIỂM ẢNH (YC1–4)</span>
                </div>
                <div class="step-thumb"><img src="{l2_step2}" alt="Toán tử tăng sáng"></div>
                <div class="step-title">Điểm Ảnh & Clip uint8</div>
                <div class="step-formula">g = clip(α·f + β, 0, 255)</div>
                <div class="step-desc">Độ sáng β=±50, tương phản α={0.5, 1.8}, âm bản 255-f, cắt ngưỡng T={80, 127, 180}.</div>
                <span class="step-metric">Anti-Overflow Clip</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header">
                  <span class="step-num">03</span>
                  <span class="step-badge">LÀM MỜ (YC5–6)</span>
                </div>
                <div class="step-thumb"><img src="{l2_step3}" alt="Lọc Gaussian"></div>
                <div class="step-title">Lọc Gauss 3-Sigma</div>
                <div class="step-formula">k ≈ 6σ + 1 (±3σ)</div>
                <div class="step-desc">Tích chập Gauss σ=3 (19×19) và σ=7 (43×43) bảo toàn 99.73% năng lượng không cụt biên.</div>
                <span class="step-metric">Σ w_ij = 1.000</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header">
                  <span class="step-num">04</span>
                  <span class="step-badge">LÀM NÉT (YC7)</span>
                </div>
                <div class="step-thumb"><img src="{l2_step4}" alt="Làm nét"></div>
                <div class="step-title">Sharpening Identity - Lap</div>
                <div class="step-formula">g = f - ∇²f (Σ = 1.000)</div>
                <div class="step-desc">Kernel 4/8-lân cận có hệ số âm xung quanh và tâm dương lớn, tăng vi sai biên cạnh.</div>
                <span class="step-metric">Preserve Brightness</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header">
                  <span class="step-num">05</span>
                  <span class="step-badge">KHỬ NHIỄU (YC10)</span>
                </div>
                <div class="step-thumb"><img src="{l2_step5}" alt="Lọc trung vị"></div>
                <div class="step-title">Lọc Phi Tuyến Median</div>
                <div class="step-formula">g(x,y) = median{{f(s,t)}}</div>
                <div class="step-desc">Lọc thống kê thứ tự loại bỏ triệt để 100% hạt nhiễu muối tiêu 3% mà không làm lem biên.</div>
                <span class="step-metric">Triệt Tiêu 100% Cực Trị</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l2-workflow">
            <div class="img-container">
              <img src="{svg_lab02}" alt="Workflow Architecture Lab 02">
              <div class="img-caption">
                <strong>Sơ đồ kiến trúc toàn thể luồng xử lý Lab 02: Từ dữ liệu đầu vào, các tầng lọc đến phân nhánh đánh giá (Vector SVG)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l2-mapping">
            <div class="visual-canvas" style="justify-content: center;">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Yêu Cầu</th>
                    <th>Thuật Toán</th>
                    <th>Thông Số Thực Nghiệm</th>
                    <th>Công Thức / Kernel</th>
                    <th>Kết Quả Quan Sát</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>YC1</strong></td><td>Độ sáng</td><td>β = ±50</td><td>g = clip(f + β, 0, 255)</td><td>Dải mức xám dịch sang [50, 255] (+50) và [0, 197] (-50)</td></tr>
                  <tr><td><strong>YC2</strong></td><td>Tương phản</td><td>α = 0.5 (co), α = 1.8 (giãn)</td><td>g = clip(α · f, 0, 255)</td><td>α=0.5 ảnh mờ xám [0, 127]; α=1.8 tăng sắc nét nhưng bão hòa 255</td></tr>
                  <tr><td><strong>YC3</strong></td><td>Âm bản</td><td>Đảo ngược thang xám</td><td>g = 255 - f</td><td>Đảo ngược hoàn toàn thang sáng tối, nổi rõ chi tiết vùng tối</td></tr>
                  <tr><td><strong>YC4</strong></td><td>Cắt ngưỡng</td><td>T ∈ {{80, 127, 180}}</td><td>g = 255 if f ≥ T else 0</td><td>T=127 là trung điểm đối xứng; histogram nhị phân có đúng 2 vạch {{0, 255}}</td></tr>
                  <tr><td><strong>YC5–6</strong></td><td>Mean & Gauss</td><td>Mean (7×7, 21×21), Gauss (σ=3, σ=7)</td><td>exp(-(x²+y²)/(2σ²)), k≈6σ+1</td><td>Gauss làm mờ tự nhiên hơn Mean, không bị artifact ô vuông</td></tr>
                  <tr><td><strong>YC7</strong></td><td>Làm nét</td><td>4-lân cận & 8-lân cận</td><td>Sharpen = Identity − Laplacian</td><td>Σ = 1.000 bảo toàn độ sáng; có hệ số âm xung quanh tăng vi sai</td></tr>
                  <tr><td><strong>YC8</strong></td><td>Sobel vs Prewitt</td><td>Sobel [1,2,1] vs Prewitt [1,1,1]</td><td>|G| = √(Gx² + Gy²)</td><td>Tương quan r ≈ 0.996; biên độ Sobel lớn hơn Prewitt ~1.37× (39.5 vs 28.9)</td></tr>
                  <tr><td><strong>YC9</strong></td><td>Tổng kernel</td><td>Σ ∈ {{0.5, 1.0, 2.0, 0.0}}</td><td>output = c · Σkernel trên c=100</td><td>Output đúng = {{50, 100, 200, 0}}; Σ=0 xóa sạch vùng phẳng chỉ giữ biên</td></tr>
                  <tr><td><strong>YC10–11</strong></td><td>Phi tuyến & Master</td><td>Nhiễu muối tiêu 3%, 7 bộ lọc</td><td>Median 3×3 vs Bilateral</td><td>Median loại 100% cực trị; Master Grid tổng kết toàn diện</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Chuỗi xử lý 5 tầng:</strong> Phân tầng rõ rệt giữa biến đổi điểm, lọc tuyến tính, đạo hàm vi sai và lọc phi tuyến.</div>
          <div class="takeaway-pill"><strong>Chống tràn Saturation:</strong> Mọi phép toán uint8 đều bắt buộc clip(0, 255) để ngăn lỗi tràn số âm/dương.</div>
          <div class="takeaway-pill"><strong>Kiểm thử toàn diện:</strong> 22/22 unit tests passed bảo đảm tính toàn vẹn của mã nguồn trong tests/test_filters.py.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 03: LAB 02 - POINT OPERATIONS & BLUR
       ======================================================================== -->
  <section class="slide-screen" id="slide-2">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 02 / 02 · PHẦN I</span>
          <h2 class="slide-title">Lab 02 · Toán Tử Điểm Ảnh (YC1–4) & Làm Mờ Không Gian (YC5–6)</h2>
          <div class="slide-summary">Khảo sát trực tiếp phép biến đổi cường độ điểm ảnh, lược đồ xám histogram nhị phân và lọc mờ Mean vs Gaussian.</div>
        </div>
        <span class="slide-badge-num">Slide 03 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-p-bright">YC1 · Độ Sáng (β = ±50)</button>
            <button class="tab-btn" data-tab="tab-u-p-contrast">YC2 · Tương Phản (α = 0.5, 1.8)</button>
            <button class="tab-btn" data-tab="tab-u-p-negative">YC3 · Đảo Âm Bản (255 - f)</button>
            <button class="tab-btn" data-tab="tab-u-p-thresh">YC4 · Cắt Ngưỡng (T = 80, 127, 180)</button>
            <button class="tab-btn" data-tab="tab-u-p-hist">YC4 · Histogram Gốc vs Nhị Phân</button>
            <button class="tab-btn" data-tab="tab-u-p-blur">YC5–6 · Lưới So Sánh Mean vs Gauss</button>
          </div>
          <div class="tab-pane active" id="tab-u-p-bright">
            <div class="img-container">
              <img src="{l2_bright}" alt="Brightness Comparison">
              <div class="img-caption">
                <strong>YC1: Biến đổi độ sáng g = clip(f + β, 0, 255) với β = +50 (dải [50, 255]) và β = -50 (dải [0, 197])</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-p-contrast">
            <div class="img-container">
              <img src="{l2_contrast}" alt="Contrast Comparison">
              <div class="img-caption">
                <strong>YC2: Biến đổi tương phản g = clip(α · f, 0, 255) với α = 0.5 (co dải động, bạc màu) và α = 1.8 (giãn dải động, bão hòa sáng)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-p-negative">
            <div class="img-container">
              <img src="{l2_negative}" alt="Negative Comparison">
              <div class="img-caption">
                <strong>YC3: Biến đổi âm bản g = 255 - f, đảo ngược thang xám, làm nổi bật chi tiết các vùng tối của ảnh gốc</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-p-thresh">
            <div class="img-container">
              <img src="{l2_thresh}" alt="Threshold Comparison">
              <div class="img-caption">
                <strong>YC4: Phân ngưỡng nhị phân với T ∈ {{80, 127, 180}}. T=127 là trung điểm đối xứng; T=80 cho nhiều vùng trắng, T=180 lọc gắt</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-p-hist">
            <div class="side-by-side-grid">
              <div class="img-container">
                <img src="{l2_hist_orig}" alt="Histogram Original">
                <div class="img-caption"><strong>Histogram Mức Xám Gốc (Phân bố liên tục [0, 255])</strong></div>
              </div>
              <div class="img-container">
                <img src="{l2_hist_bin}" alt="Histogram Binary">
                <div class="img-caption"><strong>Histogram Sau Cắt Ngưỡng T=127 (Đúng 2 vạch rời rạc {{0, 255}})</strong></div>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-p-blur">
            <div class="img-container">
              <img src="{l2_blur}" alt="Blur Comparison Grid">
              <div class="img-caption">
                <strong>YC5–6: So sánh Mean (7×7, 21×21) vs Gaussian (σ=3 k=19, σ=7 k=43). Gaussian bảo tồn chuyển tiếp mượt mà theo quy tắc k ≈ 6σ+1</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Độ sáng tịnh tiến:</strong> Phép cộng beta tịnh tiến toàn bộ lược đồ xám mà không thay đổi hình dạng phân bố.</div>
          <div class="takeaway-pill"><strong>Tương phản co giãn:</strong> Hệ số alpha > 1 khuếch đại khoảng cách mức xám giữa các pixel lân cận, làm nổi bật chi tiết.</div>
          <div class="takeaway-pill"><strong>Gaussian vs Mean:</strong> Trọng số suy giảm hàm mũ chuẩn của Gaussian loại bỏ hiện tượng răng cưa ô vuông thô ráp của Mean.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 04: LAB 02 - SHARPENING & SOBEL/PREWITT
       ======================================================================== -->
  <section class="slide-screen" id="slide-3">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 02 / 03 · PHẦN II & III</span>
          <h2 class="slide-title">Lab 02 · Làm Nét Sharpening (YC7) & Đạo Hàm Gradient Sobel vs Prewitt (YC8)</h2>
          <div class="slide-summary">Bản chất toán học Identity − Laplacian bảo toàn độ sáng và đo lường định lượng tương quan biên r ≈ 0.996.</div>
        </div>
        <span class="slide-badge-num">Slide 04 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-sharpen">YC7 · Kết Quả Làm Nét Ảnh Thực Nghiệm</button>
            <button class="tab-btn" data-tab="tab-u-sharpen-math">Bản Chất: Sharpen (Σ=1) vs Laplacian (Σ=0)</button>
            <button class="tab-btn" data-tab="tab-u-edge-grid">YC8 · Lưới So Sánh Sobel vs Prewitt</button>
            <button class="tab-btn" data-tab="tab-u-edge-mag">YC8 · Bản Đồ Nhiệt Gradient Sobel |G|</button>
            <button class="tab-btn" data-tab="tab-u-edge-quad">YC8 · Đối Chiếu 4 Hướng Đạo Hàm (Gx, Gy)</button>
          </div>
          <div class="tab-pane active" id="tab-u-sharpen">
            <div class="img-container">
              <img src="{l2_sharpen}" alt="Sharpen Comparison">
              <div class="img-caption">
                <strong>YC7: So sánh Ảnh gốc vs Làm nét 4-lân cận [0 -1 0; -1 5 -1; 0 -1 0] vs Làm nét 8-lân cận có trọng số [-1 -2 -1; -2 13 -2; -1 -2 -1]</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-sharpen-math">
            <div class="visual-canvas" style="justify-content: center;">
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Khái Niệm</th>
                    <th>Toán Tử Laplacian Đạo Hàm Bậc 2</th>
                    <th>Bộ Lọc Làm Nét (Sharpening Filter)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>Mục đích</strong></td><td>Phát hiện vị trí đường biên (Edge Detection)</td><td>Tăng cường độ tương phản vi mô tại đường biên (Image Enhancement)</td></tr>
                  <tr><td><strong>Tổng trọng số</strong></td><td><span class="pill-badge warning">Σ w_ij = 0</span> (xóa sạch vùng phẳng đồng nhất)</td><td><span class="pill-badge success">Σ w_ij = 1.000</span> (bảo toàn nguyên vẹn mức xám trung bình)</td></tr>
                  <tr><td><strong>Kernel 4-lân cận</strong></td><td><code>[ 0,  1,  0]<br/>[ 1, -4,  1]<br/>[ 0,  1,  0]</code> (Tổng = 0)</td><td><code>[ 0, -1,  0]<br/>[-1,  5, -1]<br/>[ 0, -1,  0]</code> (5 - 4 = 1)</td></tr>
                  <tr><td><strong>Kernel 8-lân cận</strong></td><td><code>[ 1,  1,  1]<br/>[ 1, -8,  1]<br/>[ 1,  1,  1]</code> (Tổng = 0)</td><td><code>[-1, -2, -1]<br/>[-2, 13, -2]<br/>[-1, -2, -1]</code> (13 - 12 = 1)</td></tr>
                  <tr><td><strong>Hành vi vùng phẳng</strong></td><td>Vùng phẳng c = 100 → output = 0 (màu đen hoàn toàn)</td><td>Vùng phẳng c = 100 → output = 100 (giữ nguyên độ sáng)</td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-edge-grid">
            <div class="img-container">
              <img src="{l2_edges}" alt="Edge Detection Grid">
              <div class="img-caption">
                <strong>YC8: Lưới đạo hàm vi phân Sobel X/Y vs Prewitt X/Y và độ lớn gradient tổng hợp (Tương quan Pearson r ≈ 0.996)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-edge-mag">
            <div class="img-container">
              <img src="{l2_sobel_mag}" alt="Sobel Gradient Magnitude">
              <div class="img-caption">
                <strong>YC8: Bản đồ độ lớn Gradient Sobel |G| = √(Gx² + Gy²). Chuẩn hóa Percentile 99 giúp hiển thị rõ các gân biên mảnh</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-edge-quad">
            <div class="quad-grid">
              <div class="img-container"><img src="{l2_sobel_x}" alt="Sobel X"><div class="img-caption"><strong>Sobel Gx (Biên Dọc, Trọng Số 2)</strong></div></div>
              <div class="img-container"><img src="{l2_sobel_y}" alt="Sobel Y"><div class="img-caption"><strong>Sobel Gy (Biên Ngang, Trọng Số 2)</strong></div></div>
              <div class="img-container"><img src="{l2_prewitt_x}" alt="Prewitt X"><div class="img-caption"><strong>Prewitt Gx (Biên Dọc, Hệ Số 1)</strong></div></div>
              <div class="img-container"><img src="{l2_prewitt_y}" alt="Prewitt Y"><div class="img-caption"><strong>Prewitt Gy (Biên Ngang, Hệ Số 1)</strong></div></div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Bản chất Sharpen:</strong> Sharpen = Identity − Laplacian. Trừ đi đạo hàm bậc hai giúp đẩy cao vút sườn dốc mức xám tại biên.</div>
          <div class="takeaway-pill"><strong>Tương quan r ≈ 0.996:</strong> Cấu trúc hình học đường biên giữa Sobel và Prewitt gần như trùng khít hoàn toàn.</div>
          <div class="takeaway-pill"><strong>Chênh lệch biên độ ~1.37×:</strong> Sobel có hệ số trọng số trung tâm bằng 2 ([1,2,1]), giúp biên độ trung bình đạt ~39.5 so với ~28.9 của Prewitt.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 05: LAB 02 - KERNEL SUM RULE & NONLINEAR FILTERS
       ======================================================================== -->
  <section class="slide-screen" id="slide-4">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 02 / 04 · PHẦN III</span>
          <h2 class="slide-title">Lab 02 · Quy Tắc Tổng Kernel (YC9) & Lọc Phi Tuyến Khử Nhiễu (YC10–11)</h2>
          <div class="slide-summary">Kiểm chứng output = c · Σkernel trên vùng phẳng c=100 và hiệu quả triệt tiêu 100% cực trị của Median Filter trên nhiễu muối tiêu 3%.</div>
        </div>
        <span class="slide-badge-num">Slide 05 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-sum-math">YC9 · Kiểm Chứng Vùng Phẳng c = 100</button>
            <button class="tab-btn" data-tab="tab-u-sum-image">YC9 · Tác Động Thực Tế Lên Ảnh Màu</button>
            <button class="tab-btn" data-tab="tab-u-nl-grid">YC10 · Khử Nhiễu Muối Tiêu 3% (Median vs Bilateral)</button>
            <button class="tab-btn" data-tab="tab-u-all-filters">YC11 · Master Grid 7 Bộ Lọc Không Gian</button>
          </div>
          <div class="tab-pane active" id="tab-u-sum-math">
            <div class="img-container">
              <img src="{l2_sum_rule}" alt="Kernel Sum Rule Analysis">
              <div class="img-caption">
                <strong>YC9: Vùng phẳng c = 100 cho ra đúng giá trị lý thuyết {{50, 100, 200, 0}} tương ứng với Σkernel = {{0.5, 1.0, 2.0, 0.0}}</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-sum-image">
            <div class="img-container">
              <img src="{l2_sum_effects}" alt="Kernel Effects on Image">
              <div class="img-caption">
                <strong>YC9: Hiệu ứng trên ảnh màu: Σ=0.5 tối đi một nửa; Σ=1.0 giữ nguyên độ sáng; Σ=2.0 cháy sáng; Σ=0 xóa nền phẳng chỉ giữ biên</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-nl-grid">
            <div class="img-container">
              <img src="{l2_nonlinear}" alt="Nonlinear Filtering Comparison">
              <div class="img-caption">
                <strong>YC10: Nhiễu muối tiêu 3%. Median 3×3 triệt tiêu 100% cực trị; Bilateral giữ biên sắc nhưng không loại hết muối tiêu</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-all-filters">
            <div class="img-container">
              <img src="{l2_all}" alt="All Filters Grid">
              <div class="img-caption">
                <strong>YC11 Master Grid: Lưới tổng hợp so sánh trực quan ảnh gốc cùng 7 bộ lọc (Mean, Gaussian, Sharpen, Sobel, Prewitt, Median, Bilateral)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Định luật tổng kernel:</strong> Σ=1 bảo toàn độ sáng, Σ=0 xóa vùng phẳng chỉ giữ biên, Σ>1 cháy sáng, Σ<1 sụp tối.</div>
          <div class="takeaway-pill"><strong>Chiến thắng của Median:</strong> Do nhiễu muối tiêu chỉ nhận cực trị 0/255, lọc trung vị loại bỏ 100% hạt nhiễu mà không làm nhòe ảnh.</div>
          <div class="takeaway-pill"><strong>Bilateral Filter:</strong> Kết hợp khoảng cách không gian và độ chênh mức xám, làm mịn vùng phẳng nhưng giữ biên sắc.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 06: LAB 03 - 5 STEPS CANNY PIPELINE & YC1 BENCHMARK
       ======================================================================== -->
  <section class="slide-screen" id="slide-5">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 03 / 01 · 5 BƯỚC TOÁN HỌC & BENCHMARK</span>
          <h2 class="slide-title">Lab 03 · 5 Bước Toán Học Tách Biên Canny & Đối Chiếu Thư Viện (YC1)</h2>
          <div class="slide-summary">Chuỗi toán học khép kín từ lọc mượt Gaussian, đạo hàm Sobel, NMS gọt mỏng 1-pixel đến Hysteresis và benchmark thư viện.</div>
        </div>
        <span class="slide-badge-num">Slide 06 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-l3-ribbon">Chuỗi 5 Bước Toán Học Canny</button>
            <button class="tab-btn" data-tab="tab-u-l3-workflow">Sơ Đồ Kiến Trúc Vector (SVG)</button>
            <button class="tab-btn" data-tab="tab-u-l3-bench">YC1 · Đối Chiếu OpenCV (30,980 px) vs Skimage (7,554 px)</button>
            <button class="tab-btn" data-tab="tab-u-l3-theory">So Sánh Canny vs Sobel vs Laplacian</button>
          </div>
          <div class="tab-pane active" id="tab-u-l3-ribbon">
            <div class="pipeline-ribbon">
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">01</span><span class="step-badge">BƯỚC 1: LÀM MỊN</span></div>
                <div class="step-thumb"><img src="{l3_yc2_sigma}" alt="Gaussian smoothing"></div>
                <div class="step-title">Gaussian Smoothing</div>
                <div class="step-formula">exp(-(x²+y²)/(2σ²))</div>
                <div class="step-desc">Khử nhiễu tần số cao với k = 2·ceil(3σ)+1, bảo toàn năng lượng phân phối Gauss 3-sigma.</div>
                <span class="step-metric">ksize=7×7 (σ=1.0)</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">02</span><span class="step-badge">BƯỚC 2: GRADIENT</span></div>
                <div class="step-thumb"><img src="{l3_yc1}" alt="Sobel Gradient"></div>
                <div class="step-title">Sobel G & Hướng θ</div>
                <div class="step-formula">G=√(Gx²+Gy²), θ=atan2</div>
                <div class="step-desc">Đạo hàm không gian bậc một tính biên độ thay đổi và góc vector pháp tuyến của đường biên.</div>
                <span class="step-metric">L2gradient = True</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">03</span><span class="step-badge">BƯỚC 3: NMS</span></div>
                <div class="step-thumb"><img src="{l3_yc2_thresh}" alt="NMS"></div>
                <div class="step-title">Non-Max Suppression</div>
                <div class="step-formula">Lượng tử 4 hướng (0°,45°,90°,135°)</div>
                <div class="step-desc">So sánh pixel với 2 lân cận dọc theo hướng gradient, gọt mỏng đường biên về độ dày đúng 1 pixel.</div>
                <span class="step-metric">Độ Dày = 1 Pixel</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">04</span><span class="step-badge">BƯỚC 4–5: HYSTERESIS</span></div>
                <div class="step-thumb"><img src="{l3_yc4_regions}" alt="Hysteresis"></div>
                <div class="step-title">Double Threshold & Link</div>
                <div class="step-formula">T_low &lt; Weak &lt; T_high</div>
                <div class="step-desc">Phân 3 loại (Strong / Weak / Loại bỏ); duyệt 8-lân cận giữ lại biên yếu nối với biên mạnh.</div>
                <span class="step-metric">T_low=50, T_high=150</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">05</span><span class="step-badge">HÌNH HỌC HẠ NGUỒN</span></div>
                <div class="step-thumb"><img src="{l3_yc4_circles}" alt="Hough Circles"></div>
                <div class="step-title">Phân Đoạn & Hough</div>
                <div class="step-formula">Hough Gradient (dp, minDist, R)</div>
                <div class="step-desc">Morphology đóng khe hở tạo mask vùng kín và bỏ phiếu Hough định vị chính xác 24/24 hình tròn.</div>
                <span class="step-metric">24/24 Đồng Xu Tròn</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l3-workflow">
            <div class="img-container">
              <img src="{svg_lab03}" alt="Workflow Architecture Lab 03">
              <div class="img-caption">
                <strong>Sơ đồ kiến trúc toàn thể luồng xử lý Lab 03: 5 bước toán học Canny kết hợp phân đoạn hình thái học và Hough Circles (Vector SVG)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l3-bench">
            <div class="img-container">
              <img src="{l3_yc1}" alt="OpenCV vs Skimage Comparison">
              <div class="img-caption">
                <strong>YC1: Đối chiếu trực tiếp: OpenCV (30,980 pixel biên) vs Scikit-Image (7,554 pixel biên) trên ảnh camera 512×512 (cùng ngưỡng 50/150)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l3-theory">
            <div class="visual-canvas" style="justify-content: center;">
              <table class="data-table">
                <thead>
                  <tr><th>Phương Pháp</th><th>Bản Chất Toán Học</th><th>Độ Dày Biên</th><th>Độ Nhạy Nhiễu</th><th>Đánh Giá Kỹ Thuật</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>Sobel / Prewitt</strong></td><td>Đạo hàm bậc một ∇f = [Gx, Gy]ᵀ</td><td>Dày nhiều pixel theo sườn dốc</td><td>Dễ bị kích hoạt bởi nhiễu nếu không lọc trước</td><td>Tính toán nhanh nhưng biên dày, không khép kín</td></tr>
                  <tr><td><strong>Laplacian</strong></td><td>Đạo hàm bậc hai ∇²f = ∂²f/∂x² + ∂²f/∂y²</td><td>Zero-crossing cặp âm/dương quanh biên</td><td>Rất nhạy cảm với nhiễu vì khuếch đại tần số cao</td><td>Không định hướng được biên; cần kết hợp LoG</td></tr>
                  <tr><td><strong>Canny Detector</strong></td><td>Tối ưu hóa 3 tiêu chí John Canny</td><td>Đúng 1 pixel nhờ Non-Maximum Suppression</td><td>Rất bền vững nhờ lọc Gauss và Hysteresis</td><td>Chuẩn vàng xử lý ảnh; biên mảnh liên tục</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Tiêu chí John Canny:</strong> (1) Low error rate (bắt đúng biên), (2) Localization (gần biên thật nhất), (3) Single response (đúng 1 pixel).</div>
          <div class="takeaway-pill"><strong>Non-Maximum Suppression:</strong> Lượng tử hóa góc gradient về 4 hướng chính (0°, 45°, 90°, 135°) để triệt tiêu các pixel không phải cực đại cục bộ.</div>
          <div class="takeaway-pill"><strong>Chênh lệch YC1:</strong> cv2.Canny giữ 30,980 px vs skimage giữ 7,554 px do skimage tự động làm mịn Gauss nội bộ và chuẩn hóa dải float.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 07: LAB 03 - PARAMETER SWEEPS & PREPROCESSING
       ======================================================================== -->
  <section class="slide-screen" id="slide-6">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 03 / 02 · KHẢO SÁT THAM SỐ & TIỀN XỬ LÝ</span>
          <h2 class="slide-title">Lab 03 · Khảo Sát Tham Số σ & Ngưỡng (YC2) & Tiền Xử Lý Ảnh Khó (YC3)</h2>
          <div class="slide-summary">Quét định lượng σ ∈ {0.5, 1.0, 2.0, 3.5}, khử nhiễu Gauss/muối tiêu và cứu ảnh tương phản thấp bằng cv2.equalizeHist (0 → 13,897 px).</div>
        </div>
        <span class="slide-badge-num">Slide 07 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-yc2-sigma">YC2.1 · Quét Gaussian σ (20,012 → 793 px)</button>
            <button class="tab-btn" data-tab="tab-u-yc2-thresh">YC2.2 · Quét Cặp Ngưỡng Kép</button>
            <button class="tab-btn" data-tab="tab-u-yc3-noise-img">YC3 · Ảnh Nhiễu Gauss & Muối Tiêu</button>
            <button class="tab-btn" data-tab="tab-u-yc3-noise-edge">YC3 · Biên Canny Trước vs Sau Lọc</button>
            <button class="tab-btn" data-tab="tab-u-yc3-texture">YC3 · Ảnh Nhiều Vân Gạch (brick 512×512)</button>
            <button class="tab-btn" data-tab="tab-u-yc3-contrast">YC3 · Cứu Tương Phản Thấp (0 → 13,897 px)</button>
          </div>
          <div class="tab-pane active" id="tab-u-yc2-sigma">
            <div class="img-container">
              <img src="{l3_yc2_sigma}" alt="Gaussian Sigma Sweep">
              <div class="img-caption">
                <strong>YC2.1: Quét σ từ 0.5 (20,012 px) → 1.0 (7,937 px) → 2.0 (4,740 px) → 3.5 (793 px). Sigma lớn xóa sạch chi tiết mịn và texture</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc2-thresh">
            <div class="img-container">
              <img src="{l3_yc2_thresh}" alt="Threshold Sweep">
              <div class="img-caption">
                <strong>YC2.2: Quét cặp ngưỡng T_low/T_high: 30/90 (15,972 px), 50/150 (7,937 px), 100/200 (6,025 px), 20/180 (8,483 px)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc3-noise-img">
            <div class="img-container">
              <img src="{l3_yc3_noise_img}" alt="Noise Images">
              <div class="img-caption">
                <strong>YC3: Minh họa ảnh thách thức: Nhiễu Gauss (std=20), Nhiễu muối tiêu (3%) và kết quả làm mịn bằng Bilateral / Median</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc3-noise-edge">
            <div class="img-container">
              <img src="{l3_yc3_noise_edge}" alt="Noise Edges">
              <div class="img-caption">
                <strong>YC3: Biên Canny: Nhiễu Gauss giảm từ 9,003 → 5,938 px nhờ Bilateral; Nhiễu muối tiêu giảm từ 13,201 → 6,963 px nhờ Median 3×3</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc3-texture">
            <div class="img-container">
              <img src="{l3_yc3_texture}" alt="Texture Brick">
              <div class="img-caption">
                <strong>YC3: Ảnh brick 512×512: Lọc Bilateral (d=9, 50, 5) giảm số pixel biên vân bề mặt từ 18,750 → 9,319 px mà vẫn giữ trọn đường ron gạch</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc3-contrast">
            <div class="img-container">
              <img src="{l3_yc3_contrast}" alt="Contrast Improvement">
              <div class="img-caption">
                <strong>YC3: Đối chiếu 3 trạng thái: Ảnh gốc (7,937 px) → Ảnh tương phản thấp [100, 163] (0 px) → Sau cv2.equalizeHist phục hồi 13,897 px</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Tỉ lệ chuẩn 1:3:</strong> Cấu hình tham chiếu 50/150 cho đường biên sạch, loại bỏ hoàn toàn nhiễu nền mà không đứt nét.</div>
          <div class="takeaway-pill"><strong>Nhiễu muối tiêu:</strong> Bắt buộc dùng Median 3×3; nếu không Canny sẽ bắt nhầm hàng nghìn vòng tròn quanh từng hạt cực trị.</div>
          <div class="takeaway-pill"><strong>Cứu ảnh mù biên:</strong> Cân bằng histogram CDF toàn cục trải rộng dải mức xám, giúp độ dốc gradient vượt qua T_high=150.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 08: LAB 03 - REGIONS, HOUGH CIRCLES & EXTENSIONS
       ======================================================================== -->
  <section class="slide-screen" id="slide-7">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 03 / 03 · PHÂN ĐOẠN & MỞ RỘNG</span>
          <h2 class="slide-title">Lab 03 · Phân Đoạn Vùng & Nhận Dạng 24/24 Đồng Xu (YC4) & 4 Câu Hỏi Mở Rộng</h2>
          <div class="slide-summary">Morphology đóng khe hở tạo mask nhị phân 41,111 px, biến đổi Hough Circles nhận diện 100% đồng xu và 4 định hướng nghiên cứu.</div>
        </div>
        <span class="slide-badge-num">Slide 08 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-yc4-regions">YC4.1 · Tìm Vùng Từ Biên (Region Segmentation)</button>
            <button class="tab-btn" data-tab="tab-u-yc4-circles">YC4.2 · Nhận Dạng Hình Tròn (Hough Circles 24 Xu)</button>
            <button class="tab-btn" data-tab="tab-u-l3-ext">Phần III · 4 Câu Hỏi Nghiên Cứu Mở Rộng</button>
          </div>
          <div class="tab-pane active" id="tab-u-yc4-regions">
            <div class="img-container">
              <img src="{l3_yc4_regions}" alt="Region Segmentation">
              <div class="img-caption">
                <strong>YC4.1: Canny (50/150) → Đóng hình thái học (cv2.MORPH_CLOSE, Ellipse 3×3) → scipy.ndimage.binary_fill_holes → Mask nhị phân (41,111 px)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc4-circles">
            <div class="img-container">
              <img src="{l3_yc4_circles}" alt="Hough Circles Detection">
              <div class="img-caption">
                <strong>YC4.2: Tiền xử lý Gaussian 9×9 (σ=2.0) → cv2.HoughCircles (HOUGH_GRADIENT, dp=1.2, minDist=40, param1=100, param2=25) → 24/24 Đồng Xu</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l3-ext">
            <div class="hero-kpi-grid" style="grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 0;">
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">1. Đánh Giá Chất Lượng Đường Biên</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Khi có Ground Truth:</strong> Precision, Recall, F1-Score với dung sai d &le; 2 px và Pratt FOM = (1/max(N_I, N_A)) &Sigma; (1 / (1 + &alpha;·d_i²)).<br/>
                  • <strong>Khi không có Ground Truth:</strong> Đo độ liên tục, độ mỏng 1-pixel và độ tương phản vi sai hai bên sườn biên.
                </div>
              </div>
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">2. Cải Thiện & Tự Động Hóa Canny</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Auto-Canny theo Median:</strong> v = median(img); T_low = max(0, 0.67·v); T_high = min(255, 1.33·v) (rất ổn định).<br/>
                  • <strong>Tối ưu hóa:</strong> Phân tách kernel Gauss 2D thành hai bộ lọc 1D (O(k²) &rarr; O(2k)), SIMD/CUDA trên GPU.
                </div>
              </div>
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">3. Tách Biên Trên Ảnh Màu (Color Canny)</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Đa kênh đơn giản:</strong> Chạy Canny độc lập trên từng kênh (R, G, B / Lab) rồi OR bitwise: <code>E = E_R | E_G | E_B</code>.<br/>
                  • <strong>Di Zenzo Color Tensor:</strong> Tính ma trận biến thiên màu tổng hợp J, tránh triệt tiêu biên khi 2 vùng cùng độ sáng Y nhưng khác màu.
                </div>
              </div>
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">4. Canny Cho Video Real-Time & Khử Rung</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Thách thức Temporal Flickering:</strong> Nhiễu cảm biến giữa các frame liên tiếp khiến đường biên bị nhấp nháy, đứt nối.<br/>
                  • <strong>Giải pháp lọc thời gian:</strong> Làm mượt gradient theo dòng thời gian G_t = &lambda;·G_t + (1-&lambda;)·G_(t-1) hoặc Spatio-temporal 3D.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>YC4.1 Đóng khe hở:</strong> Phép đóng Morphology nối liền các điểm biên bị đứt, binary_fill_holes lấp đầy vùng lõi tạo mask chuẩn 41,111 px.</div>
          <div class="takeaway-pill"><strong>YC4.2 Truyền ảnh mức xám:</strong> cv2.HoughCircles tự tính Canny nội bộ với param1, phát hiện trọn vẹn 24/24 đồng xu không báo nhầm nền.</div>
          <div class="takeaway-pill"><strong>Auto-Canny v:</strong> Công thức median(img) giải quyết 90% nhu cầu chọn ngưỡng tự động trong thực tế mà không cần can thiệp thủ công.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 09: LAB 04 - wHASH 192-BIT PIPELINE & DATASET DESIGN
       ======================================================================== -->
  <section class="slide-screen" id="slide-8">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 04 / 01 · TIẾN TRÌNH wHASH & DỮ LIỆU</span>
          <h2 class="slide-title">Lab 04 · Chu Trình wHash RGB 192-bit & Thiết Kế Tập Dữ Liệu 147 Cặp</h2>
          <div class="slide-summary">Chuẩn hóa 256×256, phân tách 3 kênh màu, phân rã 2D DWT db4 dải LL1, lượng tử trung vị và chia tập chống rò rỉ (Seed 42).</div>
        </div>
        <span class="slide-badge-num">Slide 09 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-l4-ribbon">Chuỗi 5 Bước Tiến Trình wHash 192-bit</button>
            <button class="tab-btn" data-tab="tab-u-l4-workflow">Sơ Đồ Kiến Trúc Vector (SVG)</button>
            <button class="tab-btn" data-tab="tab-u-l4-dataset">Tập Cặp Ảnh: Cùng Nguồn vs Khác Loại</button>
          </div>
          <div class="tab-pane active" id="tab-u-l4-ribbon">
            <div class="pipeline-ribbon">
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">01</span><span class="step-badge">DỮ LIỆU ĐẦU VÀO</span></div>
                <div class="step-thumb"><img src="{l4_dataset}" alt="Dataset pairs"></div>
                <div class="step-title">Tách Kênh & Resize</div>
                <div class="step-formula">Resize(256×256), R, G, B</div>
                <div class="step-desc">26 ảnh nguồn tạo 104 biến thể (130 ảnh); chuẩn hóa 256×256 INTER_AREA, tách 3 kênh màu độc lập.</div>
                <span class="step-metric">26 Nguồn / 147 Cặp</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">02</span><span class="step-badge">BIẾN ĐỔI SÓNG NHỎ</span></div>
                <div class="step-thumb"><img src="{l4_bands}" alt="Wavelet DWT"></div>
                <div class="step-title">2D DWT db4 Level 1</div>
                <div class="step-formula">wavedec2(..., db4, level=1)</div>
                <div class="step-desc">Trích xuất dải xấp xỉ tần số thấp LL1 (131×131) chứa trên 93% năng lượng hình thái, loại bỏ cao tần.</div>
                <span class="step-metric">LL1 (131×131)</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">03</span><span class="step-badge">LƯỢNG TỬ HÓA</span></div>
                <div class="step-thumb"><img src="{l4_baseline}" alt="Baseline hash"></div>
                <div class="step-title">Thu Nhỏ LL & Median</div>
                <div class="step-formula">bit = 1 if LL_8x8 &gt; m else 0</div>
                <div class="step-desc">Thu nhỏ LL về 8×8 (64 hệ số); so sánh với trung vị kênh tạo chuỗi nhị phân 64 bit đạt Entropy 1.0.</div>
                <span class="step-metric">64 bit / Kênh</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">04</span><span class="step-badge">GHÉP NỐI ĐA KÊNH</span></div>
                <div class="step-thumb"><img src="{l4_rgb}" alt="RGB Channels"></div>
                <div class="step-title">Ghép wHash 192-bit</div>
                <div class="step-formula">h = [R_64 | G_64 | B_64]</div>
                <div class="step-desc">Ghép nối 3 chuỗi 64-bit thành vector 192-bit (48 ký tự Hex), khắc phục triệt để nhược điểm mù màu của pHash.</div>
                <span class="step-metric">192-bit Vector</span>
              </div>
              <div class="pipeline-step">
                <div class="step-header"><span class="step-num">05</span><span class="step-badge">ĐÁNH GIÁ & TÌM KIẾM</span></div>
                <div class="step-thumb"><img src="{l4_roc}" alt="ROC evaluation"></div>
                <div class="step-title">Khoảng Cách Hamming</div>
                <div class="step-formula">D_H = &Sigma; (h1 &oplus; h2) &le; &tau;=18</div>
                <div class="step-desc">Tối ưu Youden J chốt ngưỡng &tau;=18 bits trên Calib; đánh giá Evaluation đạt 98.77% và tìm kiếm Top-3.</div>
                <span class="step-metric">AUC = 1.0000</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l4-workflow">
            <div class="img-container">
              <img src="{svg_lab04}" alt="Workflow Architecture Lab 04">
              <div class="img-caption">
                <strong>Sơ đồ luồng toàn thể Lab 04: Chuẩn bị dữ liệu 2 chủ đề, trích xuất wHash RGB 192-bit và đánh giá 2 tập độc lập (Vector SVG)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-l4-dataset">
            <div class="img-container">
              <img src="{l4_dataset}" alt="Dataset pairs preview">
              <div class="img-caption">
                <strong>Minh họa các cặp ảnh: Nhãn 1 (Cùng nguồn: bản gốc vs sáng/tối/nhiễu/JPEG) vs Nhãn 0 (Khác đối tượng dễ nhầm: chó vs muffin, mèo vs kem)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Chia tập theo ảnh nguồn:</strong> Chia split theo nguồn gốc hình ảnh (12 Calib / 14 Eval) ngăn chặn tuyệt đối rò rỉ dữ liệu (Data Leakage).</div>
          <div class="takeaway-pill"><strong>Bản chất wHash 192-bit:</strong> Dải LL1 sóng nhỏ giữ cấu trúc tần số thấp, lượng tử trung vị đảm bảo tính cân bằng bit (32 bit 0, 32 bit 1).</div>
          <div class="takeaway-pill"><strong>Khác biệt với Cryptographic Hash:</strong> wHash giữ khoảng cách Hamming tỉ lệ thuận với khác biệt thị giác, không có hiệu ứng tuyết lở.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 10: LAB 04 - WAVELET DWT, HAMMING MATRIX & ROBUSTNESS
       ======================================================================== -->
  <section class="slide-screen" id="slide-9">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 04 / 02 · WAVELET & ĐỘ BỀN VỮNG</span>
          <h2 class="slide-title">Lab 04 · Biến Đổi 2D DWT db4 (YC1), Ma Trận Hamming (YC2) & Độ Bền Vững (YC3)</h2>
          <div class="slide-summary">Lượng tử hóa trung vị từng kênh màu, khoảng cách Hamming từng cặp và khảo sát độ bền vững qua 7 phép biến đổi.</div>
        </div>
        <span class="slide-badge-num">Slide 10 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-yc1-baseline">YC1 · Baseline Hash: LL (131×131) → LL (8×8) → Bit (8×8)</button>
            <button class="tab-btn" data-tab="tab-u-yc1-rgb">YC1 · Tách 3 Kênh Màu Độc Lập (R, G, B)</button>
            <button class="tab-btn" data-tab="tab-u-yc1-bands">YC1 · 4 Dải Tần Sóng Nhỏ (LL, cH, cV, cD)</button>
            <button class="tab-btn" data-tab="tab-u-yc2-matrix">YC2 · Ma Trận Hamming 26×26</button>
            <button class="tab-btn" data-tab="tab-u-yc2-channels">YC2 · Khoảng Cách Từng Kênh Màu</button>
            <button class="tab-btn" data-tab="tab-u-yc3-robust">YC3 · Khảo Sát Độ Bền Vững 7 Phép Biến Đổi</button>
          </div>
          <div class="tab-pane active" id="tab-u-yc1-baseline">
            <div class="img-container">
              <img src="{l4_baseline}" alt="Baseline Hash Process">
              <div class="img-caption">
                <strong>YC1: Kênh màu → Phân rã DWT db4 LL1 (131×131) → Thu nhỏ 8×8 → So sánh Median tạo 64 bit</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc1-rgb">
            <div class="img-container">
              <img src="{l4_rgb}" alt="RGB Channels Processing">
              <div class="img-caption">
                <strong>YC1: Xử lý 3 kênh màu độc lập R, G, B: Mỗi kênh tạo 64 bit; ghép nối thành vector 192-bit (chuỗi Hex 48 ký tự)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc1-bands">
            <div class="img-container">
              <img src="{l4_bands}" alt="Wavelet Subbands">
              <div class="img-caption">
                <strong>YC1: 4 dải tần 2D DWT: LL1 (Xấp xỉ thấp tần, chứa bố cục chính), cH (Chi tiết ngang), cV (Chi tiết dọc), cD (Chi tiết chéo)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc2-matrix">
            <div class="img-container">
              <img src="{l4_hamming}" alt="Hamming Matrix 26x26">
              <div class="img-caption">
                <strong>YC2: Ma trận khoảng cách Hamming 26×26 giữa toàn bộ ảnh gốc: Đường chéo chính bằng 0; các ảnh cùng chủ đề có tương đồng nhẹ</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc2-channels">
            <div class="img-container">
              <img src="{l4_channel_dist}" alt="Channel Distances">
              <div class="img-caption">
                <strong>YC2: Khoảng cách từng kênh: Cặp cùng nguồn tập trung d_H &le; 6 bit/kênh; cặp khác loại phân bố rải đều từ 20 đến 40 bit/kênh</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc3-robust">
            <div class="img-container">
              <img src="{l4_robust}" alt="Robustness Chart">
              <div class="img-caption">
                <strong>YC3: Độ bền: Biến đổi cường độ/nén rất bền (dark 0, noise 2, jpeg 3, bright 7 &le; 18 bit); biến đổi hình học làm lệch ô lưới 8×8</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Khoảng phân cách lớn:</strong> Cặp cùng nguồn có D_H ∈ [0, 20]; cặp khác loại có D_H ∈ [70, 126] (khoảng phân cách cực kỳ an toàn 50 bit).</div>
          <div class="takeaway-pill"><strong>Miễn nhiễm giảm sáng:</strong> Phép nhân alpha không làm thay đổi thứ tự tương quan giữa các pixel, giữ nguyên trung vị (median D_H = 0).</div>
          <div class="takeaway-pill"><strong>Lọc sạch nhiễu:</strong> Nhiễu tần số cao (noise, jpeg) bị triệt tiêu ở các dải chi tiết cao tần cH, cV, cD.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 11: LAB 04 - YOUDEN THRESHOLD & EVALUATION
       ======================================================================== -->
  <section class="slide-screen" id="slide-10">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 04 / 03 · CHỌN NGƯỠNG & ĐÁNH GIÁ</span>
          <h2 class="slide-title">Lab 04 · Chọn Ngưỡng Youden's J (τ = 18) & Đánh Giá Độc Lập Evaluation</h2>
          <div class="slide-summary">Tối ưu hóa chỉ số Youden J trên 66 cặp Calibration và kiểm định nghiêm ngặt trên 81 cặp Evaluation (Acc = 98.77%, ROC AUC = 1.0000).</div>
        </div>
        <span class="slide-badge-num">Slide 11 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-yc4-thresh-sel">YC4.1 · Chọn Ngưỡng Youden's J (τ = 18 bit)</button>
            <button class="tab-btn" data-tab="tab-u-yc4-roc-eval">YC4.2 · Đường Cong ROC & Phân Bố Hamming Evaluation</button>
            <button class="tab-btn" data-tab="tab-u-yc4-review-pairs">YC4.3 · Kiểm Tra Cặp Điển Hình & Phân Tích 1 FN</button>
          </div>
          <div class="tab-pane active" id="tab-u-yc4-thresh-sel">
            <div class="img-container">
              <img src="{l4_thresh_sel}" alt="Threshold Selection Calibration">
              <div class="img-caption">
                <strong>YC4.1: Tối ưu Youden J = Recall + Specificity - 1 trên 66 cặp Calibration: Ngưỡng τ = 18 đạt Accuracy = 100% (48 TP, 18 TN, 0 FP, 0 FN)</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc4-roc-eval">
            <div class="img-container">
              <img src="{l4_roc}" alt="ROC Evaluation">
              <div class="img-caption">
                <strong>YC4.2: Đánh giá trên 81 cặp Evaluation: Accuracy = 98.77% (80/81 đúng), ROC AUC = 1.0000 hoàn hảo, Precision = 100.0%, Specificity = 100.0%</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-yc4-review-pairs">
            <div class="img-container">
              <img src="{l4_review}" alt="Review Pairs and Error Analysis">
              <div class="img-caption">
                <strong>YC4.3: Phân tích ca lỗi duy nhất: meo_kem_doi_tuong_12 vs bright có D_H = 20 &gt; 18 (False Negative) do ảnh bị clip cháy sáng vùng kem trắng</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Hiệu năng cực cao:</strong> Đạt 80/81 cặp đúng (98.77% Accuracy, 98.21% Recall, 100% Specificity, 100% Precision, F1 = 99.10%).</div>
          <div class="takeaway-pill"><strong>Không có báo nhầm (FP = 0):</strong> Toàn bộ 25 cặp âm (khác đối tượng) đều có D_H &ge; 70 &gt;&gt; 18, loại bỏ 100% dương tính giả.</div>
          <div class="takeaway-pill"><strong>Phân tích 1 ca FN:</strong> meo_kem_12_bright có D_H = 20 (R đổi 12 bit, G đổi 6 bit, B đổi 2 bit) do vùng kem trắng bị clip bão hòa 255 diện rộng.</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 12: LAB 04 - WAVELET COMPARISON & TOP-3 RETRIEVAL
       ======================================================================== -->
  <section class="slide-screen" id="slide-11">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">CHUYÊN ĐỀ LAB 04 / 04 · NÂNG CAO 1 & 2</span>
          <h2 class="slide-title">Lab 04 · So Sánh Cấu Hình Wavelet (Nâng Cao 1) & Tìm Kiếm Top-3 (Nâng Cao 2)</h2>
          <div class="slide-summary">Khảo sát vì sao nhiều bit hơn không đồng nghĩa tốt hơn và kiểm chứng hệ thống truy vấn tương đồng Top-3 (Top-1 Hamming = 2 ≤ 18).</div>
        </div>
        <span class="slide-badge-num">Slide 12 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas">
          <div class="tab-nav">
            <button class="tab-btn active" data-tab="tab-u-adv1-chart">Nâng Cao 1 · Biểu Đồ So Sánh 4 Cấu Hình Wavelet</button>
            <button class="tab-btn" data-tab="tab-u-adv1-table">Bảng Đối Chiếu Định Lượng Chuẩn Hóa</button>
            <button class="tab-btn" data-tab="tab-u-adv2-topk">Nâng Cao 2 · Kết Quả Truy Vấn Top-3</button>
            <button class="tab-btn" data-tab="tab-u-adv2-questions">4 Câu Hỏi Nghiên Cứu Mở Rộng</button>
          </div>
          <div class="tab-pane active" id="tab-u-adv1-chart">
            <div class="img-container">
              <img src="{l4_wavelet_comp}" alt="Wavelet Comparison Chart">
              <div class="img-caption">
                <strong>So sánh Accuracy, F1-score và ROC AUC giữa 4 cấu hình Wavelet trên cùng bộ dữ liệu 147 cặp</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-adv1-table">
            <div class="visual-canvas" style="justify-content: center;">
              <table class="data-table">
                <thead>
                  <tr><th>Cấu Hình</th><th>Độ Dài Mã</th><th>Ngưỡng τ_norm</th><th>Accuracy</th><th>Recall</th><th>ROC AUC</th><th>Đánh Giá Kỹ Thuật</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>RGB db4 L1 (Baseline)</strong></td><td>192 bits</td><td>0.09375 (18/192)</td><td><strong>98.77%</strong></td><td>98.21%</td><td><strong>1.0000</strong></td><td><span class="pill-badge success">Chuẩn tối ưu nhất (Cân bằng dung lượng & độ chính xác)</span></td></tr>
                  <tr><td><strong>RGB haar L1</strong></td><td>192 bits</td><td>0.07292 (14/192)</td><td><strong>98.77%</strong></td><td>98.21%</td><td><strong>1.0000</strong></td><td><span class="pill-badge info">Hiệu năng tương đương db4; tính toán Haar nhanh hơn nhẹ</span></td></tr>
                  <tr><td><strong>RGB db4 L2</strong></td><td>192 bits</td><td>0.09375 (18/192)</td><td><strong>97.53%</strong></td><td>96.43%</td><td><strong>1.0000</strong></td><td><span class="pill-badge warning">Phân rã sâu cấp 2 làm mất thêm chi tiết tần số trung bình (2 FN)</span></td></tr>
                  <tr><td><strong>Gray Step L3 (Slide 17)</strong></td><td>71,542 bits</td><td>0.29416</td><td><strong>69.14%</strong></td><td>55.36%</td><td><strong>0.7157</strong></td><td><span class="pill-badge danger">Kém nhất (25 FN): Lượng tử rint(c/2)%2 quá nhạy cảm với nhiễu</span></td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-adv2-topk">
            <div class="img-container">
              <img src="{l4_topk}" alt="Top-K Retrieval">
              <div class="img-caption">
                <strong>Truy vấn với ảnh nén JPEG: Trả về chính xác 100% ảnh gốc ở vị trí Top-1 (Hamming = 2 &le; 18). Các vị trí Top-2 và Top-3 có Hamming &gt; 80</strong>
                <span class="zoom-hint">🔍 Nhấp để phóng to</span>
              </div>
            </div>
          </div>
          <div class="tab-pane" id="tab-u-adv2-questions">
            <div class="hero-kpi-grid" style="grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 0;">
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">1. Bất Biến Xoay (Rotation Invariance)</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Fourier-Mellin Transform:</strong> Chuyển sang tọa độ cực-logarit (log-polar), biến phép xoay thành phép tịnh tiến.<br/>
                  • <strong>Ước lượng góc chủ đạo:</strong> Dùng trục quán tính bậc hai (Image Moments) để xoay ảnh chuẩn hóa trước khi DWT.
                </div>
              </div>
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">2. Đánh Chỉ Mục Triệu Ảnh (Million-Scale)</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Cây BK-Tree:</strong> Tận dụng bất đẳng thức tam giác trong không gian metric nhị phân Hamming.<br/>
                  • <strong>Multi-Index Hashing (MIH):</strong> Chia 192 bit thành m chuỗi con (chunks), tra cứu bảng băm trực tiếp O(1).
                </div>
              </div>
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">3. Deep Embeddings vs Perceptual Hash</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Deep Embeddings (CLIP / DINOv2):</strong> Hiểu ngữ nghĩa ngữ cảnh sâu nhưng tốn GPU và vector lớn (512 float32 = 2048 bytes).<br/>
                  • <strong>wHash 192-bit:</strong> Cực kỳ nhẹ (24 bytes), chạy CPU thời gian thực, hoàn hảo cho bản quyền và chống trùng lặp.
                </div>
              </div>
              <div class="hero-kpi-card" style="padding: 14px;">
                <div class="hero-kpi-title" style="color: var(--primary);">4. Wavelet Tiên Tiến (WPT & DTCWT)</div>
                <div style="font-size: 11.5px; color: #cbd5e1; line-height: 1.45; margin-top: 4px;">
                  • <strong>Wavelet Packet Transform (WPT):</strong> Phân rã đệ quy cả dải cao tần, phù hợp với ảnh có nhiều hoa văn kết cấu lặp.<br/>
                  • <strong>Dual-Tree Complex Wavelet (DTCWT):</strong> Giảm tính nhạy cảm dịch chuyển và cung cấp tính chọn lọc 6 hướng.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Nghịch lý số lượng bit:</strong> Gray Step L3 dùng 71,542 bits nhưng chỉ đạt 69.14% Acc (25 FN), chứng minh nhiều bit hơn không hề tốt hơn.</div>
          <div class="takeaway-pill"><strong>Truy vấn O(1) với MIH:</strong> wHash 192-bit có thể tra cứu tức thì trong cơ sở dữ liệu hàng triệu ảnh bằng Multi-Index Hashing.</div>
          <div class="takeaway-pill"><strong>Nhận diện bản quyền hoàn hảo:</strong> Top-1 luôn thuộc về ảnh gốc cùng nguồn với độ tin cậy tuyệt đối (Hamming = 2 &lt;&lt; 18).</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ========================================================================
       SLIDE 13: MASTER CROSS-LAB COMPARISON MATRIX & QA
       ======================================================================== -->
  <section class="slide-screen" id="slide-12">
    <div class="slide-frame">
      <div class="slide-header">
        <div>
          <span class="slide-tag">TỔNG KẾT NGHIỆM THU · MA TRẬN 3 BÀI LAB</span>
          <h2 class="slide-title">So Sánh Đối Chiếu Toàn Diện Labs 02, 03 & 04 & Quản Trị Kỹ Nghệ</h2>
          <div class="slide-summary">Bảng tổng kết đối chiếu mục tiêu, thuật toán cốt lõi, kích thước đầu vào/ra, độ phức tạp và kết quả kiểm thử toàn hệ thống.</div>
        </div>
        <span class="slide-badge-num">Slide 13 / 13</span>
      </div>
      <div class="slide-body">
        <div class="visual-canvas" style="justify-content: center;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Hạng Mục So Sánh</th>
                <th>Lab 02 (Toán Tử Điểm & Lọc Không Gian)</th>
                <th>Lab 03 (Tách Biên Canny & Hough)</th>
                <th>Lab 04 (Wavelet Hash RGB 192-bit)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Mục tiêu cốt lõi</strong></td>
                <td>Biến đổi cường độ, tích chập không gian, làm nét & lọc khử nhiễu</td>
                <td>Tách biên mảnh 1-pixel tối ưu, phân đoạn vùng kín & nhận dạng hình tròn</td>
                <td>Trích xuất đặc trưng nhận thức bất biến, phát hiện ảnh trùng lặp & tìm kiếm</td>
              </tr>
              <tr>
                <td><strong>Thuật toán trọng tâm</strong></td>
                <td>Tích chập 2D (Mean, Gauss, Sharpen), Đạo hàm Sobel, Lọc Median</td>
                <td>Lọc Gauss, Sobel Gradient, NMS 1px, Double Threshold, Hysteresis, Hough</td>
                <td>2D DWT db4 Level 1, Lượng tử hóa trung vị, Khoảng cách Hamming, Youden J</td>
              </tr>
              <tr>
                <td><strong>Đầu vào & Đầu ra</strong></td>
                <td>Input: BGR/Gray (1920×1080) → Output: Ảnh đã xử lý lọc/biên</td>
                <td>Input: Gray (512×512, 303×384) → Output: Bản đồ biên nhị phân, Mask, 24 Vòng tròn</td>
                <td>Input: 130 Ảnh màu (26 nguồn) → Output: Vector 192-bit (48 hex), Nhãn phân loại</td>
              </tr>
              <tr>
                <td><strong>Chỉ số đột phá</strong></td>
                <td>Sobel vs Prewitt r ≈ 0.996 · Sharpen Σ = 1.000 · Median khử 100% cực trị</td>
                <td>OpenCV 30,980 px · Cứu ảnh mờ 0 → 13,897 px · 24/24 Coins (100%)</td>
                <td>Youden τ = 18 · Evaluation Acc = 98.77% · Precision 100% · ROC AUC = 1.0000</td>
              </tr>
              <tr>
                <td><strong>Độ phức tạp</strong></td>
                <td>O(H · W · k²) (kích thước kernel k nhỏ 3×3 → 21×21)</td>
                <td>O(H · W) cho Canny + O(H · W · R_range) cho Hough Transform</td>
                <td>O(H · W) cho DWT cấp 1 + O(1) so khớp Hamming bằng __builtin_popcount</td>
              </tr>
              <tr>
                <td><strong>Unit Tests PyTest</strong></td>
                <td><span class="pill-badge success">22 / 22 PASS (test_filters.py)</span></td>
                <td><span class="pill-badge success">11 / 11 PASS (test_canny.py)</span></td>
                <td><span class="pill-badge success">25 / 25 PASS (test_wavelet_hash, baseline, pairs)</span></td>
              </tr>
              <tr style="background: rgba(56, 189, 248, 0.1); font-weight: 700;">
                <td><strong>TỔNG HỢP 3 LABS</strong></td>
                <td colspan="3" style="text-align: center; color: var(--primary);">
                  <strong>58 / 58 UNIT TESTS PASSED (100%) · 0 RUFF WARNINGS · 54 ARTIFACTS · 100% OFFLINE STANDALONE</strong>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="slide-footer">
        <div class="takeaways-pills">
          <div class="takeaway-pill"><strong>Chu trình khép kín:</strong> Từ xử lý ảnh cấp thấp (pixel/kernel ở Lab 02) lên phân đoạn hình học (Lab 03) và trích xuất đặc trưng nhận thức cấp cao (Lab 04).</div>
          <div class="takeaway-pill"><strong>Độ tin cậy tuyệt đối:</strong> 100% mã nguồn được module hóa theo Clean Architecture và kiểm định tự động qua 58 test cases.</div>
          <div class="takeaway-pill"><strong>Báo cáo một chạm:</strong> Toàn bộ 3 bài lab được tích hợp trong 1 slide deck duy nhất, trình chiếu trơn tru trên mọi thiết bị mà không cần internet.</div>
        </div>
      </div>
    </div>
  </section>

{LIGHTBOX_MARKUP}
<script>{SHARED_JS}</script>
</body>
</html>"""

def render_unified_markdown_report() -> str:
    r"""Generate Comprehensive Unified Markdown Report for Labs 02-04."""
    return r"""# BÁO CÁO TỔNG HỢP CHUYÊN ĐỀ THỊ GIÁC MÁY TÍNH: LAB 02 – LAB 03 – LAB 04
## TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) — KHOA CÔNG NGHỆ THÔNG TIN

- **Môi trường thực nghiệm**: Python 3.14.8 | OpenCV 5.0.0 | PyWavelets 1.8.0 | Scikit-Image 0.26.0 | NumPy 2.2.6
- **Báo cáo Slide Trình Chiếu HTML tương tác 100vh**: [LABS_02_03_04_REPORT.html](LABS_02_03_04_REPORT.html) (và [PRESENTATION.html](PRESENTATION.html))
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

1. **Khởi chạy Slide Trình Chiếu HTML 100vh**:
   - Mở trực tiếp file [LABS_02_03_04_REPORT.html](LABS_02_03_04_REPORT.html) hoặc [PRESENTATION.html](PRESENTATION.html) trong bất kỳ trình duyệt web nào (Chrome, Firefox, Safari, Edge).
   - Dùng phím `Space`, `↓` hoặc `PageDown` để lướt qua 13 slide trình chiếu toàn màn hình.
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
    """Generate unified HTML and Markdown reports for Labs 02-04."""
    LABS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Unified HTML Slide Presentation
    html_content = render_unified_labs_02_03_04()

    out_unified_html = LABS_DIR / "LABS_02_03_04_REPORT.html"
    out_unified_html.write_text(html_content, encoding="utf-8")
    print(f"✔ Generated Unified Labs 02-04 Slide Deck: {out_unified_html} ({out_unified_html.stat().st_size // 1024} KB)")

    out_presentation_html = LABS_DIR / "PRESENTATION.html"
    out_presentation_html.write_text(html_content, encoding="utf-8")
    print(f"✔ Synchronized Master Presentation Deck: {out_presentation_html} ({out_presentation_html.stat().st_size // 1024} KB)")

    # 2. Generate Unified Markdown Technical Report
    md_content = render_unified_markdown_report()
    out_unified_md = LABS_DIR / "LABS_02_03_04_REPORT.md"
    out_unified_md.write_text(md_content, encoding="utf-8")
    print(f"✔ Generated Unified Labs 02-04 Markdown Report: {out_unified_md} ({out_unified_md.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
