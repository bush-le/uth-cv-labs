"""Render the Markdown report as a standalone report following the experiment flow."""
import ast
import base64
import html
import json
import re
from pathlib import Path

import markdown


def split_sections(content, level):
    pattern = rf'<h{level} id="([^"]+)">(.*?)</h{level}>'
    matches = list(re.finditer(pattern, content, re.S))
    return [(m[1], re.sub('<[^>]+>', '', m[2]), content[m.end():matches[i+1].start() if i+1 < len(matches) else len(content)]) for i, m in enumerate(matches)]


def details(title, body):
    return f'<details><summary>{html.escape(title)}</summary><div class="detail-body">{body}</div></details>'


def code_excerpt(notebook, cells):
    parts = []
    for index in cells:
        source = ''.join(notebook['cells'][index]['source'])
        lines = source.splitlines()
        selected = []
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                name = ast.unparse(statement.value.func)
                if name in ('print', 'show_grid') or name.startswith('visual.'):
                    continue
            selected.append('\n'.join(lines[statement.lineno-1:statement.end_lineno]))
        snippet = '\n'.join(selected)
        ast.parse(snippet)
        parts.append(f'<p class="code-note">Cell {index+1} của notebook; trích phần thực hiện, bỏ lời gọi in/vẽ cấp ngoài. Các biến được chuẩn bị ở cell trước.</p><pre><code class="language-python">{html.escape(snippet)}</code></pre>')
    return details('Xem code từ notebook', ''.join(parts)) if parts else ''


def make_map(nodes, paths, description):
    output = '<svg class="map-svg" viewBox="0 0 1100 440" role="img" aria-labelledby="map-title map-desc" xmlns="http://www.w3.org/2000/svg"><title id="map-title">Sơ đồ thí nghiệm</title><desc id="map-desc">'+html.escape(description)+'</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#789887"/></marker></defs>'
    for path in paths:
        output += f'<path d="{path}" fill="none" stroke="#789887" stroke-width="2" marker-end="url(#arrow)"/>'
    for x, y, width, target, title, subtitle in nodes:
        output += f'<a href="#{target}"><rect x="{x}" y="{y}" width="{width}" height="76" rx="12" fill="#edf5e8" stroke="#acc7ac"/><text x="{x+width/2}" y="{y+31}" text-anchor="middle" class="title">{html.escape(title)}</text><text x="{x+width/2}" y="{y+55}" text-anchor="middle">{html.escape(subtitle)}</text></a>'
    return output+'</svg>'


def main():
    folder = Path(__file__).parent
    root = folder.parents[1]
    lab = folder.name[-2:]
    config = json.loads((folder/'report_layout.json').read_text(encoding='utf-8'))
    report = root/f'docs/labs/lab{lab}/LAB_{lab}_REPORT.md'
    content = markdown.markdown(report.read_text(encoding='utf-8'), extensions=['fenced_code', 'toc', 'tables'])
    def embed(match):
        path = (report.parent/match[1]).resolve()
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')+'"'
    content = re.sub(r'src="([^"]+\.png)"', embed, content)
    sections = split_sections(content, 2)
    notebook = json.loads((root/f'notebooks/labs/lab_{lab}.ipynb').read_text(encoding='utf-8'))
    result_groups = {}
    consumed = set(config['result_sections'])
    for i in consumed:
        for ident, title, body in split_sections(sections[i][2], 3):
            result_groups[ident] = (title, body)
    stages = []
    for number, stage in enumerate(config['stages'], 1):
        target, title, route, intro, origin, cells, takeaway = stage
        if origin is None:
            body = ''
        elif isinstance(origin, int):
            consumed.add(origin)
            body = sections[origin][2]
        else:
            body = result_groups[origin][1]
        figures = re.findall(r'<img\b[^>]*>', body)
        if origin is None and cells:
            for output in notebook['cells'][cells[0]].get('outputs', []):
                image = output.get('data', {}).get('image/png')
                if image:
                    encoded = ''.join(image) if isinstance(image, list) else image
                    figures.append('<img alt="Ảnh camera đầu vào từ output notebook" src="data:image/png;base64,'+encoded+'">')
                    break
        narrative = re.sub(r'<p>\s*<img\b[^>]*>\s*</p>', '', body)
        narrative = re.sub(r'<img\b[^>]*>', '', narrative)
        # Preserve existing explanatory code as well as current notebook excerpts.
        cards = []
        for image in figures:
            caption_match = re.search(r'alt="([^"]*)"', image)
            caption = html.unescape(caption_match[1]) if caption_match else title
            image = image.replace('<img ', '<img loading="lazy" ')
            cards.append(f'<figure class="image-card">{image}<figcaption><b>{html.escape(title)}</b>{html.escape(caption)}</figcaption></figure>')
        stage_body = ''.join(cards)+f'<p class="takeaway">{html.escape(takeaway)}</p>'+code_excerpt(notebook, cells)
        if narrative.strip():
            stage_body += details('Giải thích, số liệu & giới hạn', narrative)
        if number < len(config['stages']):
            nxt = config['stages'][number]
            stage_body += f'<a class="next" href="#{nxt[0]}">Đọc tiếp · {html.escape(nxt[1])} →</a>'
        stages.append(f'<section class="stage" id="{target}"><div class="stage-head"><span class="number">{number:02d}</span><div><div class="eyebrow">THEO DÕI THÍ NGHIỆM</div><h2>{html.escape(title)}</h2></div></div><div class="mini-flow">{html.escape(route)}</div><p class="intro">{html.escape(intro)}</p>{stage_body}</section>')
    lead = content.split("<h2", 1)[0]
    lead = re.sub(r"<h1.*?</h1>", "", lead, flags=re.S)
    support = [details("Thông tin bản báo cáo", lead)] if lead.strip() else []
    for i, (ident, title, body) in enumerate(sections):
        if i not in consumed:
            support.append(f'<div id="{ident}">'+details(title, body)+'</div>')
    aliases = ''.join(f'<span id="{ident}"></span>' for ident in result_groups)
    nav = '<a href="#flow">Sơ đồ</a>'+''.join(f'<a href="#{stage[0]}">{i+1:02d} · {html.escape(stage[1])}</a>' for i, stage in enumerate(config['stages']))+'<a href="#support">Chi tiết</a><button type="button" onclick="window.print()">In / PDF</button>'
    css = (folder/'report.css').read_text(encoding='utf-8')
    js = (folder/'report.js').read_text(encoding='utf-8')
    diagram = make_map(config['nodes'], config['paths'], config['description'])
    output = '<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'+f'<title>Lab {lab} — Báo cáo theo flow</title><style>{css}</style></head><body><main><header><div class="eyebrow">COMPUTER VISION / LAB {lab}</div><h1>{html.escape(config["hero"])}</h1><p>{html.escape(config["subtitle"])}</p><div class="links"><a href="../../../notebooks/labs/lab_{lab}.ipynb">Notebook thực hành ↗</a><a href="LAB_{lab}_REPORT.md">Báo cáo Markdown ↗</a></div></header><nav aria-label="Lộ trình">{nav}</nav><section class="flow-map" id="flow"><div class="eyebrow">SƠ ĐỒ THÍ NGHIỆM</div><h2>{html.escape(config["map_title"])}</h2><p>{html.escape(config["description"])}</p><div class="map-scroll" tabindex="0" aria-label="Sơ đồ; cuộn ngang trên màn hình nhỏ">{diagram}</div></section>'+aliases+''.join(stages)+'<section class="support" id="support"><h2>Lý thuyết, quy trình & tài liệu đối chiếu</h2>'+''.join(support)+'</section><footer>Hình và số liệu giữ từ báo cáo hiện có; không chạy lại thí nghiệm. Code được trích từ notebook khi tạo HTML.</footer></main><dialog id="viewer" aria-label="Xem ảnh lớn"><button type="button">Đóng ×</button><img alt=""></dialog>'+f'<script>{js}</script></body></html>'
    report.with_suffix('.html').write_text(output, encoding='utf-8')
    print(f'Lab {lab}: {len(stages)} stages; rendered from Markdown.')


if __name__ == '__main__':
    main()
