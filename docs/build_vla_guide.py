"""
Build "Vision-language-action models, from the ground up" as an A4 PDF.

Style follows a LaTeX article: serif body, numbered sections, a linked
contents list, "In plain words" and "Worked example" boxes, ruled tables.

    py build_vla_guide.py            -> writes VLA_guide.pdf next to this file
"""

import io
import math
import os

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, KeepTogether,
                                ListFlowable, ListItem, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "VLA_guide.pdf")

# ---------------------------------------------------------------- fonts
FONTS = r"C:\Windows\Fonts"
for name, file in [("Serif", "times.ttf"), ("Serif-Bold", "timesbd.ttf"),
                   ("Serif-Italic", "timesi.ttf"), ("Serif-BoldItalic", "timesbi.ttf"),
                   ("Mono", "consola.ttf"), ("Mono-Bold", "consolab.ttf"),
                   ("Mono-Italic", "consolai.ttf"), ("Mono-BoldItalic", "consolaz.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, file)))
registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                   italic="Serif-Italic", boldItalic="Serif-BoldItalic")
registerFontFamily("Mono", normal="Mono", bold="Mono-Bold",
                   italic="Mono-Italic", boldItalic="Mono-BoldItalic")

# ---------------------------------------------------------------- colours
LINK_HEX = "#1F78A4"
LINK = HexColor(LINK_HEX)
INK = HexColor("#1A1A1A")
PLAIN_HDR, PLAIN_BODY = HexColor("#0B6694"), HexColor("#F3F3FB")
WORK_HDR, WORK_BODY, WORK_EDGE = HexColor("#B4B4B4"), HexColor("#F6F6F6"), HexColor("#B4B4B4")
RED, TEAL, INDIGO = HexColor("#B8141C"), HexColor("#00807F"), HexColor("#3A3AA8")
LAV, LAV_DARK = HexColor("#E9E9F9"), HexColor("#A9A9E0")
GREY_FILL, GREY_EDGE = HexColor("#EFEFEF"), HexColor("#777777")
CODE_BG, CODE_EDGE = HexColor("#F7F7F7"), HexColor("#D0D0D0")

# ---------------------------------------------------------------- page geometry
PAGE_W, PAGE_H = A4
MARGIN_X = 22 * mm
W = PAGE_W - 2 * MARGIN_X

# ---------------------------------------------------------------- styles
def style(name, **kw):
    base = dict(fontName="Serif", fontSize=10.5, leading=13.6, textColor=INK)
    base.update(kw)
    return ParagraphStyle(name, **base)

BODY = style("body", alignment=TA_JUSTIFY, spaceAfter=6)
LEAD = style("lead", alignment=TA_JUSTIFY, spaceAfter=6, keepWithNext=1)   # intro to a table/figure
TITLE = style("title", fontName="Serif-Bold", fontSize=19, leading=23, alignment=TA_CENTER)
SUB = style("sub", fontSize=12, leading=15, alignment=TA_CENTER)
BYLINE = style("byline", fontSize=11.5, alignment=TA_CENTER)
HOWTO = style("howto", fontSize=9.6, leading=12, alignment=TA_JUSTIFY,
              leftIndent=10 * mm, rightIndent=10 * mm)
H1 = style("h1", fontName="Serif-Bold", fontSize=15, leading=19, spaceBefore=14,
           spaceAfter=9, keepWithNext=1)
H2 = style("h2", fontName="Serif-Bold", fontSize=11.5, leading=14, spaceBefore=8,
           spaceAfter=5, keepWithNext=1)
CONTENTS = style("contents", fontName="Serif-Bold", fontSize=14, leading=18, spaceAfter=6)
TOC = style("toc", fontName="Serif-Bold", fontSize=10.5, leading=13)
TOC_R = style("tocr", fontName="Serif-Bold", fontSize=10.5, leading=13, alignment=TA_RIGHT)
BOX_TXT = style("boxtxt", fontSize=10, leading=12.8, alignment=TA_JUSTIFY, spaceAfter=3)
BOX_HDR = style("boxhdr", fontName="Serif-Bold", fontSize=9.5, leading=11, textColor=white)
TH = style("th", fontName="Serif-Bold", fontSize=9.6, leading=11.8)
TD = style("td", fontSize=9.6, leading=11.8)
EQ = style("eq", alignment=TA_CENTER, spaceBefore=2, spaceAfter=7)
SMALL = style("small", fontSize=8.8, leading=11, alignment=TA_LEFT)
CODE = ParagraphStyle("code", fontName="Mono", fontSize=8.4, leading=10.6, textColor=INK)


# ---------------------------------------------------------------- content helpers
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def c(s):
    """Inline code."""
    return '<font face="Mono" size="9">{}</font>'.format(esc(s))


def P(text, st=BODY):
    return Paragraph(text, st)


def bullets(items, st=BODY):
    return ListFlowable([ListItem(P(t, st), leftIndent=14) for t in items],
                        bulletType="bullet", start="\u2022", leftIndent=14,
                        bulletFontName="Serif", bulletFontSize=10, spaceAfter=4)


def numbered(items, st=BODY):
    return ListFlowable([ListItem(P(t, st), leftIndent=16) for t in items],
                        bulletType="1", bulletFormat="%s.", leftIndent=16,
                        bulletFontName="Serif", bulletFontSize=10.5, spaceAfter=4)


def box(kind, paragraphs):
    plain = kind == "plain"
    title = "In plain words" if plain else "Worked example"
    hdr_col = PLAIN_HDR if plain else WORK_HDR
    body_col = PLAIN_BODY if plain else WORK_BODY
    edge = PLAIN_HDR if plain else WORK_EDGE
    body = [p if not isinstance(p, str) else P(p, BOX_TXT) for p in paragraphs]
    t = Table([[P(title, BOX_HDR)], [body]], colWidths=[W], cornerRadii=[3, 3, 3, 3])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), hdr_col),
        ("BACKGROUND", (0, 1), (0, 1), body_col),
        ("BOX", (0, 0), (-1, -1), 0.8, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 3),
        ("BOTTOMPADDING", (0, 0), (0, 0), 3),
        ("TOPPADDING", (0, 1), (0, 1), 6),
        ("BOTTOMPADDING", (0, 1), (0, 1), 5),
    ]))
    return [Spacer(1, 3), KeepTogether([t]), Spacer(1, 9)]   # never split header from body


def btable(rows, widths, head=True):
    data = [[P(cell, TH if head and i == 0 else TD) for cell in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0, hAlign="CENTER",
              spaceBefore=2, spaceAfter=10)
    st = [("LINEABOVE", (0, 0), (-1, 0), 1.0, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 6),
          ("TOPPADDING", (0, 0), (-1, -1), 2.4),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4)]
    if head:
        st.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, INK))
    t.setStyle(TableStyle(st))
    return [t]          # no Spacer: keepWithNext on the lead paragraph must reach the table


def code(text):
    t = Table([[Preformatted(text.strip("\n"), CODE)]], colWidths=[W],
              spaceBefore=2, spaceAfter=9)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.4, CODE_EDGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [t]


# ---------------------------------------------------------------- drawing helpers
def text(d, x, y, s, size=9, col=INK, anchor="middle", font="Serif"):
    d.add(String(x, y, s, fontName=font, fontSize=size, fillColor=col, textAnchor=anchor))


def rbox(d, x, y, w, h, stroke=INK, fill=white, sw=0.8, dash=None):
    r = Rect(x, y, w, h, rx=3, ry=3, fillColor=fill, strokeColor=stroke, strokeWidth=sw)
    if dash:
        r.strokeDashArray = dash
    d.add(r)


def arrow(d, x1, y1, x2, y2, col=INK, sw=0.9, head=5.5):
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    d.add(Line(x1, y1, bx, by, strokeColor=col, strokeWidth=sw))
    left = (bx + head * 0.45 * math.sin(ang), by - head * 0.45 * math.cos(ang))
    right = (bx - head * 0.45 * math.sin(ang), by + head * 0.45 * math.cos(ang))
    d.add(Polygon([x2, y2, left[0], left[1], right[0], right[1]],
                  fillColor=col, strokeColor=col, strokeWidth=0.3))


def centered(d):
    # One-cell table: a Drawing cannot carry spacing, and a bare Spacer before it
    # would break keepWithNext on the paragraph that introduces the figure.
    t = Table([[d]], colWidths=[W], hAlign="CENTER", spaceBefore=4, spaceAfter=10)
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return [t]


def fig_vla():
    d = Drawing(W, 150)
    ins = [("camera images", 112, INK, 112), ("instruction text", 70, INK, 87),
           ("joint angles", 28, RED, 62)]
    for label, y, col, target in ins:
        rbox(d, 0, y, 105, 24, stroke=col)
        text(d, 52.5, y + 8.5, label, col=col)
        arrow(d, 105, y + 12, 150, target, col=col)
    rbox(d, 150, 40, 150, 95, stroke=INDIGO, fill=LAV)
    text(d, 225, 103, "VLM backbone", size=10.5, font="Serif-Bold", col=INDIGO)
    text(d, 225, 86, "vision encoder", col=INDIGO)
    text(d, 225, 73, "+ language model", col=INDIGO)
    text(d, 225, 52, "pretrained on image-text data", size=8, font="Serif-Italic", col=INDIGO)
    rbox(d, 345, 108, 125, 24, stroke=TEAL)
    text(d, 407.5, 116.5, "text answer", col=TEAL)
    arrow(d, 300, 120, 345, 120, col=TEAL)
    text(d, 322, 124, "VLM", size=8, col=TEAL, font="Serif-Bold")
    rbox(d, 345, 42, 125, 40, stroke=RED)
    text(d, 407.5, 65, "action head", font="Serif-Bold", col=RED)
    text(d, 407.5, 50, "50 \u00d7 6 joint targets", col=RED)
    arrow(d, 300, 62, 345, 62, col=RED)
    text(d, 322, 66, "VLA", size=8, col=RED, font="Serif-Bold")
    text(d, W / 2, 6, "red: what a VLA adds to a VLM", size=8.5, col=RED, font="Serif-Italic")
    return centered(d)


def fig_chunks():
    d = Drawing(W, 128)
    x0, sec = 40, 150.0                      # 1 s = 150 pt
    d.add(Line(30, 18, 455, 18, strokeColor=INK, strokeWidth=0.8))
    arrow(d, 450, 18, 462, 18)
    text(d, 462, 6, "time", size=8.5, anchor="end", font="Serif-Italic")
    marks = [(0, "0 s"), (0.833, "0.83 s"), (1.667, "1.67 s"), (2.5, "2.5 s")]
    for t, lab in marks:
        x = x0 + t * sec
        d.add(Circle(x, 18, 2.6, fillColor=RED, strokeColor=RED))
        text(d, x, 6, lab, size=8)
    rows = [(0.0, 86, "chunk 1"), (0.833, 61, "chunk 2"), (1.667, 36, "chunk 3")]
    for t, y, lab in rows:
        xs = x0 + t * sec
        xe = min(xs + 1.667 * sec, 455)
        rbox(d, xs, y, xe - xs, 13, stroke=LAV_DARK, fill=LAV, sw=0.6, dash=[2, 2])
        d.add(Rect(xs, y, 0.833 * sec, 13, fillColor=LAV_DARK, strokeColor=INDIGO,
                   strokeWidth=0.6))
        text(d, xs + 4, y + 16, lab + " predicted", size=8, anchor="start", col=INDIGO)
    d.add(Rect(300, 113, 12, 8, fillColor=LAV_DARK, strokeColor=INDIGO, strokeWidth=0.6))
    text(d, 316, 114, "executed", size=8, anchor="start")
    r = Rect(368, 113, 12, 8, fillColor=LAV, strokeColor=LAV_DARK, strokeWidth=0.6)
    r.strokeDashArray = [2, 2]
    d.add(r)
    text(d, 384, 114, "replaced", size=8, anchor="start")
    text(d, 30, 114, "red dots: a new prediction starts", size=8, anchor="start", col=RED)
    return centered(d)


def fig_task():
    d = Drawing(W, 130)
    rbox(d, 0, 45, 128, 52, stroke=INDIGO, fill=LAV)
    text(d, 64, 81, "VLA \u00b7 task A", font="Serif-Bold", col=INDIGO)
    text(d, 64, 67, "pick up the brick and", size=8.8)
    text(d, 64, 56, "show it to the camera", size=8.8)
    rbox(d, 172, 51, 106, 40, stroke=GREY_EDGE, fill=GREY_FILL)
    text(d, 225, 74, "Anomalib", font="Serif-Bold")
    text(d, 225, 60, "anomaly score", size=8.8)
    arrow(d, 128, 71, 172, 71)
    rbox(d, 330, 86, 140, 40, stroke=TEAL, fill=white)
    text(d, 400, 109, "VLA \u00b7 task B", font="Serif-Bold", col=TEAL)
    text(d, 400, 95, "put it in the good bowl", size=8.8)
    rbox(d, 330, 10, 140, 40, stroke=RED, fill=white)
    text(d, 400, 33, "VLA \u00b7 task C", font="Serif-Bold", col=RED)
    text(d, 400, 19, "put it in the defective bowl", size=8.8)
    arrow(d, 278, 80, 330, 104, col=TEAL)
    text(d, 296, 98, "normal", size=8, col=TEAL, font="Serif-Italic")
    arrow(d, 278, 62, 330, 32, col=RED)
    text(d, 288, 33, "anomaly", size=8, col=RED, font="Serif-Italic")
    text(d, 64, 30, "then the next brick", size=8, font="Serif-Italic", col=GREY_EDGE)
    return centered(d)


def fig_hooks():
    d = Drawing(W, 100)
    boxes = [(0, 92, "read robot", "+ cameras", None),
             (124, 100, "action source", "chooses the action", "2"),
             (256, 100, "on_action_ready", "your callbacks", "1"),
             (388, 82, "send_action", "robot \u2192 motors", "3")]
    for x, w, l1, l2, num in boxes:
        hot = num == "1"
        rbox(d, x, 36, w, 40, stroke=RED if hot else INDIGO, fill=white if hot else LAV)
        text(d, x + w / 2, 59, l1, size=9, font="Mono" if "_" in l1 else "Serif-Bold",
             col=RED if hot else INDIGO)
        text(d, x + w / 2, 45, l2, size=8.5)
        if num:
            col = RED if hot else INDIGO
            d.add(Circle(x + w / 2, 88, 7, fillColor=white, strokeColor=col, strokeWidth=0.9))
            text(d, x + w / 2, 85, num, size=8.5, col=col, font="Serif-Bold")
    for xa, xb in [(92, 124), (224, 256), (356, 388)]:
        arrow(d, xa, 56, xb, 56)
    d.add(Line(429, 36, 429, 18, strokeColor=GREY_EDGE, strokeWidth=0.8))
    d.add(Line(429, 18, 46, 18, strokeColor=GREY_EDGE, strokeWidth=0.8))
    arrow(d, 46, 18, 46, 36, col=GREY_EDGE)
    text(d, 237, 5, "every tick, e.g. 30 times a second", size=8, font="Serif-Italic",
         col=GREY_EDGE)
    return centered(d)


def fig_space():
    d = Drawing(W, 128)
    text(d, 0, 116, "when training", size=8.5, anchor="start", font="Serif-Italic",
         col=GREY_EDGE)
    text(d, 0, 52, "when running", size=8.5, anchor="start", font="Serif-Italic",
         col=GREY_EDGE)
    rows = [(76, [("recorded joint angles", "6 numbers", INK, white),
                  ("gripper pose", "x, y, z, yaw, grip", INDIGO, LAV),
                  ("train the VLA", "in this space", INK, white)], "FK"),
            (12, [("VLA output", "x, y, z, yaw, grip", INDIGO, LAV),
                  ("joint targets", "6 numbers", INK, white),
                  ("motors", "SO-101", INK, white)], "IK")]
    xs = [(0, 125), (185, 150), (385, 85)]
    for y, cells, lab in rows:
        for (x, w), (l1, l2, col, fill) in zip(xs, cells):
            rbox(d, x, y, w, 34, stroke=col, fill=fill)
            text(d, x + w / 2, y + 20, l1, size=9, font="Serif-Bold", col=col)
            text(d, x + w / 2, y + 8, l2, size=8.5)
        arrow(d, 125, y + 17, 185, y + 17, col=RED)
        text(d, 155, y + 21, lab, size=8.5, col=RED, font="Serif-Bold")
        arrow(d, 335, y + 17, 385, y + 17)
    return centered(d)


# ---------------------------------------------------------------- document
SECTIONS = [
    ("problem", "The problem they solve"),
    ("vlm", "From VLM to VLA"),
    ("io", "What goes in, what comes out"),
    ("learn", "How it learns: imitation learning"),
    ("chunks", "Action chunks and the control loop"),
    ("choose", "Choosing a policy"),
    ("task", "Your task, the VLA way"),
    ("studio", "Training in Physical AI Studio"),
    ("between", "Between the model and the motors"),
    ("space", "Changing the action space"),
    ("laptop", "Running on the laptop"),
    ("traps", "Traps when you train and run this"),
    ("cheat", "Cheat sheet"),
    ("try", "Try it yourself"),
]
NUM = {key: i + 1 for i, (key, _) in enumerate(SECTIONS)}


def H(key):
    n = NUM[key]
    title = dict(SECTIONS)[key]
    p = Paragraph('<a name="{}"/>{}&nbsp;&nbsp;&nbsp;{}'.format(key, n, title), H1)
    p._toc = (key, "{}  {}".format(n, title))
    return [CondPageBreak(30 * mm), p]


def ref(key):
    return '<a href="#{}" color="{}">{}</a>'.format(key, LINK_HEX, NUM[key])


def runin(head, rest):
    return P("<b>{}</b> {}".format(head, rest))


class GuideDoc(BaseDocTemplate):
    def __init__(self, target, **kw):
        super().__init__(target, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                         topMargin=22 * mm, bottomMargin=24 * mm, **kw)
        frame = Frame(MARGIN_X, 24 * mm, W, PAGE_H - 46 * mm, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("page", [frame], onPage=self._footer)])
        self.toc_pages = {}

    @staticmethod
    def _footer(canv, doc):
        canv.saveState()
        canv.setFont("Serif", 10)
        canv.setFillColor(INK)
        canv.drawCentredString(PAGE_W / 2, 13 * mm, str(doc.page))
        canv.restoreState()

    def afterFlowable(self, flowable):
        info = getattr(flowable, "_toc", None)
        if info:
            key, label = info
            self.toc_pages[key] = self.page
            self.canv.bookmarkPage(key + "_outline")
            self.canv.addOutlineEntry(label, key + "_outline", level=0)


def toc_block(pages):
    rows = []
    for key, title in SECTIONS:
        left = '<a href="#{k}"><font color="{c}">{n}&nbsp;&nbsp;&nbsp;{t}</font></a>'.format(
            k=key, c=LINK_HEX, n=NUM[key], t=title)
        rows.append([P(left, TOC), P(str(pages.get(key, "")), TOC_R)])
    t = Table(rows, colWidths=[W - 40, 40])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.1),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3.1)]))
    return [P("Contents", CONTENTS), t, Spacer(1, 6)]


def make_story(pages):
    s = []
    # ---------------------------------------------------------- title block
    s += [Spacer(1, 12 * mm),
          P("Vision-language-action models, from the ground up", TITLE), Spacer(1, 5),
          P("What a VLM is, how a VLA turns it into robot motion,<br/>"
            "and how to train, steer and run one on the SO-101 for the brick-sorting cell", SUB),
          Spacer(1, 10 * mm),
          P("Written for the Junction 2026 Intel Physical AI Challenge, November 2026", BYLINE),
          Spacer(1, 7 * mm),
          P("<b>How to read this.</b> Go in order: each section uses the one before. "
            "Sections {a}\u2013{b} explain what a VLA is and how it learns. Section {t} applies it "
            "to the sorting task, Sections {s}\u2013{sp} are the hands-on part at your station, and "
            "Section {tr} lists the mistakes that cost the most time. A note on the names: a "
            "<i>VLM</i> (vision-language model) reads images and text and answers in text; a "
            "<i>VLA</i> (vision-language-action model) answers with motor commands. Facts about "
            "Physical AI Studio come from its <font face='Mono' size='8.6'>junction2026</font> "
            "branch; check details with the Intel mentors on site.".format(
                a=ref("vlm"), b=ref("chunks"), t=ref("task"), s=ref("studio"),
                sp=ref("space"), tr=ref("traps")), HOWTO),
          Spacer(1, 6 * mm)]
    s += toc_block(pages)

    # ---------------------------------------------------------- 1
    s += H("problem")
    s.append(P(
        "In many factories people still pick out faulty parts by hand. A robot could do it, but "
        "programming every motion (where the part is, how to approach it, how hard to grip) takes "
        "specialist work, and the program breaks when a part lies a few centimetres off. "
        "<i>Imitation learning</i> turns this around: you show the robot the task a few dozen "
        "times, and a neural network learns to copy you. That network is called a <i>policy</i>."))
    s.append(P(
        "A vision-language-action model is a policy built on top of a large pretrained model that "
        "already understands images and text. Because it starts with that knowledge, it needs far "
        "fewer demonstrations to learn something new. This guide covers:"))
    s.append(bullets([
        "what a VLM is, and the two things a VLA adds to it;",
        "what goes into the model and what comes out, with real shapes;",
        "how it learns from your teleoperation, and why demonstrations matter most;",
        "how to train it in Physical AI Studio and run it on the laptop;",
        "where your own code can sit between the model and the motors."]))

    # ---------------------------------------------------------- 2
    s += H("vlm")
    s.append(P(
        "A <i>vision-language model</i> takes an image and a question and answers in text: "
        "\u201cWhat colour is the brick?\u201d \u2014 \u201cRed.\u201d Inside are two parts. A "
        "<i>vision encoder</i> turns the image into a list of vectors, called tokens. A "
        "<i>language model</i> reads those tokens together with the words of the question. "
        "Trained on huge amounts of image-text data, it learns what things are and where they are."))
    s.append(P(
        "A VLA keeps that backbone and adds two things: the robot\u2019s joint angles as an extra "
        "input, and an <i>action head</i> that outputs numbers for the motors instead of words. "
        "Then it is fine-tuned on robot demonstrations.", LEAD))
    s += fig_vla()
    s += box("plain", [
        "A VLA is a VLM that answers with motion instead of words. The vision-language part "
        "works out <i>what</i> and <i>where</i>; the action head works out <i>how to move</i>."])
    s.append(runin("Why the pretraining matters.",
        "The backbone already knows \u201cbrick\u201d, \u201cbowl\u201d, \u201cleft\u201d and "
        "\u201cred\u201d. Your demonstrations only have to teach the motion. That is why 50 to 100 "
        "demonstrations can be enough, where a policy learned from scratch would need many more."))
    s.append(runin("Not every policy is a VLA.",
        "ACT (Action Chunking with Transformers) learns from scratch and takes no text. It is "
        "small and fast, and often the best first baseline. Section {} compares them.".format(
            ref("choose"))))

    # ---------------------------------------------------------- 3
    s += H("io")
    s.append(P("At every control step the policy receives an observation and returns actions:",
               LEAD))
    s += btable([
        ["", "What it is", "On your station", "Shape"],
        ["In", "camera images", "top view and wrist camera", "e.g. 2 \u00d7 480 \u00d7 640 \u00d7 3"],
        ["In", "robot state", "the follower\u2019s 6 joint angles", "6 numbers"],
        ["In", "instruction", "\u201cput the brick in the good bowl\u201d", "text"],
        ["Out", "action chunk", "the next 50 joint targets", "50 \u00d7 6 numbers"],
    ], [28, 92, 205, 145])
    s.append(P(
        "The joints, in order: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, "
        "gripper. Each action is a <i>target position</i> for all six: the servos drive "
        "themselves there."))
    s.append(runin("Units.",
        "The SO-101 driver can work in raw ticks (0 to 4095 per turn), in degrees, or in a "
        "normalized range of \u2212100 to 100 after calibration. A model must run in the unit it "
        "was trained in."))
    s += box("work", [
        "<b>One observation</b> (numbers made up): two camera frames; state "
        "(0.0, \u221298.5, 95.1, 72.3, \u22121.2, 1.8) in degrees; instruction \u201cpick up the "
        "brick and show it to the inspection camera\u201d.",
        "<b>The answer</b> is 50 rows such as (0.4, \u221297.9, 94.0, 72.8, \u22121.2, 1.8), "
        "(0.9, \u221297.2, 92.8, 73.4, \u22121.1, 1.8), \u2026: a smooth path for the next 50 "
        "control steps. At 30 steps per second that is 50 / 30 \u2248 1.7 s of motion."])

    # ---------------------------------------------------------- 4
    s += H("learn")
    s.append(runin("Teleoperation.",
        "You move the leader arm by hand and the follower copies it in real time. Meanwhile "
        "Studio records every camera frame, every joint angle and the instruction you typed. One "
        "demonstration from start to finish is an <i>episode</i>; a set of episodes is a "
        "<i>dataset</i>."))
    s.append(runin("Training.",
        "The model sees one moment of an episode and must predict the actions you actually took "
        "next. The difference between its guess and your movement is the <i>loss</i>. Training "
        "adjusts the weights to make the loss small, over many passes through the data."))
    s.append(runin("Fine-tuning.",
        "The backbone starts from pretrained weights, and training only nudges it towards your "
        "robot and your task. That is why one cloud GPU for a few hours is enough."))
    s += box("work", [
        "<b>Dataset size.</b> 80 episodes of 12 s at 30 frames per second give "
        "80 \u00d7 12 \u00d7 30 = 28,800 training samples, each with two images, six joint angles "
        "and the 50 actions that followed. A large model can memorise that easily, which is why "
        "<i>variety</i> (different brick positions, some clutter) matters more than the count."])
    s += box("plain", [
        "The policy can only be as good as your demonstrations. It copies hesitations, jerky "
        "moves and mistakes as faithfully as the good parts."])

    # ---------------------------------------------------------- 5
    s += H("chunks")
    s.append(P(
        "The model does not predict one step at a time. It predicts a <i>chunk</i>, for example "
        "the next 50 steps, and the robot starts executing it. Before the chunk runs out, the "
        "model looks again and predicts a new chunk, which replaces the rest of the old one.",
        LEAD))
    s += fig_chunks()
    s.append(P("Chunks help in three ways:", LEAD))
    s.append(bullets([
        "<b>smooth motion:</b> consecutive actions come from one prediction, so they fit together;",
        "<b>slow inference is fine:</b> a 250 ms prediction does not stall the arm, because the "
        "current chunk keeps running;",
        "<b>less drift:</b> small errors do not pile up step after step."]))
    s.append(runin("Real-time chunking (RTC).",
        "The next chunk is computed while the current one runs, then blended in so the arm does "
        "not jerk at the switch. Intel uses it for Pi0.5 on Core Ultra; the PhysicalAI runtime "
        "offers it as {}.".format(c("RTCExecution"))))
    s += box("work", [
        "<b>Latency budget.</b> At 30 steps per second one step is 33 ms. If inference takes "
        "250 ms, about 8 steps pass during one prediction. Replanning every 25 steps (0.83 s) "
        "leaves plenty of margin in a 50-step chunk. If inference took 1.2 s (36 steps), the "
        "arm would nearly run out of planned motion before the next chunk arrived. That is when "
        "a smaller model, quantization or the iGPU starts to matter."])

    # ---------------------------------------------------------- 6
    s += H("choose")
    s.append(P("Physical AI Studio trains three policy families:", LEAD))
    s += btable([
        ["Policy", "Text input", "Size", "Strong at", "Weak at"],
        ["ACT", "no", "tens of millions of parameters", "fast to train and run; quick baseline",
         "one fixed behaviour; cannot be told which bowl"],
        ["SmolVLA", "yes", "\u2248 450 million", "a real VLA small enough for the laptop",
         "slower than ACT"],
        ["Pi0.5", "yes", "\u2248 3 billion",
         "generalises best to new positions and clutter; Intel has optimised it for Core Ultra "
         "Series 3", "needs a large training GPU and careful optimisation"],
    ], [52, 46, 88, 150, 134])
    s.append(P(
        "A practical order: <b>ACT first</b>, to prove the record\u2013train\u2013deploy chain in "
        "hours; <b>SmolVLA next</b>, for the instruction switch in Section {}; <b>Pi0.5</b> if "
        "time and credit remain.".format(ref("task"))))
    s += box("plain", [
        "The instruction is the steering wheel. Only a policy with text input can be told at run "
        "time which bowl to use."])

    # ---------------------------------------------------------- 7
    s += H("task")
    s.append(P(
        "The arm must pick a brick, show it to the inspection camera and drop it in the good or "
        "the defective bowl. The VLA cannot know the bowl before the brick is inspected, and "
        "judging defects is Anomalib\u2019s job. So the run is split into three instructions for "
        "<i>one</i> model, with Anomalib in the middle:", LEAD))
    s += fig_task()
    s.append(P("Record three kinds of episodes into one dataset:", LEAD))
    s += btable([
        ["", "Instruction (exact text)", "Starts", "Ends", "Episodes"],
        ["A", "pick up the brick and show it to the inspection camera",
         "brick on table, arm at rest", "brick held still at the camera", "50\u201380"],
        ["B", "put the brick in the good bowl", "brick at the camera",
         "brick dropped, arm at rest", "25\u201330"],
        ["C", "put the brick in the defective bowl", "brick at the camera",
         "brick dropped, arm at rest", "25\u201330"],
    ], [18, 168, 100, 128, 56])
    s.append(bullets([
        "Copy and paste the instruction. One changed character is a different instruction.",
        "End task A in the same steady hold every time: it is where B and C begin.",
        "In A, vary the brick\u2019s position and rotation a lot, and add some clutter.",
        "In B and C, use good <i>and</i> defective bricks, so the instruction decides the bowl, "
        "not the brick\u2019s look.",
        "Delete fumbled episodes. Never move a camera once recording has started."]))
    s.append(P("The glue that closes the loop is short:", LEAD))
    s += code("""
while bricks_left():
    run_policy("pick up the brick and show it to the inspection camera")
    wait_until_still(near=INSPECT_POSE)
    score = anomalib(inspection_camera.read())
    if score > THRESHOLD:
        run_policy("put the brick in the defective bowl")
    else:
        run_policy("put the brick in the good bowl")
""")
    s.append(runin("The inspection side.",
        "Anomalib learns from <i>good</i> bricks only and flags anything unusual. Install it in "
        "its own environment, so its PyTorch does not clash with Studio\u2019s, and train it on "
        "the laptop\u2019s Intel GPU: it does not need cloud credit. PatchCore gives very good "
        "results in 3\u20135 minutes; PaDiM trains in about a minute. A fixed crop around the "
        "brick and steady lighting matter more than the choice of model."))
    s += code("""
uv venv anomalib-env && source anomalib-env/bin/activate
uv pip install "anomalib[openvino,xpu]"
""")

    # ---------------------------------------------------------- 8
    s += H("studio")
    s.append(P(
        "Studio does the motor work for you: finding the arms, calibration, teleoperation, "
        "recording and test runs. The steps, from the hackathon branch:"))
    s.append(numbered([
        "<b>Start Studio.</b> Backend on port 7860, interface at "
        "{}.".format(c("http://localhost:3000")),
        "<b>Hugging Face.</b> Put a read-only token in <i>Settings \u2192 General \u2192 Hugging "
        "Face</i>. SmolVLA and Pi0.5 download their weights with it.",
        "<b>Modal.</b> Training runs on Modal\u2019s NVIDIA GPUs; each participant gets $100 of "
        "credit. Log the backend in with {}, then use <i>Settings \u2192 Training Targets \u2192 "
        "Modal Provider \u2192 Deploy and connect</i>. The first build takes several "
        "minutes.".format(c("modal setup")),
        "<b>Configure.</b> Project \u2192 robots (follower and leader) \u2192 cameras \u2192 "
        "environment. Calibrate both arms in the robot setup wizard.",
        "<b>Record.</b> Create a dataset with a default task, then <i>Add episode</i>. The loop: "
        "reset the bricks, confirm the task, \u2192 to start, demonstrate, \u2192 to accept or "
        "\u2190 to discard. Record at least 50 episodes before training.",
        "<b>Review.</b> Replay each episode; the joint-angle graph exposes the bad ones. Delete "
        "them.",
        "<b>Train.</b> On the Models page pick the policy, set <i>Run on</i> to your Modal target "
        "and choose a GPU. Watch the loss curve. Stopping early still saves and exports the model.",
        "<b>Test.</b> Studio exports to PyTorch, OpenVINO, ONNX and ExecuTorch. Try the model on "
        "the arm with <i>Run model</i>."]))
    s += code("""
git clone -b junction2026 https://github.com/open-edge-platform/physical-ai-studio.git

# terminal 1, from the folder you cloned into: the backend
cd physical-ai-studio/application/backend && cp .env.example .env
uv run --extra xpu physicalai-studio serve

# terminal 2, from the same folder: the interface
cd physical-ai-studio/application/ui && npm install && npm run start

# once, as the user that runs the backend: log it in to Modal
uvx --python 3.13 --from 'modal[api-proxy-support]>=1.6.3,<2' modal setup
""")
    s += box("work", [
        "<b>Which GPU.</b> Studio\u2019s own cloud presets put ACT and SmolVLA on a 16 GB GPU and "
        "Pi0.5 on a 48 GB one. On Modal that means an <b>A10G</b> (24 GB) for ACT and SmolVLA, "
        "and an <b>A100</b> (80 GB) for Pi0.5. Run a small ACT job first: a job keeps billing "
        "even when Studio is closed, for up to 24 hours."])
    s += box("plain", [
        "A serial port can be open in only one program. Close the jog script before Studio "
        "connects to an arm, and the other way round."])

    # ---------------------------------------------------------- 9
    s += H("between")
    s.append(P(
        "Studio\u2019s <i>Run model</i> sends the model\u2019s actions straight to the arm. To put "
        "your own logic in between, run the exported model with the OpenVINO PhysicalAI runtime "
        "({}). Its loop is simple:".format(c("RobotRuntime")), LEAD))
    s += code("""
while running:
    state, frames = read_robot_and_cameras()
    action = action_source.update(state, frames, step)    # (2) decides
    action = on_action_ready(action)                      # (1) your layer
    robot.send_action(action)                             # (3) drives the motors
    sleep_until_next_tick()
""")
    s += fig_hooks()
    s.append(runin("1. A callback.",
        "{} receives every action and returns a changed one: joint limits, smoothing, speed "
        "limits. The runtime ships {} as an example. If a callback raises an exception, the run "
        "stops, which gives you an emergency stop for free.".format(
            c("on_action_ready"), c("LowPassFilterCallback"))))
    s.append(runin("2. An action source.",
        "Decides what each tick\u2019s action is. Wrap {} to switch instructions, call Anomalib "
        "or insert scripted moves.".format(c("PolicySource"))))
    s.append(runin("3. A robot wrapper.",
        "Any class with {}, {}, {}, {}, {} and {} counts as a robot. Wrap the SO-101 driver to "
        "read servo load, add an inner loop, or translate an action space "
        "(Section {}).".format(c("connect"), c("disconnect"), c("get_observation"),
                               c("send_action"), c("is_connected"), c("joint_names"),
                               ref("space"))))
    s += code("""
import numpy as np

class SafetyLayer:
    def __init__(self, low, high, max_step):
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.max_step = np.asarray(max_step)          # per tick, per joint
        self.last = None

    def on_action_ready(self, *, action, step):
        a = np.clip(action, self.low, self.high)                    # joint limits
        if self.last is not None:                                   # speed limit
            a = self.last + np.clip(a - self.last, -self.max_step, self.max_step)
        self.last = a.copy()
        return a

    def on_action_sent(self, *, action, step):
        pass

runtime = RobotRuntime(robot=robot, fps=30, callbacks=[SafetyLayer(low, high, max_step)],
                       action_source=PolicySource(model=model, execution=AsyncExecution()))
""")
    s.append(P("<b>Speed.</b> The SO-101 driver sends position targets only. It sets the "
               "servos\u2019 acceleration to the maximum when it connects and ignores the "
               "{} hint. So the arm\u2019s speed is the size of the step between consecutive "
               "targets. Three places to control it:".format(c("goal_time")), H2.clone(
                   "h2body", fontName="Serif", fontSize=10.5, leading=13.6, spaceBefore=4,
                   alignment=TA_JUSTIFY, spaceAfter=6, keepWithNext=0)))
    s.append(bullets([
        "<b>a step limit in a callback</b>, as above: per joint, and it can depend on the "
        "situation (slower near the bricks);",
        "<b>the servos\u2019 own speed registers</b>, written by a robot wrapper: a hardware "
        "safety net, set high so it only catches abnormal jumps;",
        "<b>scaling the whole policy</b> faster or slower: this changes cycle time, so measure "
        "the success rate at each speed."]))
    s += box("work", [
        "<b>A wrist speed limit.</b> Limit the wrist to 90\u00b0/s. With an interpolator "
        "multiplier of 4 at 30 Hz the robot gets 120 commands per second, so "
        "max_step = 90 / 120 = 0.75\u00b0 per tick. A target 3\u00b0 away is then reached in 4 "
        "ticks, 33 ms, instead of in one jump."])
    s += box("plain", [
        "Keep the layer gentle in normal motion. The policy learned from unfiltered "
        "demonstrations; heavy smoothing makes the arm lag and shows the model states it has "
        "never seen."])

    # ---------------------------------------------------------- 10
    s += H("space")
    s.append(P(
        "Instead of correcting actions after the model, you can change <i>what the model "
        "outputs</i>. With top-down grasping, a Duplo pick needs only x, y, z, a rotation about "
        "the vertical, and the gripper: five numbers. Forward kinematics (FK) turns joint angles "
        "into a gripper pose; inverse kinematics (IK) turns a pose back into joint angles. The "
        "model then cannot ask for an impossible combination of joints.", LEAD))
    s += fig_space()
    s += box("work", [
        "<b>The arm in its vertical plane.</b> Take shoulder and elbow as two links of "
        "L<sub>1</sub> = L<sub>2</sub> = 0.12 m (illustrative) and a target at reach r = 0.15 m, "
        "height z = 0.05 m.",
        "cos q<sub>2</sub> = (r<super>2</super> + z<super>2</super> \u2212 L<sub>1</sub>"
        "<super>2</super> \u2212 L<sub>2</sub><super>2</super>) / (2 L<sub>1</sub>L<sub>2</sub>) "
        "= (0.025 \u2212 0.0288) / 0.0288 = \u22120.132, so q<sub>2</sub> = 97.6\u00b0.",
        "q<sub>1</sub> = atan2(z, r) \u2212 atan2(L<sub>2</sub> sin q<sub>2</sub>, L<sub>1</sub> "
        "+ L<sub>2</sub> cos q<sub>2</sub>) = 18.4\u00b0 \u2212 48.8\u00b0 = \u221230.4\u00b0.",
        "For a gripper pointing straight down, the wrist takes what is left: "
        "q<sub>3</sub> = \u221290\u00b0 \u2212 q<sub>1</sub> \u2212 q<sub>2</sub> = "
        "\u2212157.2\u00b0. Check with FK: 0.12 cos(\u221230.4\u00b0) + 0.12 cos(67.2\u00b0) = "
        "0.150 m. The three joints always move together: one cannot change without the others."])
    s.append(P("The pipeline, using what Studio already supports:", LEAD))
    s.append(numbered([
        "record normally in Studio, in joint angles;",
        "export the dataset (LeRobot v3 format);",
        "convert <i>both</i> the state and the actions with FK;",
        "import the converted dataset into Studio and train as usual;",
        "deploy with a robot wrapper: FK in {}, IK in {}.".format(
            c("get_observation"), c("send_action"))]))
    s.append(P("<b>Risks.</b> SmolVLA was pretrained mostly on joint-angle data, so a new "
               "space throws away part of what it knows and may need <i>more</i> "
               "demonstrations. The SO-101 cannot reach every pose, so IK must clamp, not crash. "
               "Calibration errors become position errors. Studio\u2019s <i>Run model</i> will not "
               "understand the new space, so you deploy with your own runtime."))
    s += box("plain", [
        "Get the joint-angle pipeline working first. Then build this as an experiment and measure "
        "both. Two systems compared with numbers make a far stronger story for the judges than "
        "one."])

    # ---------------------------------------------------------- 11
    s += H("laptop")
    s.append(P(
        "The Core Ultra laptop has three engines. A sensible starting split, to be confirmed by "
        "measurement:", LEAD))
    s += btable([
        ["Part", "Start on", "Why"],
        ["VLA policy (OpenVINO)", "integrated GPU", "the largest model, and the most compute"],
        ["Anomalib (OpenVINO)", "NPU", "small and steady, runs once per brick"],
        ["Control loop, glue, cameras", "CPU", "simple work that must answer quickly"],
    ], [150, 90, 230])
    s.append(P(
        "OpenVINO can lower precision to FP16 or INT8, which makes inference faster and cooler. "
        "Check the success rate after every change: a faster model that drops bricks loses "
        "points. The runtime has callbacks that log to JSONL, monitor the system and show "
        "latency in Rerun, which helps with the evidence below."))
    s += btable([
        ["Criterion", "Points", "What judges look at"],
        ["Task success", "25", "accuracy, cycle time, robustness to shifted and cluttered bricks"],
        ["Intel tech integration", "25", "NPU and iGPU offloading, OpenVINO, Robotics AI Suite"],
        ["Dataset and pipeline quality", "20", "clean teleoperation data, fine-tuning in Studio"],
        ["System efficiency", "20", "latency per step, thermal headroom, power"],
        ["Creativity", "10", "ideas beyond the baseline"],
    ], [140, 45, 285])
    s.append(P("Judges want measurements, not claims. Log latency per inference step, time per "
               "sorting cycle, success rate over a fixed number of bricks, power and temperature, "
               "from the first working run."))

    # ---------------------------------------------------------- 12
    s += H("traps")
    s += btable([
        ["Trap", "What to do"],
        ["A camera moved after recording", "The policy fails though \u201cnothing changed\u201d. "
         "Fix the mounts; re-record if one moved."],
        ["Instruction text differs by one character", "The model sees a new instruction. Keep "
         "the three strings in one file and copy them."],
        ["Fumbled demonstrations kept", "The policy copies them. Review and delete before "
         "training."],
        ["Heavy filtering after the model", "Lag and unseen states. Keep limits above your "
         "demonstration speed."],
        ["State and action in different spaces", "Nonsense outputs. Convert both, in training "
         "and at run time."],
        ["Two programs on one serial port", "\u201cCould not open port\u201d. Close one before "
         "starting the other."],
        ["A Modal job left running", "Credit drains even with Studio closed. Stop it in the "
         "Modal dashboard."],
        ["No Hugging Face token", "Weight downloads fail. Set a read-only token before "
         "training SmolVLA or Pi0.5."],
        ["Torque on with an old goal", "The arm snaps to it. Set goal = present position "
         "before enabling torque."],
        ["Unit mismatch", "Ticks, degrees and normalized values differ. Check the driver\u2019s "
         "unit."],
    ], [175, 295])

    # ---------------------------------------------------------- 13
    s += H("cheat")
    s += btable([
        ["Question", "Answer"],
        ["VLM", "image + text \u2192 text"],
        ["VLA", "images + joint angles + instruction \u2192 action chunk"],
        ["Episode", "one recorded demonstration"],
        ["Action chunk", "the next \u2248 50 joint targets, replaced before it runs out"],
        ["ACT or a VLA?", "ACT: no text, fast. VLA: takes an instruction, generalises better"],
        ["How much data", "at least 50 clean, varied episodes"],
        ["Where policies train", "Modal: A10G for ACT and SmolVLA, A100 for Pi0.5"],
        ["Where Anomalib trains", "the laptop\u2019s Intel GPU, in its own environment"],
        ["Studio address", c("http://localhost:3000")],
        ["Export formats", "PyTorch, OpenVINO, ONNX, ExecuTorch"],
        ["Change actions before the motors", "callback " + c("on_action_ready")],
        ["Change what the robot decides", "a custom action source"],
        ["Change how the motors are driven", "a robot wrapper"],
        ["What sets the arm\u2019s speed", "the step between consecutive targets"],
    ], [160, 260])

    # ---------------------------------------------------------- 14
    s += H("try")
    s.append(P("Do these before the hackathon if you can; the jog script is enough for the "
               "first one."))
    s.append(numbered([
        "Read the six joint angles at your inspection pose and above both bowls. Write them down.",
        "A policy predicts 50-step chunks at 30 Hz and inference takes 400 ms. How many steps "
        "run during one prediction? Is replanning every 25 steps safe?",
        "You recorded 60 episodes of 10 s at 30 fps. How many samples is that? What would you "
        "add to make the policy robust to clutter?",
        "With Section {}, find q<sub>1</sub>, q<sub>2</sub> and q<sub>3</sub> for r = 0.18 m, "
        "z = 0.02 m and L<sub>1</sub> = L<sub>2</sub> = 0.12 m. Is the point "
        "reachable?".format(ref("space")),
        "Write a callback that stops the run if any joint target jumps more than 20\u00b0 in one "
        "tick."]))
    s.append(Spacer(1, 6))
    s.append(P("<b>Sources.</b> Physical AI Studio, branch "
               "<font face='Mono' size='8'>junction2026</font>: "
               "<font face='Mono' size='8'>github.com/open-edge-platform/physical-ai-studio</font> "
               "(JUNCTION2026.md, recording, training and Modal docs). OpenVINO PhysicalAI "
               "runtime: <font face='Mono' size='8'>github.com/openvinotoolkit/physicalai</font> "
               "(runtime, callbacks, robot interface, SO-101 driver). Anomalib: "
               "<font face='Mono' size='8'>github.com/open-edge-platform/anomalib</font>.", SMALL))
    return s


def build(target, pages):
    doc = GuideDoc(target, title="Vision-language-action models, from the ground up",
                   author="Junction 2026 study notes",
                   subject="VLAs for the Junction 2026 Intel Physical AI Challenge",
                   creator="build_vla_guide.py")
    doc.build(make_story(pages))
    return doc.toc_pages


if __name__ == "__main__":
    first = build(io.BytesIO(), {})          # pass 1: find section pages
    final = build(OUT, first)                # pass 2: contents with page numbers
    assert final == first, "pagination changed between passes"
    print("wrote", OUT)
    print("sections:", final)
