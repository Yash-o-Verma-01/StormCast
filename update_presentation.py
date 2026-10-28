"""
Script to update the StormCast presentation by inserting a dedicated
comparison table slide without modifying or overwriting original files.
Saves output as StormCast_Presentation_Updated.pptx.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def add_comparison_slide(prs):
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Theme colors
    NAVY = RGBColor(26, 44, 76)          # #1A2C4C
    DARK_BLUE = RGBColor(33, 63, 110)    # #213F6E
    STORM_BLUE = RGBColor(41, 128, 185)  # #2980B9
    ACCENT_AMBER = RGBColor(217, 119, 6) # #D97706
    WHITE = RGBColor(255, 255, 255)
    DARK_TEXT = RGBColor(30, 41, 59)     # #1E293B
    BORDER_COLOR = RGBColor(203, 213, 225)
    ROW_TINT = RGBColor(241, 245, 249)   # Light slate tint
    HIGHLIGHT_TINT = RGBColor(238, 242, 255) # Soft indigo tint for StormCast

    # Header banner (consistent with all slides)
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.3))
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()

    # Category Eyebrow (16 pt bold Amber)
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.7), Inches(0.35))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = "SYSTEM COMPARISON & BENCHMARK"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ACCENT_AMBER

    # Slide Title (34 pt bold White)
    tb2 = slide.shapes.add_textbox(Inches(0.8), Inches(0.48), Inches(11.7), Inches(0.72))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0
    p2 = tf2.paragraphs[0]
    p2.text = "Existing Forecasting Systems vs. StormCast"
    p2.font.size = Pt(34)
    p2.font.bold = True
    p2.font.color.rgb = WHITE

    # Create Table
    # 4 rows (1 header + 3 data rows), 4 columns
    rows = 4
    cols = 4
    left = Inches(0.7)
    top = Inches(1.5)
    width = Inches(11.933)
    height = Inches(5.6)

    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    # Column widths
    table.columns[0].width = Inches(2.4)  # System
    table.columns[1].width = Inches(3.2)  # Key Capabilities
    table.columns[2].width = Inches(3.1)  # How StormCast Differs
    table.columns[3].width = Inches(3.233) # Limitations & Scope

    headers = [
        "System / Provider",
        "Key Capabilities",
        "How StormCast Differs",
        "Limitations & Scope"
    ]

    for col_idx, text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.margin_left = Inches(0.12)
        cell.margin_right = Inches(0.12)
        cell.margin_top = Inches(0.1)
        cell.margin_bottom = Inches(0.1)
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    data = [
        (
            "National Weather Service\n(NOAA / NWS)",
            "• Physics-based numerical models (NWP: GFS, HRRR)\n• Continental Doppler radar network (NEXRAD)\n• Official severe weather watches & warnings",
            "• NWS relies on complex hydrodynamic equations on supercomputers taking hours\n• StormCast is a lightweight tabular ML model executing in <0.05 seconds",
            "• High computational cost; not portable to lightweight edge devices\n• StormCast prototype does NOT replace NWS warnings; currently lacks live radar"
        ),
        (
            "Commercial Weather APIs\n(e.g., OpenWeatherMap)",
            "• Multi-day rain, temperature, and cloud forecasts\n• Global REST API feeds for web/mobile apps\n• Aggregated station & satellite telemetry",
            "• OpenWeather wraps proprietary NWP models behind commercial paywalls\n• StormCast is 100% open-source, providing calibrated probability (0–100%)",
            "• Commercial black-box models; no storm risk calibration\n• StormCast currently uses manual sliders; plans to ingest OpenWeather API in future"
        ),
        (
            "StormCast\n(Current Academic Prototype)",
            "• Instant point-level storm occurrence classification\n• Calibrated probability gauge (0–100%)\n• 4-level hazard severity grading (Light to Severe)",
            "• Zero target leakage (pre-event features only)\n• Pure statistical ML (Random Forest, 83.7% Acc, 0.911 ROC-AUC) running in Streamlit",
            "• Current: Occurrence classification prototype (not yet a multi-hour future NWP forecast)\n• Future Scope: Ingest live NOAA APIs & satellite radar CNNs"
        )
    ]

    for row_idx, row_data in enumerate(data, start=1):
        is_stormcast = (row_idx == 3)
        for col_idx, text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            if is_stormcast:
                cell.fill.fore_color.rgb = HIGHLIGHT_TINT
            elif row_idx % 2 == 1:
                cell.fill.fore_color.rgb = ROW_TINT
            else:
                cell.fill.fore_color.rgb = WHITE

            cell.margin_left = Inches(0.12)
            cell.margin_right = Inches(0.12)
            cell.margin_top = Inches(0.08)
            cell.margin_bottom = Inches(0.08)

            tf = cell.text_frame
            tf.word_wrap = True
            
            lines = text.split("\n")
            for l_idx, line in enumerate(lines):
                p = tf.paragraphs[0] if l_idx == 0 else tf.add_paragraph()
                p.text = line
                if col_idx == 0 and l_idx == 0:
                    p.font.size = Pt(15)
                    p.font.bold = True
                    p.font.color.rgb = STORM_BLUE if not is_stormcast else NAVY
                elif col_idx == 0 and l_idx > 0:
                    p.font.size = Pt(13)
                    p.font.italic = True
                    p.font.color.rgb = DARK_TEXT
                else:
                    p.font.size = Pt(13.5)
                    p.font.color.rgb = DARK_TEXT

    return slide


def main():
    source_pptx = "StormCast_Presentation.pptx"
    output_pptx = "StormCast_Presentation_Updated.pptx"

    print(f"[*] Loading presentation: {source_pptx}")
    prs = Presentation(source_pptx)
    initial_count = len(prs.slides)
    print(f"[*] Initial slides: {initial_count}")

    # Add the comparison table slide
    print("[*] Appending System Comparison slide with professional formatting...")
    add_comparison_slide(prs)

    # Reorder slides so the comparison slide is Slide 9, right before Limitations/Future Scope
    # In python-pptx, new slide is at index 10. Let's move slide 10 to index 8 (Slide 9 position).
    slide_id_list = prs.slides._sldIdLst
    new_slide_elem = slide_id_list[-1]
    slide_id_list.insert(8, new_slide_elem)

    print(f"[*] Total slides in updated deck: {len(prs.slides)}")
    prs.save(output_pptx)
    print(f"[+] Saved updated presentation to: {output_pptx}")


if __name__ == "__main__":
    main()
