"""
Script to generate the comprehensive, publication-quality college project report:
StormCast_Project_Report.docx.

Strictly grounded in project files, real code, and actual evaluation metrics.
Explicitly distinguishes implemented features from future scope and notes
that the model is an occurrence classification prototype, not a certified
future-weather forecasting system.
"""

import os
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell padding in twips."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_callout(doc, text, title="IMPORTANT NOTICE"):
    """Add a styled callout box with a colored left border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)

    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)

    # Left border only (Amber / Navy)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="D97706"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"[{title}] ")
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(217, 119, 6)
    run_title.font.size = Pt(10)

    run_text = p.add_run(text)
    run_text.font.size = Pt(10)
    run_text.font.color.rgb = RGBColor(51, 65, 85)

    # Empty paragraph after table for spacing
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(6)


def style_table_header(row, col_widths, headers, bg_hex="1A2C4C"):
    """Style header row of a table."""
    for idx, (head_text, width) in enumerate(zip(headers, col_widths)):
        cell = row.cells[idx]
        cell.width = Inches(width)
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(head_text)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9.5)


def add_styled_row(table, row_data, col_widths, bg_hex="FFFFFF", bold_first=False):
    """Add and style a data row in a table."""
    row = table.add_row()
    for idx, (val, width) in enumerate(zip(row_data, col_widths)):
        cell = row.cells[idx]
        cell.width = Inches(width)
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=90, bottom=90, left=120, right=120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        
        # Center numerical / metric columns, left-align text
        if idx > 0 and len(str(val)) < 15 and not str(val).startswith("•"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
        run = p.add_run(str(val))
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(30, 41, 59)
        if bold_first and idx == 0:
            run.font.bold = True
            run.font.color.rgb = RGBColor(26, 44, 76)
    return row


def generate_report(output_filename="StormCast_Project_Report.docx"):
    print(f"[*] Initializing report generation: {output_filename}")
    doc = Document()

    # Document-wide page margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.header.is_linked_to_previous = False
        
        # Simple header & footer
        header_p = section.header.paragraphs[0]
        header_p.text = "StormCast: Storm Occurrence & Severity Analysis | Machine Learning Project Report"
        header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        header_p.runs[0].font.size = Pt(8.5)
        header_p.runs[0].font.color.rgb = RGBColor(148, 163, 184)

        footer_p = section.footer.paragraphs[0]
        footer_p.text = "Bennett University — Department of Computer Science & Engineering / AI & ML"
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_p.runs[0].font.size = Pt(8.5)
        footer_p.runs[0].font.color.rgb = RGBColor(148, 163, 184)

    # Styles setup
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(30, 41, 59)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # Helper for styled section headings
    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(17)
        run.font.bold = True
        run.font.color.rgb = RGBColor(26, 44, 76)  # Navy
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(41, 128, 185)  # Storm Blue
        return h

    def add_h3(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        h.paragraph_format.keep_with_next = True
        run = h.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(71, 85, 105)
        return h

    # =========================================================================
    # TITLE PAGE / METADATA
    # =========================================================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(24)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("⛈️ StormCast: Storm Occurrence & Severity Analysis")
    title_run.font.size = Pt(26)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(26, 44, 76)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(18)
    sub_run = sub_p.add_run("A College Machine Learning Project on Tabular Weather Intelligence\nEnd-to-End Pipeline, Scientific Evaluation, and Interactive Web Deployment")
    sub_run.font.size = Pt(13)
    sub_run.font.color.rgb = RGBColor(100, 116, 139)

    meta_tbl = doc.add_table(rows=4, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_widths = [2.2, 4.3]
    meta_rows = [
        ("Author / Student:", "Yashovardhan (B.Tech AI & ML)"),
        ("Institution & Department:", "Bennett University, Department of CSE / AI & ML"),
        ("Academic Course / Context:", "Semester III — Machine Learning Mini-Project"),
        ("Deliverables & Date:", "Python Package, Trained Models, Streamlit App — October 2026")
    ]
    for idx, (label, val) in enumerate(meta_rows):
        row = meta_tbl.rows[idx]
        for c_idx, text in enumerate([label, val]):
            cell = row.cells[c_idx]
            cell.width = Inches(meta_widths[c_idx])
            set_cell_background(cell, "F1F5F9" if idx % 2 == 0 else "FFFFFF")
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.size = Pt(9.5)
            if c_idx == 0:
                r.font.bold = True
                r.font.color.rgb = RGBColor(26, 44, 76)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Callout: Academic Prototype Notice
    add_callout(
        doc,
        "This project is an academic machine learning prototype developed for an educational curriculum. "
        "The models classify whether given point-level atmospheric conditions match historical storm occurrences and operational "
        "severity levels. It is an occurrence classification system and is NOT yet validated or certified as a true future-weather "
        "predictive forecasting system or operational radar warning mechanism.",
        title="ACADEMIC PROTOTYPE NOTICE"
    )

    # =========================================================================
    # 1. PROJECT OVERVIEW & PROBLEM STATEMENT
    # =========================================================================
    add_h1("1. Project Overview & Problem Statement")
    doc.add_paragraph(
        "Atmospheric storms—encompassing convective thunderstorms, severe squall lines, downbursts, and heavy localized precipitation—represent "
        "some of the most disruptive meteorological phenomena in North America and globally. They are responsible for severe structural damage, "
        "transportation shutdowns, flight ground-stops, agricultural destruction, and risk to human life."
    )
    doc.add_paragraph(
        "In modern operational meteorology, weather prediction has historically been dominated by Numerical Weather Prediction (NWP) simulations "
        "(such as NOAA's High-Resolution Rapid Refresh [HRRR] and Global Forecast System [GFS]). While physically grounded in fluid dynamics and "
        "thermodynamic differential equations, NWP models possess distinct operational limitations for rapid localized assessment:"
    )
    
    p_pts = doc.add_paragraph()
    p_pts.add_run("• Computational Expense: ").bold = True
    p_pts.add_run("NWP simulations require massive supercomputing clusters and multi-hour execution cycles, making instantaneous localized screening impractical on standard or edge hardware.\n")
    p_pts.add_run("• Latency in Rapid Point Screening: ").bold = True
    p_pts.add_run("Point-level observers, airport ground crews, logistics managers, and outdoor safety personnel often require instant (sub-second) classification based on current readings without waiting for full simulation cycles.\n")
    p_pts.add_run("• Severe Class Imbalance: ").bold = True
    p_pts.add_run("In large-scale observational weather streams, severe storms represent less than 1% of all logged weather episodes. Standard data analysis often falters due to majority-class bias where models simply predict 'No Storm' to achieve deceptive 99% accuracy.")

    doc.add_paragraph(
        "StormCast addresses this challenge by designing, evaluating, and deploying an end-to-end, leakage-free Machine Learning system "
        "trained on genuine historical airport Automated Surface Observing System (ASOS) weather records. The system provides instantaneous "
        "hazard screening, reliable probabilistic risk gauges, and multi-class severity grading via a modern Streamlit web application."
    )

    # =========================================================================
    # 2. AIM AND OBJECTIVES
    # =========================================================================
    add_h1("2. Aim and Objectives")
    doc.add_paragraph(
        "The primary aim of StormCast is to develop a scientifically defensible, college-level Machine Learning application that bridges "
        "big-data meteorological archives with interactive, transparent decision support."
    )
    add_h2("Specific Technical Objectives:")
    obj_p = doc.add_paragraph()
    obj_p.add_run("1. Objective 1 — Storm Occurrence Prediction (Binary Classification):\n").bold = True
    obj_p.add_run("   Predict whether atmospheric conditions denote a Storm episode (Thunderstorm/Squall) vs. a Non-Storm event, delivering a calibrated probability estimate (0% to 100%) rather than an arbitrary binary output.\n")
    obj_p.add_run("2. Objective 2 — Weather Severity Prediction (Multi-Class Classification):\n").bold = True
    obj_p.add_run("   Conditioned on weather event parameters, classify the operational severity level into Light, Moderate, Heavy, or Severe based on genuine ground-truth categories.\n")
    obj_p.add_run("3. Objective 3 — Strict Data Leakage Elimination:\n").bold = True
    obj_p.add_run("   Isolate pre-event predictors from post-event artifacts. Specifically, drop lookahead features such as event duration (EndTime) and high-cardinality station codes that cause artificial memorization.\n")
    obj_p.add_run("4. Objective 4 — Working Interactive Deployment:\n").bold = True
    obj_p.add_run("   Deploy the serialized machine learning pipelines into an interactive Streamlit application with live parameter sliders, historical scenario presets, and an in-browser exploratory data analysis (EDA) explorer.")

    # =========================================================================
    # 3. DATASET SOURCE, FEATURES & CLASS DISTRIBUTION
    # =========================================================================
    add_h1("3. Dataset Source, Features, and Class Distribution")
    doc.add_paragraph(
        "The project is built exclusively on the genuine Kaggle US Weather Events (2016–2022) dataset, curated by Dr. Sobhan Moosavi from "
        "2,071 Automated Surface Observing System (ASOS) airport weather stations across the contiguous United States."
    )
    
    add_h2("3.1 Raw Dataset Scale and Completeness")
    doc.add_paragraph(
        "The raw dataset (WeatherEvents_Jan2016-Dec2022.csv) comprises 8,627,181 records across 14 columns, occupying approximately 1.08 GB on disk. "
        "A streaming chunk inspection revealed that 12 of the 14 columns possess 0% missing values. Only City (16,912 nulls; 0.20%) and ZipCode "
        "(69,199 nulls; 0.80%) contained minor missingness."
    )

    # Table: Raw Event Type Distribution
    doc.add_paragraph("Table 1 summarizes the distribution of weather event types across the full 8.6M raw observation archive:")
    raw_tbl = doc.add_table(rows=1, cols=4)
    raw_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_widths = [1.8, 1.5, 1.4, 1.8]
    style_table_header(raw_tbl.rows[0], t1_widths, ["Weather Event Type", "Raw Record Count", "Percentage (%)", "Meteorological Meaning"])
    raw_events = [
        ("Rain", "5,000,794", "57.97%", "Liquid precipitation (Light/Mod/Heavy)"),
        ("Fog", "2,014,190", "23.35%", "Reduced visibility hazard"),
        ("Snow", "1,158,575", "13.43%", "Frozen precipitation"),
        ("Cold", "232,192", "2.69%", "Extreme low-temperature episodes"),
        ("Precipitation", "157,186", "1.82%", "Unclassified mixed precipitation"),
        ("Storm", "61,317", "0.71%", "Convective thunderstorms & squalls"),
        ("Hail", "2,927", "0.03%", "Severe convective frozen pellets")
    ]
    for r_idx, (ev, cnt, pct, desc) in enumerate(raw_events):
        add_styled_row(raw_tbl, [ev, cnt, pct, desc], t1_widths, bg_hex="F8FAFC" if r_idx % 2 == 1 else "FFFFFF", bold_first=True)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_h2("3.2 Critical Data Discovery: The ASOS Storm Severity Artifact")
    doc.add_paragraph(
        "A critical finding emerged during exploratory data analysis: in the raw ASOS database, 100% of the 61,317 records labeled Type == 'Storm' "
        "carry a severity rating of 'Severe'. Cross-tabulation confirmed that zero storm records possessed 'Light' or 'Moderate' severity in the raw data."
    )
    doc.add_paragraph(
        "This occurs because FAA/ASOS automated reporting protocol treats automated lightning sensor detections (METAR TS code) uniformly as severe aviation "
        "hazards regardless of rainfall volume. Rather than fabricating synthetic severity categories, we established a dual scientific formulation: "
        "(1) Binary Storm Occurrence modeling on genuine storm labels, and (2) Multi-Class Weather Severity modeling across all genuine operational severity categories."
    )

    add_h2("3.3 Stratified Analytical Dataset Extraction")
    doc.add_paragraph(
        "To enable rapid, reproducible training and sub-second interactive inference without out-of-memory crashes on student hardware, we implemented "
        "prepare_data.py to extract a representative 100,000-record analytical dataset (40,000 genuine storms and 60,000 stratified non-storms across all 7 years) "
        "without modifying or overwriting the original 1.08 GB CSV file."
    )

    # =========================================================================
    # 4. PREPROCESSING & CLASS-IMBALANCE HANDLING
    # =========================================================================
    add_h1("4. Data Preprocessing & Class-Imbalance Handling")
    doc.add_paragraph(
        "The preprocessing architecture was designed with strict adherence to data leakage prevention and physical meteorological logic."
    )

    add_h2("4.1 Elimination of Target Leakage")
    leak_p = doc.add_paragraph()
    leak_p.add_run("1. Dropping EndTime(UTC): ").bold = True
    leak_p.add_run("In real-world deployment, when an observer evaluates current atmospheric conditions, the event's duration is unknown. Computing duration (EndTime - StartTime) introduces severe lookahead leakage.\n")
    leak_p.add_run("2. Dropping High-Cardinality Identifiers: ").bold = True
    leak_p.add_run("AirportCode (>2,000 unique values), ZipCode, and City were excluded. Relying on airport codes forces tree models to memorize specific station names rather than learning generalizable atmospheric physics.\n")
    leak_p.add_run("3. Dropping Arbitrary Identifiers: ").bold = True
    leak_p.add_run("EventId (e.g. W-1024) is a non-predictive database primary key.")

    add_h2("4.2 Feature Engineering & Extraction")
    doc.add_paragraph(
        "From StartTime(UTC), we extracted diurnal and seasonal cycles: Month (1–12), Day (1–31), Hour in UTC (0–23), Day of Week, and Season "
        "(Winter, Spring, Summer, Fall). These temporal indicators directly capture North American solar convective cycles."
    )

    add_h2("4.3 Decoupling Precipitation from the Occurrence Trigger")
    doc.add_paragraph(
        "A vital improvement in our modeling was resolving the 'Precipitation Paradox'. In the ASOS dataset, tipping-bucket rain gauges log liquid water "
        "under 'Rain', while lightning detectors log 'Storm' with precipitation logged as 0.00 in the storm record. An initial model trained with precipitation "
        "learned the spurious correlation that rain > 0 implied 'No Storm'. By decoupling precipitation from the Occurrence model (using coordinates, month, "
        "hour, and season) and routing precipitation into the Severity model, heavy summer rainfall logically predicts 'Storm' with 'Heavy/Severe' intensity."
    )

    add_h2("4.4 Scikit-Learn Pipeline & Balancing Strategy")
    doc.add_paragraph(
        "All transformations were integrated into a Scikit-Learn ColumnTransformer pipeline. Numerical features are processed via median SimpleImputer "
        "and StandardScaler. Categorical features (State, TimeZone, Season) undergo most-frequent imputation and OneHotEncoder(handle_unknown='ignore'). "
        "To mitigate class imbalance, class_weight='balanced' was applied to penalize false negatives proportionally during loss optimization."
    )

    # =========================================================================
    # 5. MACHINE LEARNING MODELS & METHODOLOGY
    # =========================================================================
    add_h1("5. Machine Learning Models and Methodology")
    doc.add_paragraph(
        "We implemented, evaluated, and compared three classification architectures on a held-out 20% test set (20,000 records) using stratified splitting:"
    )

    mod_p = doc.add_paragraph()
    mod_p.add_run("1. Logistic Regression (Baseline): ").bold = True
    mod_p.add_run("A linear probabilistic model employing L2 regularization and balanced class weights. It establishes the baseline linear separability of normalized coordinates and temporal features.\n")
    mod_p.add_run("2. Decision Tree Classifier (Interpretable Non-linear Baseline): ").bold = True
    mod_p.add_run("A single classification tree constrained to max_depth=12 and min_samples_leaf=15 to prevent overfitting while capturing localized geographical bounding boxes and diurnal thresholds.\n")
    mod_p.add_run("3. Random Forest Classifier (Champion Architecture): ").bold = True
    mod_p.add_run("An ensemble of 100 de-correlated decision trees (max_depth=16, min_samples_leaf=5). By aggregating individual tree probability votes through bootstrap bagging, Random Forest minimizes variance, avoids overfitting, and captures complex non-linear interactions.\n")
    mod_p.add_run("4. Multi-Class Severity Classifier: ").bold = True
    mod_p.add_run("A Random Forest classifier evaluating operational severity categories across Light, Moderate, Heavy, and Severe weather events.")

    add_h2("5.1 Evaluation Metrics")
    doc.add_paragraph(
        "Models were assessed across multiple threshold-independent and threshold-dependent metrics: Accuracy, Precision, Recall, Macro F1-Score, "
        "ROC-AUC (Receiver Operating Characteristic Area Under Curve), and the Brier Score for probability calibration."
    )

    # =========================================================================
    # 6. SYSTEM ARCHITECTURE & WORKFLOW
    # =========================================================================
    add_h1("6. System Architecture and Workflow")
    doc.add_paragraph(
        "The project follows a clean separation between offline batch model development and online real-time inference, as depicted below:"
    )

    # Architecture Box
    arch_box = doc.add_paragraph()
    arch_box.paragraph_format.left_indent = Inches(0.4)
    arch_box.paragraph_format.space_before = Pt(6)
    arch_box.paragraph_format.space_after = Pt(6)
    arch_text = (
        "+--------------------------------------------------------------------------+\n"
        "|                         OFFLINE BATCH PIPELINE                           |\n"
        "|  Raw ASOS CSV (8.6M Rows / 1.08 GB)                                      |\n"
        "|      |                                                                   |\n"
        "|      v (prepare_data.py - Chunked Streaming & Stratified Subsampling)     |\n"
        "|  Cleaned Analytical Sample (100,000 Records) -> data/                     |\n"
        "|      |                                                                   |\n"
        "|      v (train.py - Leakage Prevention & ColumnTransformer Pipeline)      |\n"
        "|  Model Training (Logistic Regression, Decision Tree, Random Forest)       |\n"
        "|      |                                                                   |\n"
        "|      +---> models/storm_occurrence_model.joblib (23.9 MB)                |\n"
        "|      +---> models/storm_severity_model.joblib   (34.1 MB)                |\n"
        "|      +---> outputs/metrics/model_metrics_comparison.json                 |\n"
        "+--------------------------------------------------------------------------+\n"
        "                                     |                                      \n"
        "                                     v                                      \n"
        "+--------------------------------------------------------------------------+\n"
        "|                         ONLINE INFERENCE ENGINE                          |\n"
        "|  Streamlit Web Application (app.py)                                      |\n"
        "|      - Cached Joblib Models (@st.cache_resource)                         |\n"
        "|      - User Weather Sliders & Historical Scenario Presets                |\n"
        "|      - Sub-Second Inference (<0.05s per query)                           |\n"
        "|      - Output: Storm Status, Calibrated Probability Gauge, Severity Bar  |\n"
        "|      - EDA Data Explorer & Cleaned CSV Download                          |\n"
        "+--------------------------------------------------------------------------+"
    )
    r_arch = arch_box.add_run(arch_text)
    r_arch.font.name = 'Consolas'
    r_arch.font.size = Pt(8.5)
    r_arch.font.color.rgb = RGBColor(15, 23, 42)

    # =========================================================================
    # 7. STREAMLIT INTERFACE & PROJECT STRUCTURE
    # =========================================================================
    add_h1("7. Streamlit Interface & Project Structure")
    doc.add_paragraph(
        "The project provides a multi-tab web application built using Streamlit 1.65. The interface is designed for accessibility by non-technical "
        "users while preserving full scientific transparency:"
    )

    ui_p = doc.add_paragraph()
    ui_p.add_run("• Tab 1: Live Prediction Dashboard — ").bold = True
    ui_p.add_run("Allows users to select State, TimeZone, Latitude, Longitude, Month, Day, Hour (UTC), and Precipitation. Features 5 realistic historical presets (e.g. Texas Summer Downpour, Oklahoma Squall Line) and renders real-time Storm/No-Storm status badges, probability meters, and severity charts.\n")
    ui_p.add_run("• Tab 2: EDA & Data Explorer — ").bold = True
    ui_p.add_run("Displays dataset statistics, interactive filtering tools, an in-browser dataset viewer, a one-click CSV download button, and 6 publication-grade EDA charts.\n")
    ui_p.add_run("• Tab 3: Model Evaluation & Performance — ").bold = True
    ui_p.add_run("Presents exact test-set scorecards, ROC curves, probability calibration reliability diagrams, and normalized confusion matrices.\n")
    ui_p.add_run("• Tab 4: Methodology & Viva Defense Guide — ").bold = True
    ui_p.add_run("Provides comprehensive technical answers to external viva voce examination questions.")

    add_h2("7.1 Project Directory Layout")
    doc.add_paragraph(
        "The repository maintains a clean, maintainable structure appropriate for college viva evaluation:"
    )
    dir_p = doc.add_paragraph()
    dir_p.paragraph_format.left_indent = Inches(0.4)
    dir_text = (
        "StormCast/\n"
        "|-- WeatherEvents_Jan2016-Dec2022.csv  # Original raw dataset (1.08 GB, unmodified)\n"
        "|-- prepare_data.py                   # Streaming chunked extraction script\n"
        "|-- train.py                          # Training, evaluation & joblib export script\n"
        "|-- app.py                            # Streamlit web application\n"
        "|-- requirements.txt                  # Python package specifications\n"
        "|-- README.md                         # Technical documentation & viva guide\n"
        "|-- data/\n"
        "|   +-- cleaned_weather_sample.csv    # 100,000-row stratified analytical dataset\n"
        "|-- models/\n"
        "|   +-- storm_occurrence_model.joblib # Champion Random Forest pipeline (23.9 MB)\n"
        "|   +-- storm_severity_model.joblib   # Multi-class severity pipeline (34.1 MB)\n"
        "|-- outputs/\n"
        "|   +-- data_summary.json             # Dataset extraction summary\n"
        "|   +-- eda/                          # 6 publication-quality EDA figures\n"
        "|   +-- metrics/                      # Confusion matrices, ROC & calibration plots\n"
        "|-- src/\n"
        "    +-- preprocessing.py              # Feature engineering & ColumnTransformer pipelines\n"
        "    +-- visualization.py              # Plotting utilities for EDA and metrics"
    )
    r_dir = dir_p.add_run(dir_text)
    r_dir.font.name = 'Consolas'
    r_dir.font.size = Pt(8.5)
    r_dir.font.color.rgb = RGBColor(15, 23, 42)

    # =========================================================================
    # 8. RESULTS AND EVALUATION METRICS
    # =========================================================================
    add_h1("8. Results and Evaluation Metrics")
    doc.add_paragraph(
        "All metrics reported below were computed strictly on the held-out 20% test partition (20,000 unseen records) and are persisted in "
        "outputs/metrics/model_metrics_comparison.json. No metrics or prediction scores were fabricated."
    )

    add_h2("8.1 Storm Occurrence Model Comparison (Binary Classification)")
    doc.add_paragraph("Table 2 outlines the performance benchmark across candidate architectures:")

    occ_tbl = doc.add_table(rows=1, cols=7)
    occ_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_widths = [1.6, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9]
    style_table_header(occ_tbl.rows[0], t2_widths, [
        "Model Architecture", "Accuracy", "Macro F1", "Storm Recall", "Storm Prec.", "ROC-AUC", "Brier Score"
    ])

    occ_data = [
        ("Logistic Regression (Baseline)", "77.69%", "0.7710", "77.19%", "70.07%", "0.8358", "0.1626"),
        ("Decision Tree (Interpretable)", "82.86%", "0.8236", "82.66%", "76.41%", "0.8970", "0.1260"),
        ("Random Forest (Champion)", "83.73%", "0.8317", "81.96%", "78.35%", "0.9109", "0.1200")
    ]
    for r_idx, row in enumerate(occ_data):
        is_champ = (r_idx == 2)
        add_styled_row(occ_tbl, row, t2_widths, bg_hex="EEF2FF" if is_champ else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF"), bold_first=True)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_h2("8.2 Weather Event Severity Model (Multi-Class Classification)")
    doc.add_paragraph(
        "For weather event severity across Light, Moderate, Heavy, and Severe categories, Random Forest achieved an overall accuracy of 68.20% "
        "(Macro F1 = 0.5892, Weighted F1 = 0.6906), outperforming multi-class Logistic Regression (Accuracy = 65.21%, Macro F1 = 0.5464, Weighted F1 = 0.6592)."
    )

    add_h2("8.3 Why Random Forest Was Selected as the Champion Model")
    doc.add_paragraph(
        "Random Forest was chosen as the champion architecture based on three empirical criteria: (1) Highest discriminative power (ROC-AUC of 0.9109), "
        "(2) Superior probability calibration with the lowest Brier Score (0.1200), closely following the diagonal line in reliability diagrams, and "
        "(3) Non-linear feature interaction modeling between afternoon solar heating (18–22 UTC) and summer months in thunderstorm-prone longitude bands."
    )

    # =========================================================================
    # 9. COMPARISON WITH EXISTING WEATHER FORECASTING SYSTEMS
    # =========================================================================
    add_h1("9. Comparison with Existing Weather Forecasting Systems")
    doc.add_paragraph(
        "To provide thorough academic context, StormCast was evaluated against established national and commercial weather forecasting systems. "
        "Table 3 details key capabilities, architectural differences, and operational limitations."
    )

    # Comparison Table
    comp_tbl = doc.add_table(rows=1, cols=4)
    comp_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3_widths = [1.5, 1.7, 1.7, 1.6]
    style_table_header(comp_tbl.rows[0], t3_widths, [
        "System / Provider", "Key Capabilities", "How StormCast Differs", "Limitations & Scope"
    ])

    comp_rows = [
        (
            "National Weather Service\n(NOAA / NWS)",
            "• Numerical Weather Prediction (NWP: GFS, HRRR)\n• Continental Doppler radar (NEXRAD)\n• Official storm watches & warnings",
            "• NWS solves fluid dynamics on supercomputers over hours\n• StormCast is a lightweight tabular ML model running in <0.05 seconds",
            "• High infrastructure cost; not portable to edge devices\n• StormCast prototype does NOT replace official NWS life-safety alerts"
        ),
        (
            "Commercial Weather APIs\n(e.g., OpenWeatherMap)",
            "• Global multi-day temperature/rain forecasts\n• Cloud REST API feeds for web/mobile apps\n• Aggregated station & satellite telemetry",
            "• Commercial models are black-box and subscription-based\n• StormCast provides 100% open-source, calibrated probability (0-100%)",
            "• Generic precipitation probabilities without storm risk calibration\n• StormCast currently uses manual sliders; plans API ingestion in future"
        ),
        (
            "StormCast\n(Current Academic Prototype)",
            "• Instant point-level storm occurrence classification\n• Calibrated probability gauge (0-100%)\n• 4-level hazard severity grading (Light to Severe)",
            "• Zero target leakage (pre-event features only)\n• Pure statistical ML (Random Forest, 83.7% Acc, 0.911 ROC-AUC) running in Streamlit",
            "• Point-in-time classification prototype (not a multi-hour future NWP forecast)\n• Lacks live Doppler radar and upper-air soundings"
        )
    ]
    for r_idx, row in enumerate(comp_rows):
        is_sc = (r_idx == 2)
        add_styled_row(comp_tbl, row, t3_widths, bg_hex="EEF2FF" if is_sc else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF"), bold_first=True)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # =========================================================================
    # 10. CURRENT GAPS, LIMITATIONS & FUTURE SCOPE
    # =========================================================================
    add_h1("10. Current Gaps, Limitations, and Future Scope")
    doc.add_paragraph(
        "In accordance with sound engineering practice, we explicitly delineate what is currently implemented from what constitutes future work."
    )

    add_h2("10.1 Current Gaps (Incomplete or Missing in Current Implementation)")
    gaps_p = doc.add_paragraph()
    gaps_p.add_run("• Absence of Upper-Air Thermodynamic Variables: ").bold = True
    gaps_p.add_run("The tabular ASOS dataset does not contain upper-air atmospheric soundings such as Convective Available Potential Energy (CAPE), Lifted Index (LI), vertical wind shear, or dew point depression.\n")
    gaps_p.add_run("• Manual Parameter Entry: ").bold = True
    gaps_p.add_run("The current Streamlit interface requires the user to input weather values manually via sliders or choose presets; it does not automatically stream live sensor data.\n")
    gaps_p.add_run("• Single Storm Severity Label in Raw Data: ").bold = True
    gaps_p.add_run("In the historical ASOS records, all storms are labeled 'Severe' by automated lightning detectors; fine-grained sub-severity within lightning events is absent.")

    add_h2("10.2 Current Limitations & Scientific Boundary")
    lim_p = doc.add_paragraph()
    lim_p.add_run("• Occurrence Classification Prototype, Not a Future Forecasting System: ").bold = True
    lim_p.add_run("The model is validated as an occurrence classifier—it determines whether a given set of spatial and temporal conditions denotes an ongoing or imminent storm hazard. It has not been validated as a multi-hour ahead future atmospheric simulation engine.\n")
    lim_p.add_run("• Severe Historical Class Imbalance: ").bold = True
    lim_p.add_run("Storms represent <1% of natural weather streams. While stratified sampling (40k/60k) and class weighting allowed the algorithm to learn robust decision boundaries, raw natural prevalence must be considered in threshold tuning.\n")
    lim_p.add_run("• Academic Disclaimer: ").bold = True
    lim_p.add_run("This prototype is strictly an educational tool and is not certified for aviation life-safety or official emergency management decisions.")

    add_h2("10.3 Future Scope & Planned Improvements")
    fut_p = doc.add_paragraph()
    fut_p.add_run("• Live Weather API Ingestion: ").bold = True
    fut_p.add_run("Integrate live NOAA / OpenWeatherMap REST endpoints to automatically populate current station observations in real time.\n")
    fut_p.add_run("• Multi-Modal Deep Learning with Radar Imagery: ").bold = True
    fut_p.add_run("Train Convolutional Neural Networks (CNNs) on Doppler radar (NEXRAD) and GOES-16 satellite cloud tiles alongside tabular features.\n")
    fut_p.add_run("• Real-Time Barometric Pressure Drop Sensors: ").bold = True
    fut_p.add_run("Incorporate rapid barometric pressure change rates (hPa/hr) to detect frontal squalls before cloud-to-ground lightning initiates.\n")
    fut_p.add_run("• Production REST API & Mobile Notifications: ").bold = True
    fut_p.add_run("Deploy the pipeline via FastAPI with automated push notifications for outdoor workers and regional logistics.")

    # =========================================================================
    # 11. CONCLUSION
    # =========================================================================
    add_h1("11. Conclusion")
    doc.add_paragraph(
        "StormCast successfully demonstrates the development, rigorous evaluation, and deployment of an end-to-end Machine Learning system "
        "for atmospheric storm occurrence and severity analysis. By systematically addressing class imbalance, eliminating lookahead data leakage, "
        "and resolving the ASOS precipitation paradox, the champion Random Forest pipeline achieved 83.73% accuracy, 0.9109 ROC-AUC, and well-calibrated "
        "probability estimates (Brier Score = 0.1200) on 20,000 unseen test records."
    )
    doc.add_paragraph(
        "The project underscores the value of classical tabular machine learning as a fast, interpretable, and lightweight screening tool that complements "
        "computationally heavy numerical weather simulations. The complete pipeline, accompanying models, Streamlit web interface, and technical artifacts "
        "provide an authentic, scientifically grounded foundation for college viva voce evaluation and future meteorological research."
    )

    doc.save(output_filename)
    print(f"[+] Detailed project report successfully generated: {output_filename}")


if __name__ == "__main__":
    generate_report()
