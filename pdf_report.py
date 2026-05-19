from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os
import datetime


# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor('#1e2d3d')
GREEN  = colors.HexColor('#16a34a')
GRAY1  = colors.HexColor('#f8f9fa')
GRAY2  = colors.HexColor('#e9ecef')
GRAY3  = colors.HexColor('#adb5bd')
GRAY4  = colors.HexColor('#6c757d')
WHITE  = colors.white
BLACK  = colors.HexColor('#1a1a1a')


def build_pdf_report(analysis_data, output_path='output_videos/match_report.pdf'):
    """
    Generates a professional PDF match analysis report.
    analysis_data is the same dict returned by load_analysis().
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title='Football Match Analysis Report',
        author='Football Analysis Platform'
    )

    styles  = getSampleStyleSheet()
    story   = []
    W, H    = A4

    # ── Custom styles ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Normal'],
        fontSize=28,
        fontName='Helvetica-Bold',
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica',
        textColor=colors.HexColor('#aaaaaa'),
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Normal'],
        fontSize=11,
        fontName='Helvetica-Bold',
        textColor=NAVY,
        spaceBefore=18,
        spaceAfter=8,
        borderPadding=(0, 0, 4, 0),
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica',
        textColor=GRAY4,
        leading=14,
    )
    metric_label_style = ParagraphStyle(
        'MetricLabel',
        parent=styles['Normal'],
        fontSize=7,
        fontName='Helvetica-Bold',
        textColor=GRAY3,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    metric_value_style = ParagraphStyle(
        'MetricValue',
        parent=styles['Normal'],
        fontSize=22,
        fontName='Helvetica-Bold',
        textColor=NAVY,
        alignment=TA_CENTER,
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Helvetica-Bold',
        textColor=WHITE,
        alignment=TA_CENTER,
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Helvetica',
        textColor=BLACK,
        alignment=TA_CENTER,
    )

    # ── Header banner ─────────────────────────────────────────────────────────
    header_data = [[
        Paragraph('Match Analysis Report', title_style),
        Paragraph(
            f"Generated {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
            ParagraphStyle('date', parent=styles['Normal'],
                           fontSize=8, fontName='Helvetica',
                           textColor=colors.HexColor('#888888'),
                           alignment=TA_RIGHT)
        )
    ]]
    header_table = Table(header_data, colWidths=[12*cm, 5*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), NAVY),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [NAVY]),
        ('TOPPADDING',  (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
        ('LEFTPADDING', (0, 0), (0, -1), 20),
        ('RIGHTPADDING', (-1, 0), (-1, -1), 16),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW',   (0, 0), (-1, -1), 3, GREEN),
    ]))
    story.append(header_table)

    subtitle_data = [[
        Paragraph(
            'Computer vision analysis powered by YOLOv8 and ByteTrack',
            subtitle_style
        )
    ]]
    subtitle_table = Table(subtitle_data, colWidths=[17*cm])
    subtitle_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), NAVY),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING',   (0, 0), (-1, -1), 20),
    ]))
    story.append(subtitle_table)
    story.append(Spacer(1, 16))

    # ── Match overview metrics ────────────────────────────────────────────────
    story.append(Paragraph('MATCH OVERVIEW', section_title_style))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=GRAY2, spaceAfter=10))

    def metric_cell(label, value, sub=''):
        return [
            Paragraph(label.upper(), metric_label_style),
            Paragraph(str(value), metric_value_style),
            Paragraph(sub, ParagraphStyle('sub', parent=styles['Normal'],
                                          fontSize=7, textColor=GRAY3,
                                          alignment=TA_CENTER))
        ]

    metrics_data = [[
        metric_cell('Duration',
                    f"{analysis_data.get('duration_sec', 0)}s",
                    'clip length'),
        metric_cell('Frames',
                    f"{analysis_data.get('total_frames', 0):,}",
                    'processed'),
        metric_cell('Team 1 Possession',
                    f"{analysis_data.get('team1_pct', 0)}%",
                    'ball control'),
        metric_cell('Team 2 Possession',
                    f"{analysis_data.get('team2_pct', 0)}%",
                    'ball control'),
    ]]

    metrics_table = Table(
        metrics_data,
        colWidths=[4.25*cm, 4.25*cm, 4.25*cm, 4.25*cm]
    )
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), GRAY1),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [GRAY1]),
        ('BOX',           (0, 0), (-1, -1), 0.5, GRAY2),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, GRAY2),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEABOVE',     (0, 0), (1, 0), 3, NAVY),
        ('LINEABOVE',     (2, 0), (3, 0), 3, GREEN),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 16))

    # ── Possession bar ────────────────────────────────────────────────────────
    story.append(Paragraph('BALL POSSESSION', section_title_style))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=GRAY2, spaceAfter=10))

    t1_pct = analysis_data.get('team1_pct', 50)
    t2_pct = analysis_data.get('team2_pct', 50)
    bar_w  = 17 * cm

    possession_data = [[
        Paragraph(f"Team 1 — {t1_pct}%",
                  ParagraphStyle('t1lbl', parent=styles['Normal'],
                                 fontSize=9, fontName='Helvetica-Bold',
                                 textColor=NAVY)),
        Paragraph(f"Team 2 — {t2_pct}%",
                  ParagraphStyle('t2lbl', parent=styles['Normal'],
                                 fontSize=9, fontName='Helvetica-Bold',
                                 textColor=GREEN, alignment=TA_RIGHT)),
    ]]
    lbl_table = Table(possession_data, colWidths=[8.5*cm, 8.5*cm])
    lbl_table.setStyle(TableStyle([
        ('LEFTPADDING',  (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(lbl_table)

    t1_w = bar_w * (t1_pct / 100)
    t2_w = bar_w * (t2_pct / 100)

    bar_data = [['', '']]
    bar_table = Table(bar_data, colWidths=[t1_w, t2_w], rowHeights=[12])
    bar_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), NAVY),
        ('BACKGROUND', (1, 0), (1, 0), GREEN),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
    ]))
    story.append(bar_table)
    story.append(Spacer(1, 16))

    # ── Pass statistics ───────────────────────────────────────────────────────
    pass_summary = analysis_data.get('pass_summary', {})

    story.append(Paragraph('PASS STATISTICS', section_title_style))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=GRAY2, spaceAfter=10))

    pass_data = [
        [
            Paragraph('TOTAL\nEVENTS', metric_label_style),
            Paragraph('TEAM 1\nPASSES', metric_label_style),
            Paragraph('TEAM 2\nPASSES', metric_label_style),
            Paragraph('INTERCEP-\nTIONS', metric_label_style),
            Paragraph('AVG PASS\nDISTANCE', metric_label_style),
        ],
        [
            Paragraph(str(pass_summary.get('total_passes', 0)),
                      metric_value_style),
            Paragraph(str(pass_summary.get('team1_passes', 0)),
                      metric_value_style),
            Paragraph(str(pass_summary.get('team2_passes', 0)),
                      metric_value_style),
            Paragraph(str(pass_summary.get('interceptions', 0)),
                      metric_value_style),
            Paragraph(
                f"{pass_summary.get('avg_pass_distance_meters', 'N/A')}m",
                metric_value_style),
        ]
    ]
    pass_table = Table(pass_data, colWidths=[3.4*cm]*5)
    pass_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), NAVY),
        ('BACKGROUND',    (0, 1), (-1, 1), GRAY1),
        ('BOX',           (0, 0), (-1, -1), 0.5, GRAY2),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, GRAY2),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(pass_table)
    story.append(Spacer(1, 16))

    # ── Formation analysis ────────────────────────────────────────────────────
    formation_summary = analysis_data.get('formation_summary', {})

    story.append(Paragraph('FORMATION ANALYSIS', section_title_style))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=GRAY2, spaceAfter=10))

    t1_form = formation_summary.get(1, {}).get('dominant_formation', 'N/A')
    t2_form = formation_summary.get(2, {}).get('dominant_formation', 'N/A')

    form_data = [[
        Paragraph('TEAM 1', metric_label_style),
        Paragraph('TEAM 2', metric_label_style),
    ], [
        Paragraph(t1_form,
                  ParagraphStyle('f1', parent=styles['Normal'],
                                 fontSize=26, fontName='Helvetica-Bold',
                                 textColor=NAVY, alignment=TA_CENTER)),
        Paragraph(t2_form,
                  ParagraphStyle('f2', parent=styles['Normal'],
                                 fontSize=26, fontName='Helvetica-Bold',
                                 textColor=GREEN, alignment=TA_CENTER)),
    ]]
    form_table = Table(form_data, colWidths=[8.5*cm, 8.5*cm])
    form_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), GRAY1),
        ('BOX',           (0, 0), (-1, -1), 0.5, GRAY2),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, GRAY2),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEABOVE',     (0, 0), (0, 0), 3, NAVY),
        ('LINEABOVE',     (1, 0), (1, 0), 3, GREEN),
    ]))
    story.append(form_table)

    # Formation breakdown
    story.append(Spacer(1, 10))
    t1_counts = formation_summary.get(1, {}).get('formation_counts', {})
    t2_counts = formation_summary.get(2, {}).get('formation_counts', {})
    t1_total  = formation_summary.get(1, {}).get('total_samples', 1)
    t2_total  = formation_summary.get(2, {}).get('total_samples', 1)

    if t1_counts or t2_counts:
        breakdown_header = [
            Paragraph('FORMATION', table_header_style),
            Paragraph('TEAM 1 %', table_header_style),
            Paragraph('FORMATION', table_header_style),
            Paragraph('TEAM 2 %', table_header_style),
        ]
        breakdown_rows = [breakdown_header]

        t1_list = list(t1_counts.items())[:5]
        t2_list = list(t2_counts.items())[:5]
        max_rows = max(len(t1_list), len(t2_list))

        for i in range(max_rows):
            t1_f = t1_list[i][0] if i < len(t1_list) else ''
            t1_p = round(t1_list[i][1] / t1_total * 100, 1) \
                if i < len(t1_list) else ''
            t2_f = t2_list[i][0] if i < len(t2_list) else ''
            t2_p = round(t2_list[i][1] / t2_total * 100, 1) \
                if i < len(t2_list) else ''

            breakdown_rows.append([
                Paragraph(str(t1_f), table_cell_style),
                Paragraph(f"{t1_p}%" if t1_p != '' else '',
                          table_cell_style),
                Paragraph(str(t2_f), table_cell_style),
                Paragraph(f"{t2_p}%" if t2_p != '' else '',
                          table_cell_style),
            ])

        bd_table = Table(breakdown_rows,
                         colWidths=[4.25*cm]*4)
        bd_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), NAVY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY1]),
            ('BOX',           (0, 0), (-1, -1), 0.5, GRAY2),
            ('INNERGRID',     (0, 0), (-1, -1), 0.5, GRAY2),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(bd_table)

    story.append(Spacer(1, 16))

    # ── Player performance ────────────────────────────────────────────────────
    player_stats = analysis_data.get('player_stats', {})

    if player_stats:
        story.append(PageBreak())
        story.append(Paragraph('PLAYER PERFORMANCE', section_title_style))
        story.append(HRFlowable(width='100%', thickness=1,
                                 color=GRAY2, spaceAfter=10))

        player_header = [
            Paragraph('PLAYER ID', table_header_style),
            Paragraph('TEAM', table_header_style),
            Paragraph('MAX SPEED (km/h)', table_header_style),
            Paragraph('DISTANCE (m)', table_header_style),
        ]

        player_rows = [player_header]
        sorted_players = sorted(
            player_stats.items(),
            key=lambda x: x[1].get('max_speed', 0),
            reverse=True
        )

        for i, (pid, stats) in enumerate(sorted_players):
            if stats.get('max_speed', 0) == 0:
                continue
            team_id = stats.get('team', '-')
            bg = GRAY1 if i % 2 == 0 else WHITE
            player_rows.append([
                Paragraph(f"Player {pid}", table_cell_style),
                Paragraph(f"Team {team_id}", table_cell_style),
                Paragraph(str(stats.get('max_speed', '-')),
                          table_cell_style),
                Paragraph(str(stats.get('total_distance', '-')),
                          table_cell_style),
            ])

        player_table = Table(player_rows,
                             colWidths=[4.25*cm]*4)
        player_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), NAVY),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY1]),
            ('BOX',           (0, 0), (-1, -1), 0.5, GRAY2),
            ('INNERGRID',     (0, 0), (-1, -1), 0.5, GRAY2),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(player_table)
        story.append(Spacer(1, 16))

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    story.append(Paragraph('PLAYER POSITION HEATMAPS', section_title_style))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=GRAY2, spaceAfter=10))

    heatmap_t1 = 'output_videos/team_1_heatmap.png'
    heatmap_t2 = 'output_videos/team_2_heatmap.png'
    lineup_img = 'output_videos/lineup_card.png'

    if os.path.exists(heatmap_t1) and os.path.exists(heatmap_t2):
        heatmap_data = [[
            Image(heatmap_t1, width=8.2*cm, height=5.5*cm),
            Image(heatmap_t2, width=8.2*cm, height=5.5*cm),
        ]]
        hm_table = Table(heatmap_data, colWidths=[8.5*cm, 8.5*cm])
        hm_table.setStyle(TableStyle([
            ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING',   (0, 0), (-1, -1), 4),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 4),
        ]))
        story.append(hm_table)

        caption_data = [[
            Paragraph('Team 1 — Positional Density',
                      ParagraphStyle('cap', parent=styles['Normal'],
                                     fontSize=8, textColor=GRAY4,
                                     alignment=TA_CENTER)),
            Paragraph('Team 2 — Positional Density',
                      ParagraphStyle('cap2', parent=styles['Normal'],
                                     fontSize=8, textColor=GRAY4,
                                     alignment=TA_CENTER)),
        ]]
        cap_table = Table(caption_data, colWidths=[8.5*cm, 8.5*cm])
        cap_table.setStyle(TableStyle([
            ('TOPPADDING',    (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(cap_table)

    story.append(Spacer(1, 12))

    # ── Lineup card ───────────────────────────────────────────────────────────
    if os.path.exists(lineup_img):
        story.append(Paragraph('LINEUP CARD', section_title_style))
        story.append(HRFlowable(width='100%', thickness=1,
                                 color=GRAY2, spaceAfter=10))
        story.append(Image(lineup_img, width=17*cm, height=5.7*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width='100%', thickness=0.5, color=GRAY2))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        'Football Analysis Platform — YOLOv8 · ByteTrack · '
        'Computer Vision · Match Intelligence',
        ParagraphStyle('footer', parent=styles['Normal'],
                       fontSize=7, textColor=GRAY3,
                       alignment=TA_CENTER)
    ))

    doc.build(story)
    print(f"PDF report saved: {output_path}")
    return output_path