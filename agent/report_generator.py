"""
Report Generator - Generate completed checklist reports in various formats
"""

import io
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


class ReportGenerator:
    """
    Generates completed QA/QC checklist reports matching Abonmarche format.
    Outputs HTML, Word (.docx), and PDF formats.
    """
    
    # Status styling
    STATUS_COLORS = {
        'YES': '#28a745',  # Green
        'NO': '#dc3545',   # Red
        'N/A': '#6c757d'   # Gray
    }
    
    STATUS_SYMBOLS = {
        'YES': '&#10004;',  # Checkmark
        'NO': '&#10008;',   # X
        'N/A': '&#8211;'    # En dash
    }
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    def generate_html_report(self, review_results: Dict[str, Any]) -> str:
        """
        Generate an HTML report that looks like the Abonmarche checklist forms.
        
        Args:
            review_results: Results from ChecklistEngine.run_review()
            
        Returns:
            HTML string of the completed checklist
        """
        project = review_results.get('project_summary', {})
        summary = review_results.get('summary', {})
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{review_results.get('checklist_name', 'QA/QC Review')} - {project.get('project_name', 'Project')}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report-container {{
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #1a365d;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .logo {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .logo-icon {{
            width: 50px;
            height: 50px;
            background: #1a365d;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .logo-icon span {{
            color: #e53e3e;
            font-size: 28px;
            font-weight: bold;
        }}
        .company-name {{
            font-size: 24px;
            font-weight: bold;
            color: #1a365d;
        }}
        .report-title {{
            text-align: right;
        }}
        .report-title h1 {{
            margin: 0;
            font-size: 20px;
            color: #1a365d;
        }}
        .report-title .phase {{
            font-size: 16px;
            color: #e53e3e;
            margin-top: 5px;
        }}
        .project-info {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .project-info .field {{
            display: flex;
        }}
        .project-info .label {{
            font-weight: bold;
            min-width: 120px;
            color: #1a365d;
        }}
        .project-info .value {{
            color: #333;
        }}
        .summary-box {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-item {{
            text-align: center;
            padding: 15px;
            border-radius: 5px;
        }}
        .summary-item.total {{
            background: #e3f2fd;
            border: 2px solid #1a365d;
        }}
        .summary-item.yes {{
            background: #d4edda;
            border: 2px solid #28a745;
        }}
        .summary-item.no {{
            background: #f8d7da;
            border: 2px solid #dc3545;
        }}
        .summary-item.na {{
            background: #e9ecef;
            border: 2px solid #6c757d;
        }}
        .summary-item .number {{
            font-size: 32px;
            font-weight: bold;
        }}
        .summary-item .label {{
            font-size: 12px;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section-title {{
            background: #1a365d;
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            margin-bottom: 0;
        }}
        .checklist-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .checklist-table th {{
            background: #e9ecef;
            padding: 10px;
            text-align: center;
            border: 1px solid #dee2e6;
            font-size: 12px;
        }}
        .checklist-table td {{
            padding: 10px;
            border: 1px solid #dee2e6;
            vertical-align: top;
        }}
        .checklist-table .item-num {{
            width: 50px;
            text-align: center;
            font-weight: bold;
        }}
        .checklist-table .item-text {{
            width: 40%;
        }}
        .checklist-table .status {{
            width: 80px;
            text-align: center;
            font-weight: bold;
        }}
        .checklist-table .comments {{
            width: 40%;
            font-size: 13px;
            color: #555;
        }}
        .status-yes {{
            color: #28a745;
        }}
        .status-no {{
            color: #dc3545;
        }}
        .status-na {{
            color: #6c757d;
        }}
        .next-phase-section {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 20px;
            border-radius: 5px;
            margin-top: 30px;
        }}
        .next-phase-section h2 {{
            color: #856404;
            margin-top: 0;
        }}
        .next-phase-list {{
            list-style: none;
            padding: 0;
        }}
        .next-phase-list li {{
            padding: 10px;
            background: white;
            margin-bottom: 10px;
            border-left: 4px solid #dc3545;
        }}
        .next-phase-list .item-id {{
            font-weight: bold;
            color: #dc3545;
        }}
        .quantities-section {{
            margin-top: 30px;
        }}
        .quantities-section h2 {{
            color: #1a365d;
            border-bottom: 2px solid #1a365d;
            padding-bottom: 10px;
        }}
        .quantity-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .quantity-table th {{
            background: #1a365d;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .quantity-table td {{
            padding: 10px;
            border: 1px solid #dee2e6;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 12px;
        }}
        @media print {{
            body {{
                background: white;
            }}
            .report-container {{
                box-shadow: none;
                padding: 0;
            }}
            .section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <div class="logo">
                <div class="logo-icon">
                    <span>A</span>
                </div>
                <div class="company-name">ABONMARCHE</div>
            </div>
            <div class="report-title">
                <h1>QA/QC REVIEW CHECKLIST</h1>
                <div class="phase">{review_results.get('checklist_name', review_results.get('phase', ''))} Review</div>
            </div>
        </div>
        
        <div class="project-info">
            <div class="field">
                <span class="label">Project:</span>
                <span class="value">{project.get('project_name', 'N/A')}</span>
            </div>
            <div class="field">
                <span class="label">Project No:</span>
                <span class="value">{project.get('project_number', 'N/A')}</span>
            </div>
            <div class="field">
                <span class="label">Client:</span>
                <span class="value">{project.get('client', 'N/A')}</span>
            </div>
            <div class="field">
                <span class="label">Review Date:</span>
                <span class="value">{datetime.now().strftime('%B %d, %Y')}</span>
            </div>
            <div class="field">
                <span class="label">Location:</span>
                <span class="value">{project.get('location', 'N/A')}</span>
            </div>
            <div class="field">
                <span class="label">Sheets Reviewed:</span>
                <span class="value">{review_results.get('total_sheets', 0)}</span>
            </div>
        </div>
        
        <div class="summary-box">
            <div class="summary-item total">
                <div class="number">{summary.get('total_items', 0)}</div>
                <div class="label">Total Items</div>
            </div>
            <div class="summary-item yes">
                <div class="number">{summary.get('yes_count', 0)}</div>
                <div class="label">Passed</div>
            </div>
            <div class="summary-item no">
                <div class="number">{summary.get('no_count', 0)}</div>
                <div class="label">Failed</div>
            </div>
            <div class="summary-item na">
                <div class="number">{summary.get('na_count', 0)}</div>
                <div class="label">N/A</div>
            </div>
        </div>
'''
        
        # Add each section
        for section in review_results.get('sections', []):
            html += f'''
        <div class="section">
            <div class="section-title">{section['title']}</div>
            <table class="checklist-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Check Item</th>
                        <th>Status</th>
                        <th>AI Comments</th>
                    </tr>
                </thead>
                <tbody>
'''
            for i, item in enumerate(section.get('items', []), 1):
                status = item.get('status', 'N/A')
                status_class = f"status-{status.lower().replace('/', '')}"
                comments = item.get('comments', '')
                if len(comments) > 200:
                    comments = comments[:200] + '...'
                
                html += f'''
                    <tr>
                        <td class="item-num">{i}</td>
                        <td class="item-text">{item.get('text', '')}</td>
                        <td class="status {status_class}">{status}</td>
                        <td class="comments">{comments}</td>
                    </tr>
'''
            
            html += '''
                </tbody>
            </table>
        </div>
'''
        
        # Add items required for next phase
        items_for_next = summary.get('items_for_next_phase', [])
        next_phase = review_results.get('next_phase')
        
        if items_for_next and next_phase:
            html += f'''
        <div class="next-phase-section">
            <h2>Items Required to Advance to {next_phase} Phase</h2>
            <p>The following {len(items_for_next)} item(s) must be addressed before advancing:</p>
            <ul class="next-phase-list">
'''
            for item in items_for_next:
                html += f'''
                <li>
                    <span class="item-id">{item.get('id', '')}:</span> {item.get('text', '')}
                    <br><em>{item.get('comments', '')[:150]}...</em>
                </li>
'''
            html += '''
            </ul>
        </div>
'''
        
        # Add quantities section if present
        quantities = review_results.get('quantities', [])
        if quantities:
            html += '''
        <div class="quantities-section">
            <h2>Extracted Quantities</h2>
'''
            for qty_group in quantities:
                html += f'''
            <h4>{qty_group.get('sheet', '')}</h4>
            <table class="quantity-table">
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Quantity</th>
                        <th>Unit</th>
                        <th>Location</th>
                    </tr>
                </thead>
                <tbody>
'''
                for qty in qty_group.get('quantities', []):
                    html += f'''
                    <tr>
                        <td>{qty.get('item', '')}</td>
                        <td>{qty.get('quantity', '')}</td>
                        <td>{qty.get('unit', '')}</td>
                        <td>{qty.get('location', '')}</td>
                    </tr>
'''
                html += '''
                </tbody>
            </table>
'''
            html += '''
        </div>
'''
        
        # Footer
        html += f'''
        <div class="footer">
            <p>Generated by RedlineAI - AI-Powered Plan Review</p>
            <p>Review Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Files Reviewed: {', '.join(review_results.get('pdf_files', []))}</p>
        </div>
    </div>
</body>
</html>
'''
        
        return html
    
    def generate_word_report(self, review_results: Dict[str, Any]) -> io.BytesIO:
        """
        Generate a Word document report matching Abonmarche checklist format.
        
        Args:
            review_results: Results from ChecklistEngine.run_review()
            
        Returns:
            BytesIO buffer containing the Word document
        """
        doc = Document()
        project = review_results.get('project_summary', {})
        summary = review_results.get('summary', {})
        
        # Set up styles
        style = doc.styles['Normal']
        style.font.name = 'Arial'
        style.font.size = Pt(10)
        
        # Header
        header = doc.add_heading('ENGINEERING DEPARTMENT', level=1)
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        subtitle = doc.add_heading(f"{review_results.get('checklist_name', 'QA/QC')} REVIEW CHECKLIST", level=2)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Project info table
        info_table = doc.add_table(rows=4, cols=4)
        info_table.style = 'Table Grid'
        
        info_data = [
            ('Client:', project.get('client', ''), 'Project Name:', project.get('project_name', '')),
            ('Project No:', project.get('project_number', ''), 'Date:', datetime.now().strftime('%m/%d/%Y')),
            ('Location:', project.get('location', ''), 'Reviewed By:', 'RedlineAI'),
            ('Sheets:', str(review_results.get('total_sheets', 0)), 'Phase:', review_results.get('phase', ''))
        ]
        
        for i, row_data in enumerate(info_data):
            row = info_table.rows[i]
            for j, text in enumerate(row_data):
                cell = row.cells[j]
                cell.text = str(text)
                if j % 2 == 0:  # Labels
                    cell.paragraphs[0].runs[0].bold = True
        
        doc.add_paragraph()
        
        # Summary
        summary_para = doc.add_paragraph()
        summary_para.add_run('SUMMARY: ').bold = True
        summary_para.add_run(
            f"Total Items: {summary.get('total_items', 0)} | "
            f"Passed: {summary.get('yes_count', 0)} | "
            f"Failed: {summary.get('no_count', 0)} | "
            f"N/A: {summary.get('na_count', 0)}"
        )
        
        doc.add_paragraph()
        
        # Checklist sections
        for section in review_results.get('sections', []):
            # Section header
            section_header = doc.add_paragraph()
            section_header.add_run(section['title']).bold = True
            
            # Items table
            table = doc.add_table(rows=1, cols=4)
            table.style = 'Table Grid'
            
            # Header row
            header_row = table.rows[0]
            headers = ['#', 'CHECK ITEM', 'STATUS', 'COMMENTS']
            for i, header_text in enumerate(headers):
                cell = header_row.cells[i]
                cell.text = header_text
                cell.paragraphs[0].runs[0].bold = True
            
            # Item rows
            for i, item in enumerate(section.get('items', []), 1):
                row = table.add_row()
                row.cells[0].text = str(i)
                row.cells[1].text = item.get('text', '')
                
                status = item.get('status', 'N/A')
                status_cell = row.cells[2]
                status_cell.text = status
                
                # Color the status
                if status == 'YES':
                    status_cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(40, 167, 69)
                elif status == 'NO':
                    status_cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(220, 53, 69)
                
                comments = item.get('comments', '')
                if len(comments) > 100:
                    comments = comments[:100] + '...'
                row.cells[3].text = comments
            
            doc.add_paragraph()
        
        # Items for next phase
        items_for_next = summary.get('items_for_next_phase', [])
        next_phase = review_results.get('next_phase')
        
        if items_for_next and next_phase:
            doc.add_heading(f'Items Required to Advance to {next_phase}', level=2)
            
            for item in items_for_next:
                para = doc.add_paragraph(style='List Bullet')
                para.add_run(f"{item.get('id', '')}: ").bold = True
                para.add_run(item.get('text', ''))
        
        # Save to buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        return buffer
    
    def generate_excel_quantities(self, review_results: Dict[str, Any]) -> io.BytesIO:
        """
        Generate an Excel spreadsheet with extracted quantities.
        
        Args:
            review_results: Results from ChecklistEngine.run_review()
            
        Returns:
            BytesIO buffer containing the Excel file
        """
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Quantities"
        
        # Header styling
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1A365D", end_color="1A365D", fill_type="solid")
        
        # Headers
        headers = ['Sheet', 'Item', 'Quantity', 'Unit', 'Location']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
        
        # Data
        row_num = 2
        for qty_group in review_results.get('quantities', []):
            sheet_name = qty_group.get('sheet', '')
            for qty in qty_group.get('quantities', []):
                ws.cell(row=row_num, column=1, value=sheet_name)
                ws.cell(row=row_num, column=2, value=qty.get('item', ''))
                ws.cell(row=row_num, column=3, value=qty.get('quantity', ''))
                ws.cell(row=row_num, column=4, value=qty.get('unit', ''))
                ws.cell(row=row_num, column=5, value=qty.get('location', ''))
                row_num += 1
        
        # Auto-size columns
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column].width = adjusted_width
        
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        
        return buffer
