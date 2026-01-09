"""
PDF Redliner - Adds visual annotations/markups to PDF drawings
Uses PyMuPDF to draw boxes, circles, text callouts for findings
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# Color definitions (RGB tuples, 0-1 range)
COLORS = {
    'critical': (1.0, 0.0, 0.0),      # Red
    'major': (1.0, 0.5, 0.0),         # Orange
    'minor': (1.0, 0.8, 0.0),         # Yellow
    'info': (0.0, 0.5, 1.0),          # Blue
    'highlight': (1.0, 1.0, 0.0),     # Yellow highlight
    'cloud': (1.0, 0.0, 0.0),         # Red revision cloud
}


@dataclass
class RedlineAnnotation:
    """Represents a single annotation to add to the PDF"""
    page_number: int  # 1-indexed
    annotation_type: str  # box, circle, cloud, text, highlight, strikeout
    severity: str = "info"  # critical, major, minor, info
    
    # Position (normalized 0-1, or absolute in points)
    x: float = 0.0
    y: float = 0.0
    width: float = 0.1
    height: float = 0.05
    normalized: bool = True  # If True, coords are 0-1; if False, absolute points
    
    # Content
    text: str = ""
    comment: str = ""
    
    def get_color(self) -> Tuple[float, float, float]:
        """Get RGB color based on severity"""
        return COLORS.get(self.severity, COLORS['info'])


class PDFRedliner:
    """
    Adds redline annotations to PDF drawings.
    
    Supports:
    - Rectangular boxes with colored borders
    - Circles/ellipses
    - Revision clouds
    - Text callouts with leader lines
    - Highlight/strikeout annotations
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize with a PDF file.
        
        Args:
            pdf_path: Path to the original PDF
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Open document for editing
        self.doc = fitz.open(str(self.pdf_path))
        self.page_count = len(self.doc)
        
        # Track annotations added
        self.annotations_added = 0
    
    def __del__(self):
        """Clean up without saving"""
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()
    
    def _get_absolute_rect(
        self, 
        page: fitz.Page,
        x: float, 
        y: float, 
        width: float, 
        height: float,
        normalized: bool = True
    ) -> fitz.Rect:
        """
        Convert coordinates to absolute rectangle.
        
        Args:
            page: PyMuPDF page object
            x, y: Top-left corner
            width, height: Size
            normalized: If True, coords are 0-1 ratios
        """
        if normalized:
            page_rect = page.rect
            abs_x = page_rect.width * x
            abs_y = page_rect.height * y
            abs_w = page_rect.width * width
            abs_h = page_rect.height * height
        else:
            abs_x, abs_y, abs_w, abs_h = x, y, width, height
        
        return fitz.Rect(abs_x, abs_y, abs_x + abs_w, abs_y + abs_h)
    
    def add_box(
        self,
        page_num: int,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Tuple[float, float, float] = (1, 0, 0),
        line_width: float = 2.0,
        fill_opacity: float = 0.1,
        normalized: bool = True,
        comment: str = ""
    ) -> None:
        """
        Add a rectangular box annotation.
        
        Args:
            page_num: Page number (1-indexed)
            x, y: Top-left corner
            width, height: Size
            color: RGB color tuple (0-1)
            line_width: Border width in points
            fill_opacity: Fill transparency (0-1)
            normalized: If True, coords are 0-1 ratios
            comment: Optional comment text
        """
        if page_num < 1 or page_num > self.page_count:
            logger.warning(f"Invalid page number: {page_num}")
            return
        
        page = self.doc[page_num - 1]
        rect = self._get_absolute_rect(page, x, y, width, height, normalized)
        
        # Add rectangle annotation
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=color, fill=color)
        annot.set_opacity(fill_opacity)
        annot.set_border(width=line_width)
        
        if comment:
            annot.set_info(content=comment)
        
        annot.update()
        self.annotations_added += 1
    
    def add_circle(
        self,
        page_num: int,
        x: float,
        y: float,
        radius: float,
        color: Tuple[float, float, float] = (1, 0, 0),
        line_width: float = 2.0,
        normalized: bool = True,
        comment: str = ""
    ) -> None:
        """
        Add a circle annotation.
        
        Args:
            page_num: Page number (1-indexed)
            x, y: Center point
            radius: Circle radius
            color: RGB color tuple
            line_width: Border width
            normalized: If True, coords are 0-1 ratios
            comment: Optional comment text
        """
        if page_num < 1 or page_num > self.page_count:
            return
        
        page = self.doc[page_num - 1]
        
        if normalized:
            page_rect = page.rect
            cx = page_rect.width * x
            cy = page_rect.height * y
            r = min(page_rect.width, page_rect.height) * radius
        else:
            cx, cy, r = x, y, radius
        
        rect = fitz.Rect(cx - r, cy - r, cx + r, cy + r)
        
        # Circle annotation
        annot = page.add_circle_annot(rect)
        annot.set_colors(stroke=color)
        annot.set_border(width=line_width)
        
        if comment:
            annot.set_info(content=comment)
        
        annot.update()
        self.annotations_added += 1
    
    def add_text_callout(
        self,
        page_num: int,
        x: float,
        y: float,
        text: str,
        color: Tuple[float, float, float] = (1, 0, 0),
        font_size: float = 10,
        normalized: bool = True
    ) -> None:
        """
        Add a text annotation (sticky note style).
        
        Args:
            page_num: Page number (1-indexed)
            x, y: Position
            text: Text content
            color: RGB color
            font_size: Font size
            normalized: If True, coords are 0-1 ratios
        """
        if page_num < 1 or page_num > self.page_count:
            return
        
        page = self.doc[page_num - 1]
        
        if normalized:
            page_rect = page.rect
            px = page_rect.width * x
            py = page_rect.height * y
        else:
            px, py = x, y
        
        point = fitz.Point(px, py)
        
        # Add text annotation (comment icon)
        annot = page.add_text_annot(point, text)
        annot.set_colors(stroke=color)
        annot.update()
        self.annotations_added += 1
    
    def add_freetext(
        self,
        page_num: int,
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: Tuple[float, float, float] = (1, 0, 0),
        bg_color: Tuple[float, float, float] = (1, 1, 0.8),
        font_size: float = 10,
        normalized: bool = True
    ) -> None:
        """
        Add a freetext annotation (text box on the page).
        
        Args:
            page_num: Page number (1-indexed)
            x, y, width, height: Position and size
            text: Text content
            color: Text color
            bg_color: Background color
            font_size: Font size
            normalized: If True, coords are 0-1 ratios
        """
        if page_num < 1 or page_num > self.page_count:
            return
        
        page = self.doc[page_num - 1]
        rect = self._get_absolute_rect(page, x, y, width, height, normalized)
        
        # Add freetext annotation
        annot = page.add_freetext_annot(
            rect,
            text,
            fontsize=font_size,
            fontname="helv",
            text_color=color,
            fill_color=bg_color,
            border_color=color
        )
        annot.update()
        self.annotations_added += 1
    
    def add_highlight(
        self,
        page_num: int,
        rect: Tuple[float, float, float, float],
        color: Tuple[float, float, float] = (1, 1, 0),
        normalized: bool = True
    ) -> None:
        """
        Add a highlight annotation.
        
        Args:
            page_num: Page number (1-indexed)
            rect: (x, y, width, height) tuple
            color: Highlight color
            normalized: If True, coords are 0-1 ratios
        """
        if page_num < 1 or page_num > self.page_count:
            return
        
        page = self.doc[page_num - 1]
        annot_rect = self._get_absolute_rect(page, *rect, normalized)
        
        # Highlight uses a quad (four points)
        quad = annot_rect.quad
        annot = page.add_highlight_annot(quad)
        annot.set_colors(stroke=color)
        annot.update()
        self.annotations_added += 1
    
    def add_revision_cloud(
        self,
        page_num: int,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Tuple[float, float, float] = (1, 0, 0),
        line_width: float = 1.5,
        normalized: bool = True,
        comment: str = ""
    ) -> None:
        """
        Add a revision cloud annotation (simulated with wavy line).
        
        Note: PyMuPDF doesn't have native cloud support, so we simulate
        with a dashed rectangle and note it as a cloud.
        
        Args:
            page_num: Page number (1-indexed)
            x, y, width, height: Position and size
            color: Cloud color
            line_width: Line width
            normalized: If True, coords are 0-1 ratios
            comment: Optional comment
        """
        if page_num < 1 or page_num > self.page_count:
            return
        
        page = self.doc[page_num - 1]
        rect = self._get_absolute_rect(page, x, y, width, height, normalized)
        
        # Use a polygon annotation with dashed line to simulate cloud
        # Create a simple rectangular path
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=color, width=line_width, dashes="[3 2]")
        shape.commit()
        
        # Add a text annotation for the comment
        if comment:
            point = fitz.Point(rect.x0, rect.y0)
            annot = page.add_text_annot(point, f"REVISION: {comment}")
            annot.set_colors(stroke=color)
            annot.update()
        
        self.annotations_added += 1
    
    def add_findings(self, findings: List[Dict[str, Any]]) -> int:
        """
        Add annotations for a list of findings.
        
        Args:
            findings: List of finding dicts with page_number, severity, 
                     location (optional), title, description
        
        Returns:
            Number of annotations added
        """
        added = 0
        
        for finding in findings:
            page_num = finding.get('page_number', 1)
            severity = finding.get('severity', 'info')
            title = finding.get('title', '')
            description = finding.get('description', '')
            location = finding.get('location', {})
            
            color = COLORS.get(severity, COLORS['info'])
            comment = f"{title}\n\n{description}"
            
            # If location is provided, add a box annotation
            if location and isinstance(location, dict):
                x = location.get('x', 0.5)
                y = location.get('y', 0.5)
                w = location.get('width', 0.1)
                h = location.get('height', 0.05)
                
                self.add_box(
                    page_num=page_num,
                    x=x, y=y, width=w, height=h,
                    color=color,
                    line_width=2.0,
                    fill_opacity=0.15,
                    normalized=True,
                    comment=comment
                )
                added += 1
            else:
                # No location - add text annotation at default position
                # Position based on finding index within the page
                default_x = 0.85
                default_y = 0.1 + (added % 10) * 0.08
                
                self.add_text_callout(
                    page_num=page_num,
                    x=default_x,
                    y=default_y,
                    text=comment,
                    color=color,
                    normalized=True
                )
                added += 1
        
        return added
    
    def add_summary_page(self, findings_summary: Dict[str, Any]) -> None:
        """
        Add a summary page at the beginning with analysis results.
        
        Args:
            findings_summary: Dict with summary statistics and key findings
        """
        # Create a new page at the beginning
        page_rect = self.doc[0].rect if self.page_count > 0 else fitz.Rect(0, 0, 612, 792)
        new_page = self.doc.new_page(0, width=page_rect.width, height=page_rect.height)
        
        # Add title
        title_rect = fitz.Rect(50, 50, page_rect.width - 50, 100)
        new_page.insert_textbox(
            title_rect,
            "REDLINE.AI ANALYSIS SUMMARY",
            fontsize=24,
            fontname="helv",
            color=(0.1, 0.2, 0.4),
            align=fitz.TEXT_ALIGN_CENTER
        )
        
        # Add summary statistics
        stats_y = 130
        stats = [
            f"Total Pages Analyzed: {findings_summary.get('page_count', 0)}",
            f"Critical Issues: {findings_summary.get('critical', 0)}",
            f"Major Issues: {findings_summary.get('major', 0)}",
            f"Minor Issues: {findings_summary.get('minor', 0)}",
            f"Informational Notes: {findings_summary.get('info', 0)}",
        ]
        
        for stat in stats:
            stat_rect = fitz.Rect(100, stats_y, page_rect.width - 100, stats_y + 25)
            new_page.insert_textbox(stat_rect, stat, fontsize=14, fontname="helv")
            stats_y += 30
        
        # Note about annotations
        note_rect = fitz.Rect(50, stats_y + 30, page_rect.width - 50, stats_y + 80)
        new_page.insert_textbox(
            note_rect,
            "Findings have been annotated throughout the document.\n"
            "Look for colored boxes and comments on each page.",
            fontsize=11,
            fontname="helv",
            color=(0.4, 0.4, 0.4),
            align=fitz.TEXT_ALIGN_CENTER
        )
        
        self.page_count = len(self.doc)
    
    def save(self, output_path: str = None) -> str:
        """
        Save the annotated PDF.
        
        Args:
            output_path: Output file path. If None, appends '_redlined' to original name.
        
        Returns:
            Path to saved file
        """
        if output_path is None:
            stem = self.pdf_path.stem
            output_path = str(self.pdf_path.parent / f"{stem}_redlined.pdf")
        
        self.doc.save(output_path)
        logger.info(f"Saved redlined PDF to {output_path} ({self.annotations_added} annotations)")
        
        return output_path
    
    def save_to_bytes(self) -> bytes:
        """
        Save the annotated PDF to bytes (for streaming/download).
        
        Returns:
            PDF as bytes
        """
        return self.doc.tobytes()
