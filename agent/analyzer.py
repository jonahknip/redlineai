"""
Full Drawing Analyzer - The core analysis engine for redline.ai
Processes every sheet in a planset using GPT-4 Vision
"""
import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

import fitz  # PyMuPDF

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of analyzing a single sheet"""
    page_number: int
    sheet_number: str = ""
    sheet_title: str = ""
    sheet_type: str = ""
    discipline: str = ""
    scale: str = ""
    title_block: Dict[str, str] = field(default_factory=dict)
    elements: Dict[str, List[Dict]] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    success: bool = True
    error: str = ""


class FullDrawingAnalyzer:
    """
    Analyzes every sheet in a planset PDF using GPT-4 Vision.
    
    This is the core engine that:
    1. Renders each page to an image
    2. Sends to GPT-4 Vision for detailed analysis
    3. Extracts structured data about elements, dimensions, annotations
    4. Identifies issues and findings
    """
    
    # Analysis prompt for GPT-4 Vision
    SHEET_ANALYSIS_PROMPT = """You are an expert civil engineering drawing analyst. Analyze this construction plan sheet in detail.

Return a JSON object with exactly this structure:
{
    "sheet_number": "The sheet number (e.g., C-101, G-001, or page number if not found)",
    "sheet_title": "The sheet title from the title block",
    "sheet_type": "One of: cover, plan, profile, detail, section, schedule, notes, general",
    "discipline": "One of: general, civil, structural, landscape, electrical, plumbing, mechanical",
    "scale": "The drawing scale (e.g., 1\"=20', VARIES, NTS)",
    
    "title_block": {
        "project_name": "Full project name",
        "project_number": "Project number if shown",
        "date": "Date shown",
        "drawn_by": "Initials of drafter",
        "checked_by": "Initials of checker", 
        "approved_by": "Initials of approver",
        "revision": "Current revision",
        "engineer": "Engineer name or firm"
    },
    
    "elements": {
        "dimensions": [{"text": "45.00'", "type": "linear", "location": "center of sheet"}],
        "annotations": [{"text": "SEE DETAIL A", "type": "callout"}],
        "utilities": [{"type": "water", "size": "8\" DI", "description": "Water main"}],
        "roads": [{"name": "Main Street", "type": "existing", "width": "24'"}],
        "drainage": [{"type": "inlet", "size": "4'x4'", "description": "Type C inlet"}],
        "structures": [{"type": "retaining wall", "description": "Proposed CMU wall"}],
        "references": [{"text": "SEE SHEET C-102", "target": "C-102"}]
    },
    
    "observations": [
        "Key observation about the drawing content",
        "Another important note"
    ],
    
    "findings": [
        {
            "severity": "critical|major|minor|info",
            "category": "dimension|annotation|reference|scale|title_block|missing|unclear",
            "title": "Brief title",
            "description": "Detailed description of the issue",
            "recommendation": "Suggested action to resolve",
            "location": "Where on the sheet (e.g., upper right, center)"
        }
    ],
    
    "summary": "2-3 sentence summary of what this sheet shows"
}

Important:
- Be thorough - extract ALL visible dimensions, annotations, and elements
- Flag any issues: missing dimensions, unclear annotations, inconsistent scales, reference errors
- For findings, use severity: critical (must fix), major (should fix), minor (consider), info (observation)
- If something is unclear or you can't read it, note it as a finding
- Return ONLY valid JSON, no markdown or explanation"""

    def __init__(self, pdf_path: str, api_key: str = None):
        """
        Initialize the analyzer with a PDF file.
        
        Args:
            pdf_path: Path to the planset PDF
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required for analysis")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        
        self.client = OpenAI(api_key=self.api_key)
        self.doc = fitz.open(str(self.pdf_path))
        self.page_count = len(self.doc)
        
        # Results storage
        self.results: List[AnalysisResult] = []
        self.project_info: Dict[str, Any] = {}
        
    def __del__(self):
        """Clean up PDF document"""
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()
    
    def render_page_to_image(self, page_num: int, zoom: float = 1.5) -> str:
        """
        Render a PDF page to a base64-encoded PNG image.
        
        Args:
            page_num: Page number (0-indexed)
            zoom: Zoom factor for rendering quality
            
        Returns:
            Base64-encoded PNG image string
        """
        if page_num < 0 or page_num >= self.page_count:
            raise ValueError(f"Invalid page number: {page_num}")
        
        page = self.doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def analyze_sheet(self, page_num: int) -> AnalysisResult:
        """
        Analyze a single sheet using GPT-4 Vision.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            AnalysisResult with extracted data and findings
        """
        result = AnalysisResult(page_number=page_num + 1)  # 1-indexed for display
        
        try:
            # Render page to image
            img_base64 = self.render_page_to_image(page_num)
            
            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.SHEET_ANALYSIS_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            # Parse response
            raw_response = response.choices[0].message.content
            result.raw_response = raw_response
            
            # Clean up JSON response
            json_text = raw_response.strip()
            if json_text.startswith('```'):
                json_text = json_text.split('\n', 1)[1] if '\n' in json_text else json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            if json_text.startswith('json'):
                json_text = json_text[4:].strip()
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Populate result
            result.sheet_number = data.get('sheet_number', f"Page {page_num + 1}")
            result.sheet_title = data.get('sheet_title', '')
            result.sheet_type = data.get('sheet_type', 'unknown')
            result.discipline = data.get('discipline', 'general')
            result.scale = data.get('scale', '')
            result.title_block = data.get('title_block', {})
            result.elements = data.get('elements', {})
            result.references = [ref.get('target', '') for ref in data.get('elements', {}).get('references', [])]
            result.findings = data.get('findings', [])
            result.summary = data.get('summary', '')
            
            # Add page number to each finding
            for finding in result.findings:
                finding['page_number'] = page_num + 1
                finding['sheet_number'] = result.sheet_number
            
            result.success = True
            
        except json.JSONDecodeError as e:
            result.success = False
            result.error = f"Failed to parse AI response: {str(e)}"
            logger.error(f"JSON parse error on page {page_num + 1}: {e}")
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Analysis error on page {page_num + 1}: {e}")
        
        return result
    
    def analyze_all_sheets(
        self, 
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[AnalysisResult]:
        """
        Analyze all sheets in the planset.
        
        Args:
            progress_callback: Optional callback(current, total, status_message)
            
        Returns:
            List of AnalysisResult for each sheet
        """
        self.results = []
        
        for page_num in range(self.page_count):
            if progress_callback:
                progress_callback(
                    page_num + 1, 
                    self.page_count, 
                    f"Analyzing page {page_num + 1} of {self.page_count}..."
                )
            
            result = self.analyze_sheet(page_num)
            self.results.append(result)
            
            # Extract project info from first successful sheet (usually cover)
            if result.success and not self.project_info:
                self.project_info = result.title_block.copy()
        
        if progress_callback:
            progress_callback(self.page_count, self.page_count, "Analysis complete!")
        
        return self.results
    
    def get_all_findings(self) -> List[Dict[str, Any]]:
        """Get all findings from all analyzed sheets"""
        all_findings = []
        for result in self.results:
            all_findings.extend(result.findings)
        
        # Sort by severity (critical first)
        severity_order = {'critical': 0, 'major': 1, 'minor': 2, 'info': 3}
        all_findings.sort(key=lambda f: severity_order.get(f.get('severity', 'info'), 4))
        
        return all_findings
    
    def get_all_elements(self) -> Dict[str, List[Dict]]:
        """Get all extracted elements grouped by category"""
        all_elements: Dict[str, List[Dict]] = {}
        
        for result in self.results:
            for category, items in result.elements.items():
                if category not in all_elements:
                    all_elements[category] = []
                for item in items:
                    item_copy = item.copy()
                    item_copy['source_sheet'] = result.sheet_number
                    item_copy['page_number'] = result.page_number
                    all_elements[category].append(item_copy)
        
        return all_elements
    
    def get_sheet_index(self) -> List[Dict[str, Any]]:
        """Get a summary index of all sheets"""
        return [
            {
                'page_number': r.page_number,
                'sheet_number': r.sheet_number,
                'sheet_title': r.sheet_title,
                'sheet_type': r.sheet_type,
                'discipline': r.discipline,
                'finding_count': len(r.findings),
                'element_count': sum(len(v) for v in r.elements.values()),
                'success': r.success,
                'error': r.error
            }
            for r in self.results
        ]
    
    def get_findings_summary(self) -> Dict[str, int]:
        """Get count of findings by severity"""
        findings = self.get_all_findings()
        summary = {'critical': 0, 'major': 0, 'minor': 0, 'info': 0}
        for f in findings:
            sev = f.get('severity', 'info')
            if sev in summary:
                summary[sev] += 1
        return summary
    
    def to_json(self) -> Dict[str, Any]:
        """Export full analysis as JSON-serializable dict"""
        return {
            'project_info': self.project_info,
            'page_count': self.page_count,
            'sheets': [
                {
                    'page_number': r.page_number,
                    'sheet_number': r.sheet_number,
                    'sheet_title': r.sheet_title,
                    'sheet_type': r.sheet_type,
                    'discipline': r.discipline,
                    'scale': r.scale,
                    'title_block': r.title_block,
                    'elements': r.elements,
                    'references': r.references,
                    'findings': r.findings,
                    'summary': r.summary,
                    'success': r.success,
                    'error': r.error
                }
                for r in self.results
            ],
            'all_findings': self.get_all_findings(),
            'all_elements': self.get_all_elements(),
            'findings_summary': self.get_findings_summary(),
            'sheet_index': self.get_sheet_index()
        }
