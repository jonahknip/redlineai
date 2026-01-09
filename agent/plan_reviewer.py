#!/usr/bin/env python3
"""
Civil Engineering Plan Set Review Agent
A tool that acts as a Civil Engineering Project Manager to review construction plan sets
and generate comprehensive summary reports.
"""

import fitz  # PyMuPDF
import re
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# OpenAI for AI-powered report generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class SheetInfo:
    """Information about a single sheet in the plan set"""
    sheet_number: int
    sheet_title: str = ""
    stations: list = field(default_factory=list)
    scale: str = ""
    date: str = ""


@dataclass
class ProjectInfo:
    """Core project identification information"""
    project_name: str = ""
    project_number: str = ""
    location: str = ""
    owner: str = ""
    engineer_of_record: str = ""
    engineer_license: str = ""
    surveyor: str = ""
    creation_date: str = ""
    revision_date: str = ""


@dataclass
class PlanSetAnalysis:
    """Complete analysis of a civil engineering plan set"""
    project_info: ProjectInfo = field(default_factory=ProjectInfo)
    total_sheets: int = 0
    sheet_index: dict = field(default_factory=dict)
    station_range: tuple = ("", "")
    disciplines_covered: list = field(default_factory=list)
    key_features: list = field(default_factory=list)
    review_flags: list = field(default_factory=list)
    completeness_score: float = 0.0
    sheets: list = field(default_factory=list)


class CivilEngineeringPMAgent:
    """
    Civil Engineering Project Manager Agent for Plan Set Review

    This agent analyzes construction plan sets and generates comprehensive
    summary reports from a PM perspective.
    """

    # Standard sheet types expected in a civil plan set
    EXPECTED_SHEET_TYPES = {
        'cover': ['cover', 'title', 'index'],
        'typical_sections': ['typical', 'section', 'standard'],
        'survey': ['survey', 'existing', 'topographic'],
        'site_plan': ['site plan', 'layout', 'general plan'],
        'grading': ['grading', 'earthwork', 'contour'],
        'drainage': ['drainage', 'storm', 'swppp', 'erosion'],
        'utilities': ['utility', 'water', 'sewer', 'electric', 'gas'],
        'paving': ['paving', 'pavement', 'roadway'],
        'signing_striping': ['sign', 'stripe', 'marking', 'traffic'],
        'mot': ['mot', 'traffic control', 'maintenance of traffic'],
        'landscape': ['landscape', 'planting', 'irrigation'],
        'lighting': ['lighting', 'electrical', 'photometric'],
        'structural': ['structural', 'retaining wall', 'bridge'],
        'details': ['detail', 'construction detail'],
    }

    # Key items to flag for PM review
    REVIEW_TRIGGERS = [
        ('permit', 'Permit requirements identified'),
        ('easement', 'Easement areas noted'),
        ('utility conflict', 'Potential utility conflicts'),
        ('phase', 'Project phasing indicated'),
        ('temporary', 'Temporary construction elements'),
        ('demolition', 'Demolition work required'),
        ('retaining wall', 'Retaining wall construction'),
        ('traffic signal', 'Signal work included'),
        ('railroad', 'Railroad coordination needed'),
        ('wetland', 'Wetland areas present'),
        ('floodplain', 'Floodplain considerations'),
        ('ada', 'ADA compliance elements'),
        ('right of way', 'ROW acquisition may be needed'),
    ]

    def __init__(self, pdf_path: str):
        """Initialize the agent with a plan set PDF"""
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Plan set not found: {pdf_path}")

        self.doc = fitz.open(str(self.pdf_path))
        self.analysis = PlanSetAnalysis()
        self.full_text = ""

    def __del__(self):
        """Clean up PDF document"""
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()

    def extract_all_text(self) -> str:
        """Extract text from all pages"""
        texts = []
        for page in self.doc:
            texts.append(page.get_text())
        self.full_text = "\n".join(texts)
        return self.full_text

    def analyze_project_info(self) -> ProjectInfo:
        """Extract project identification information"""
        info = ProjectInfo()

        # Get metadata
        metadata = self.doc.metadata
        info.creation_date = metadata.get('creationDate', '')
        info.revision_date = metadata.get('modDate', '')

        # Parse cover sheet (usually page 1)
        cover_text = self.doc[0].get_text() if len(self.doc) > 0 else ""

        # Extract project name - look for common patterns
        project_patterns = [
            r'([A-Z][A-Z\s\d\.\-]+(?:IMPROVEMENTS|PROJECT|CONSTRUCTION|DEVELOPMENT))',
            r'(S\.?R\.?\s*\d+[^\n]+)',
            r'PROJECT:\s*([^\n]+)',
        ]
        for pattern in project_patterns:
            match = re.search(pattern, cover_text, re.IGNORECASE)
            if match:
                info.project_name = match.group(1).strip()
                break

        # Extract project number
        project_num_match = re.search(r'(?:PA|PROJECT\s*(?:NO\.?|#)?)\s*(\d+)', cover_text, re.IGNORECASE)
        if project_num_match:
            info.project_number = project_num_match.group(1)

        # Extract location
        location_patterns = [
            r'(SECTION\s+\d+[^\n]+(?:TOWNSHIP|COUNTY)[^\n]+)',
            r'([A-Z][a-z]+\s+(?:County|Township)[^\n]*)',
            r'(City of [A-Za-z\s]+)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, cover_text)
            if match:
                info.location = match.group(1).strip()
                break

        # Extract owner/client
        owner_patterns = [
            r'(?:OWNER|CLIENT|FOR):\s*([^\n]+)',
            r'(City of [A-Za-z\s]+)',
            r'(County of [A-Za-z\s]+)',
        ]
        for pattern in owner_patterns:
            match = re.search(pattern, cover_text, re.IGNORECASE)
            if match:
                info.owner = match.group(1).strip()
                break

        # Extract engineer info
        eng_match = re.search(r'(?:ENGINEER|SURVEYOR)[^\n]*\n([^\n]+)', cover_text)
        if eng_match:
            info.engineer_of_record = eng_match.group(1).strip()

        # Extract PE license number
        pe_match = re.search(r'(?:PE|P\.E\.)\s*(?:NO\.?\s*)?(\d+)', cover_text)
        if pe_match:
            info.engineer_license = f"PE{pe_match.group(1)}"

        self.analysis.project_info = info
        return info

    def analyze_sheet_index(self) -> dict:
        """Parse the sheet index from cover sheet"""
        cover_text = self.doc[0].get_text() if len(self.doc) > 0 else ""

        sheet_index = {}

        # Common sheet index patterns
        # Pattern: Sheet numbers followed by title
        index_patterns = [
            r'(\d+(?:-\d+)?)\s+([A-Za-z][A-Za-z\s&:,\-]+)',
        ]

        for pattern in index_patterns:
            matches = re.findall(pattern, cover_text)
            for sheet_range, title in matches:
                title = title.strip()
                if len(title) > 3 and not title.isupper():  # Filter noise
                    sheet_index[sheet_range] = title

        self.analysis.sheet_index = sheet_index
        return sheet_index

    def analyze_station_range(self) -> tuple:
        """Determine the project station range"""
        if not self.full_text:
            self.extract_all_text()

        stations = re.findall(r'(\d+)\+(\d+)', self.full_text)
        if stations:
            station_values = [(int(s[0]) * 100 + int(s[1])) for s in stations]
            min_sta = min(station_values)
            max_sta = max(station_values)
            self.analysis.station_range = (
                f"{min_sta // 100}+{min_sta % 100:02d}",
                f"{max_sta // 100}+{max_sta % 100:02d}"
            )
        return self.analysis.station_range

    def identify_disciplines(self) -> list:
        """Identify engineering disciplines covered in the plan set"""
        if not self.full_text:
            self.extract_all_text()

        text_lower = self.full_text.lower()
        disciplines = []

        for discipline, keywords in self.EXPECTED_SHEET_TYPES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    disciplines.append(discipline)
                    break

        self.analysis.disciplines_covered = list(set(disciplines))
        return self.analysis.disciplines_covered

    def identify_key_features(self) -> list:
        """Identify key project features and scope elements"""
        if not self.full_text:
            self.extract_all_text()

        text_lower = self.full_text.lower()
        features = []

        feature_keywords = {
            'Roadway widening': ['widening', 'widen'],
            'New pavement': ['new pavement', 'pavement construction'],
            'Pavement rehabilitation': ['rehab', 'overlay', 'resurfacing'],
            'Intersection improvements': ['intersection', 'turn lane'],
            'Sidewalk construction': ['sidewalk', 'pedestrian'],
            'Multi-use path': ['multi-use', 'trail', 'path'],
            'Storm drainage': ['storm sewer', 'drainage', 'inlet'],
            'Sanitary sewer': ['sanitary', 'sewer main'],
            'Water main': ['water main', 'waterline'],
            'Retaining walls': ['retaining wall', 'wall construction'],
            'Bridge work': ['bridge', 'culvert'],
            'Traffic signals': ['traffic signal', 'signal installation'],
            'Street lighting': ['lighting', 'luminaire', 'light pole'],
            'Landscaping': ['landscape', 'planting', 'tree'],
            'Erosion control': ['erosion', 'swppp', 'bmp'],
        }

        for feature, keywords in feature_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    features.append(feature)
                    break

        self.analysis.key_features = list(set(features))
        return self.analysis.key_features

    def flag_review_items(self) -> list:
        """Identify items requiring PM attention"""
        if not self.full_text:
            self.extract_all_text()

        text_lower = self.full_text.lower()
        flags = []

        for trigger, description in self.REVIEW_TRIGGERS:
            if trigger in text_lower:
                flags.append(description)

        self.analysis.review_flags = list(set(flags))
        return self.analysis.review_flags

    def calculate_completeness(self) -> float:
        """Calculate plan set completeness score based on expected elements"""
        expected_elements = [
            ('cover', 'Cover sheet present'),
            ('typical', 'Typical sections included'),
            ('grading', 'Grading plans provided'),
            ('detail', 'Construction details included'),
            ('erosion', 'Erosion control plans'),
        ]

        if not self.full_text:
            self.extract_all_text()

        text_lower = self.full_text.lower()
        found = sum(1 for keyword, _ in expected_elements if keyword in text_lower)

        self.analysis.completeness_score = (found / len(expected_elements)) * 100
        return self.analysis.completeness_score

    def perform_full_analysis(self) -> PlanSetAnalysis:
        """Perform complete plan set analysis"""
        self.analysis.total_sheets = len(self.doc)

        self.extract_all_text()
        self.analyze_project_info()
        self.analyze_sheet_index()
        self.analyze_station_range()
        self.identify_disciplines()
        self.identify_key_features()
        self.flag_review_items()
        self.calculate_completeness()

        return self.analysis

    def generate_summary_report(self) -> str:
        """Generate a concise one-page PM review report"""
        if not self.analysis.total_sheets:
            self.perform_full_analysis()

        info = self.analysis.project_info
        
        # Identify missing/incomplete items
        missing_items = []
        if not info.project_name:
            missing_items.append("Project name not identified on cover sheet")
        if not info.project_number:
            missing_items.append("Project number not found")
        if not info.engineer_of_record:
            missing_items.append("Engineer of Record not identified")
        if not info.engineer_license:
            missing_items.append("PE license number not found")
        if not self.analysis.sheet_index:
            missing_items.append("Sheet index could not be parsed")
        if self.analysis.station_range[0] == "" and self.analysis.station_range[1] == "":
            missing_items.append("Station range not identified")
        
        # Check for missing expected disciplines
        expected_disciplines = {'cover', 'grading', 'drainage', 'details'}
        found_disciplines = set(self.analysis.disciplines_covered)
        missing_disciplines = expected_disciplines - found_disciplines
        for disc in missing_disciplines:
            missing_items.append(f"Missing discipline: {disc.replace('_', ' ').title()}")
        
        # Build concise report
        report = f"""PLANSET REVIEW REPORT
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d')}    Sheets: {self.analysis.total_sheets}    Completeness: {self.analysis.completeness_score:.0f}%

PROJECT SUMMARY
{'-'*60}
Project:  {info.project_name or '[NOT FOUND]'}
Number:   {info.project_number or '[NOT FOUND]'}
Location: {info.location or '[NOT FOUND]'}
Owner:    {info.owner or '[NOT FOUND]'}
Engineer: {info.engineer_of_record or '[NOT FOUND]'} {('(PE ' + info.engineer_license + ')') if info.engineer_license else ''}
Stations: {self.analysis.station_range[0] or 'N/A'} to {self.analysis.station_range[1] or 'N/A'}

DISCIPLINES: {', '.join(sorted([d.replace('_', ' ').title() for d in self.analysis.disciplines_covered])) or 'None identified'}

KEY FEATURES: {', '.join(sorted(self.analysis.key_features)[:8]) or 'None identified'}
"""

        # Issues/Errors Section
        if missing_items or self.analysis.review_flags:
            report += f"""
ISSUES & FLAGS
{'-'*60}
"""
            if missing_items:
                report += "Missing/Incomplete:\n"
                for item in missing_items[:5]:  # Limit to 5
                    report += f"  ! {item}\n"
            
            if self.analysis.review_flags:
                report += "Review Flags:\n"
                for flag in self.analysis.review_flags[:5]:  # Limit to 5
                    report += f"  * {flag}\n"
        
        # Generate To-Do list based on findings
        todos = []
        
        # Always include these
        todos.append("Verify PE seal and signature on all sheets")
        
        # Conditional todos based on findings
        if not info.project_number:
            todos.append("Confirm project number with client")
        if 'Potential utility conflicts' in self.analysis.review_flags:
            todos.append("Schedule utility coordination meeting")
        if 'Permit requirements identified' in self.analysis.review_flags:
            todos.append("Verify all permits obtained")
        if 'Easement areas noted' in self.analysis.review_flags:
            todos.append("Confirm easement acquisitions complete")
        if 'ROW acquisition may be needed' in self.analysis.review_flags:
            todos.append("Verify ROW acquisition status")
        if 'Project phasing indicated' in self.analysis.review_flags:
            todos.append("Review phasing plan with contractor")
        if 'Erosion control' in self.analysis.key_features:
            todos.append("Confirm NPDES/erosion control permit")
        if 'Floodplain considerations' in self.analysis.review_flags:
            todos.append("Verify floodplain permit status")
        if 'Railroad coordination needed' in self.analysis.review_flags:
            todos.append("Initiate railroad coordination")
        if 'Traffic signals' in self.analysis.key_features:
            todos.append("Coordinate with signal contractor")
        if 'drainage' in self.analysis.disciplines_covered:
            todos.append("Review drainage calculations")
        if missing_disciplines:
            todos.append("Request missing plan sheets from designer")
        
        # Add generic items if list is short
        if len(todos) < 5:
            todos.append("Conduct pre-construction meeting")
            todos.append("Identify long-lead procurement items")
        
        report += f"""
PM TO-DO LIST
{'-'*60}
"""
        for i, todo in enumerate(todos[:8], 1):  # Limit to 8 items
            report += f"  [ ] {i}. {todo}\n"
        
        report += f"""
{'='*60}
End of Report
"""
        return report

    def export_json(self) -> dict:
        """Export analysis as JSON-serializable dict"""
        if not self.analysis.total_sheets:
            self.perform_full_analysis()

        return {
            'project_info': asdict(self.analysis.project_info),
            'total_sheets': self.analysis.total_sheets,
            'sheet_index': self.analysis.sheet_index,
            'station_range': list(self.analysis.station_range),
            'disciplines_covered': self.analysis.disciplines_covered,
            'key_features': self.analysis.key_features,
            'review_flags': self.analysis.review_flags,
            'completeness_score': self.analysis.completeness_score,
        }

    def extract_page_images(self, page_numbers: list = None, max_pages: int = 5) -> list:
        """Extract images from PDF pages for vision analysis"""
        import base64
        
        if page_numbers is None:
            # Default to first few pages (cover, index, typical sections)
            page_numbers = list(range(min(max_pages, len(self.doc))))
        
        images = []
        for page_num in page_numbers:
            if page_num >= len(self.doc):
                continue
            
            page = self.doc[page_num]
            # Render page to image at reasonable resolution
            mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom for readability
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to base64
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            images.append({
                'page_num': page_num + 1,
                'base64': img_base64,
                'width': pix.width,
                'height': pix.height
            })
        
        return images

    def analyze_with_vision(self, checklist: dict = None) -> dict:
        """Use GPT-4 Vision to analyze plan sheet images"""
        if not OPENAI_AVAILABLE:
            return {'success': False, 'error': 'OpenAI not available'}
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {'success': False, 'error': 'No API key'}
        
        # Extract images from key pages
        images = self.extract_page_images(max_pages=5)
        
        if not images:
            return {'success': False, 'error': 'Could not extract images'}
        
        # Build checklist prompt if provided - handle both flat and sectioned formats
        checklist_prompt = ""
        if checklist:
            checklist_name = checklist.get('name', 'QA/QC Checklist')
            checklist_phase = checklist.get('phase', '')
            checklist_prompt = f"\n\n=== {checklist_name} ({checklist_phase}) ===\n"
            checklist_prompt += f"Description: {checklist.get('description', '')}\n\n"
            
            # Check if we have sections (new format) or just items (old format)
            if checklist.get('sections'):
                for section in checklist.get('sections', []):
                    section_title = section.get('title', 'General')
                    checklist_prompt += f"\n## {section_title}\n"
                    for item in section.get('items', []):
                        req = "REQUIRED" if item.get('required') else "Optional"
                        checklist_prompt += f"- [{req}] {item.get('id', '')}: {item.get('text', '')}\n"
            else:
                # Flat items format
                checklist_prompt += "Verify these checklist items:\n"
                for item in checklist.get('items', []):
                    req = "REQUIRED" if item.get('required') else "Optional"
                    section = item.get('section', '')
                    section_prefix = f"[{section}] " if section else ""
                    checklist_prompt += f"- [{req}] {section_prefix}{item.get('text', '')}\n"
        
        # Build messages with images
        messages = [
            {
                "role": "system",
                "content": """You are an expert Civil Engineering Project Manager reviewing construction plan sets. 
Analyze the provided plan sheet images and extract:
1. Project information from title blocks (name, number, owner, engineer, date)
2. Sheet identification (sheet number, title, discipline)
3. Key elements visible on each sheet
4. Any issues, missing elements, or items requiring attention
5. Professional seal/signature status

Be thorough and specific. Note anything that appears incomplete or requires follow-up."""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please analyze these {len(images)} plan sheets from a construction planset. Identify all project information, sheet details, and any issues or items requiring attention.{checklist_prompt}"
                    }
                ]
            }
        ]
        
        # Add images to the message
        for img in images:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['base64']}",
                    "detail": "high"
                }
            })
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4000,
                temperature=0.3
            )
            
            vision_analysis = response.choices[0].message.content
            
            return {
                'success': True,
                'analysis': vision_analysis,
                'pages_analyzed': len(images)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_ai_report(self, use_vision: bool = True, checklist: dict = None, custom_instructions: str = "") -> str:
        """Generate a professional QA/QC review report optimized for Word/PDF export"""
        if not OPENAI_AVAILABLE:
            return self.generate_summary_report()
        
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return self.generate_summary_report()
        
        if not self.analysis.total_sheets:
            self.perform_full_analysis()
        
        info = self.analysis.project_info
        review_date = datetime.now().strftime('%B %d, %Y')
        
        # Get vision analysis if enabled
        vision_results = ""
        if use_vision:
            vision_data = self.analyze_with_vision(checklist)
            if vision_data.get('success'):
                vision_results = vision_data['analysis']
        
        # Extract text from plan set for context
        sample_text = ""
        for i, page in enumerate(self.doc[:5]):
            sample_text += f"\n--- Page {i+1} ---\n"
            sample_text += page.get_text()[:2000]
        
        # Build the checklist items for AI to evaluate
        checklist_items_text = ""
        total_items = 0
        sections_list = []
        if checklist and checklist.get('sections'):
            for section in checklist.get('sections', []):
                section_title = section.get('title', 'General')
                sections_list.append(section_title)
                checklist_items_text += f"\n\n=== {section_title} ===\n"
                for item in section.get('items', []):
                    total_items += 1
                    item_id = item.get('id', '')
                    item_text = item.get('text', '')
                    required = "REQUIRED" if item.get('required') else "Optional"
                    checklist_items_text += f"[{item_id}] ({required}) {item_text}\n"
        
        checklist_name = checklist.get('name', 'QA/QC Review') if checklist else 'General Review'
        checklist_phase = checklist.get('phase', '') if checklist else ''
        
        # Pre-fill project info
        project_name = info.project_name or 'Project Name Not Identified'
        project_number = info.project_number or 'N/A'
        location = info.location or 'Location Not Identified'
        owner = info.owner or 'Owner Not Identified'
        engineer = info.engineer_of_record or 'Engineer Not Identified'
        total_sheets = self.analysis.total_sheets
        
        prompt = f"""Generate a QA/QC checklist review report in HTML format.

PROJECT DATA:
- Project Name: {project_name}
- Project Number: {project_number}
- Location: {location}
- Owner/Client: {owner}
- Engineer: {engineer}
- Total Sheets: {total_sheets}
- Review Type: {checklist_name}
- Review Phase: {checklist_phase}
- Review Date: {review_date}

PLAN ANALYSIS:
{vision_results[:4000] if vision_results else 'Vision analysis not available'}

PLAN TEXT EXTRACTED:
{sample_text[:2000]}

CHECKLIST ({total_items} items):
{checklist_items_text}

Generate HTML report. For each checklist item, assign:
- PASS (green #27ae60) = Verified in plans
- FAIL (red #e74c3c) = Not met/missing  
- REVIEW (orange #f39c12) = Needs manual check
- N/A (gray #95a5a6) = Not applicable

Output this exact HTML structure with all {total_items} items evaluated:

<div class="report">
<style>
.report {{ font-family: Calibri, Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }}
.header {{ text-align: center; border-bottom: 3px solid #c8102e; padding-bottom: 20px; margin-bottom: 25px; }}
.header h1 {{ color: #1b365d; margin: 10px 0; font-size: 24px; }}
.header p {{ margin: 5px 0; color: #666; }}
.info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
.info-table td {{ padding: 8px 12px; border: 1px solid #ddd; }}
.info-table td:first-child {{ background: #f5f5f5; font-weight: bold; width: 35%; }}
.summary {{ display: table; width: 100%; margin-bottom: 25px; }}
.summary-box {{ display: table-cell; text-align: center; padding: 15px; border: 1px solid #ddd; }}
.summary-box.pass {{ background: #d4edda; }}
.summary-box.fail {{ background: #f8d7da; }}
.summary-box.review {{ background: #fff3cd; }}
.summary-box.na {{ background: #e9ecef; }}
.summary-box strong {{ display: block; font-size: 28px; }}
.section {{ margin-bottom: 20px; }}
.section-title {{ background: #1b365d; color: white; padding: 10px 15px; font-weight: bold; margin: 0; }}
.checklist {{ width: 100%; border-collapse: collapse; }}
.checklist th {{ background: #f0f0f0; padding: 10px; border: 1px solid #ddd; text-align: left; font-size: 12px; }}
.checklist td {{ padding: 8px 10px; border: 1px solid #ddd; font-size: 12px; vertical-align: top; }}
.checklist .status {{ text-align: center; width: 60px; }}
.checklist .id {{ width: 80px; font-family: monospace; color: #666; }}
.checklist .item {{ width: 40%; }}
.checklist .notes {{ color: #555; }}
.badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 10px; font-weight: bold; }}
.badge.pass {{ background: #27ae60; }}
.badge.fail {{ background: #e74c3c; }}
.badge.review {{ background: #f39c12; }}
.badge.na {{ background: #95a5a6; }}
.findings {{ background: #f8f9fa; padding: 20px; margin: 25px 0; border-left: 4px solid #c8102e; }}
.findings h2 {{ color: #1b365d; margin-top: 0; font-size: 16px; }}
.findings h3 {{ font-size: 14px; margin: 15px 0 8px; }}
.findings ul {{ margin: 0 0 10px 0; padding-left: 20px; }}
.findings li {{ margin: 5px 0; }}
.signatures {{ display: table; width: 100%; margin-top: 40px; }}
.sig-block {{ display: table-cell; text-align: center; padding: 0 15px; }}
.sig-line {{ border-bottom: 1px solid #333; height: 40px; margin-bottom: 5px; }}
.sig-label {{ font-size: 11px; color: #666; }}
.footer {{ text-align: center; margin-top: 30px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 10px; color: #999; }}
</style>

<div class="header">
<p style="letter-spacing: 2px; font-size: 11px;">ABONMARCHE</p>
<h1>{checklist_name}</h1>
<p>{checklist_phase} Design Phase Review</p>
<p style="margin-top: 15px;">Review Date: {review_date}</p>
</div>

<table class="info-table">
<tr><td>Project Name</td><td>{project_name}</td></tr>
<tr><td>Project Number</td><td>{project_number}</td></tr>
<tr><td>Location</td><td>{location}</td></tr>
<tr><td>Client/Owner</td><td>{owner}</td></tr>
<tr><td>Engineer of Record</td><td>{engineer}</td></tr>
<tr><td>Total Sheets Reviewed</td><td>{total_sheets}</td></tr>
<tr><td>Reviewed By</td><td>AI-Assisted Review (Redline.ai)</td></tr>
</table>

<div class="summary">
<div class="summary-box pass"><strong>[COUNT]</strong>PASS</div>
<div class="summary-box fail"><strong>[COUNT]</strong>FAIL</div>
<div class="summary-box review"><strong>[COUNT]</strong>REVIEW</div>
<div class="summary-box na"><strong>[COUNT]</strong>N/A</div>
</div>

<!-- FOR EACH SECTION: -->
<div class="section">
<div class="section-title">[SECTION NAME]</div>
<table class="checklist">
<tr><th class="status">Status</th><th class="id">ID</th><th class="item">Checklist Item</th><th class="notes">Comments</th></tr>
<!-- FOR EACH ITEM: -->
<tr>
<td class="status"><span class="badge [pass/fail/review/na]">[STATUS]</span></td>
<td class="id">[ID]</td>
<td class="item">[Item description]</td>
<td class="notes">[Specific observation]</td>
</tr>
</table>
</div>

<div class="findings">
<h2>Key Findings & Recommendations</h2>
<h3 style="color: #e74c3c;">Critical Issues (Action Required)</h3>
<ul>[List FAIL items]</ul>
<h3 style="color: #f39c12;">Items Requiring Manual Review</h3>
<ul>[List REVIEW items]</ul>
<h3 style="color: #1b365d;">General Observations</h3>
<p>[Overall assessment]</p>
</div>

<div class="signatures">
<div class="sig-block"><div class="sig-line"></div><p class="sig-label">QA/QC Reviewer</p></div>
<div class="sig-block"><div class="sig-line"></div><p class="sig-label">Date</p></div>
<div class="sig-block"><div class="sig-line"></div><p class="sig-label">Project Manager</p></div>
</div>

<div class="footer">Generated by Redline.ai | Abonmarche QA/QC System | {review_date}</div>
</div>

REQUIREMENTS:
1. Include ALL {total_items} checklist items - do not skip any
2. Replace [COUNT] with actual counts
3. Replace all [...] placeholders with real content
4. Provide specific comments for each item based on plan analysis
5. If uncertain about an item, mark as REVIEW not PASS
6. Return ONLY the HTML, no explanations or markdown"""

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You generate professional QA/QC review reports in HTML format. Output clean HTML only, no markdown. Include embedded CSS in a style tag. Evaluate every checklist item."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16000,
                temperature=0.1
            )
            
            html_report = response.choices[0].message.content
            
            # Clean up any markdown code block markers
            html_report = html_report.strip()
            if html_report.startswith('```html'):
                html_report = html_report[7:]
            if html_report.startswith('```'):
                html_report = html_report[3:]
            if html_report.endswith('```'):
                html_report = html_report[:-3]
            
            return html_report.strip()
            
        except Exception as e:
            # Fall back to basic report on error
            print(f"OpenAI error: {e}")
            return self.generate_summary_report()


def main():
    """Main entry point for the plan reviewer"""
    if len(sys.argv) < 2:
        print("Usage: python plan_reviewer.py <path_to_plan_set.pdf> [--json]")
        print("\nCivil Engineering PM Agent - Plan Set Review Tool")
        print("Analyzes construction plan sets and generates PM summary reports.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_json = '--json' in sys.argv

    try:
        agent = CivilEngineeringPMAgent(pdf_path)

        if output_json:
            result = agent.export_json()
            print(json.dumps(result, indent=2))
        else:
            report = agent.generate_summary_report()
            print(report)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing plan set: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
