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
        
        # Project info
        project_name = info.project_name or 'Project Name Not Identified'
        project_number = info.project_number or 'N/A'
        location = info.location or 'Location Not Identified'
        owner = info.owner or 'Owner Not Identified'
        engineer = info.engineer_of_record or 'Engineer Not Identified'
        total_sheets = self.analysis.total_sheets
        
        checklist_name = checklist.get('name', 'QA/QC Review') if checklist else 'General Review'
        checklist_phase = checklist.get('phase', '') if checklist else ''
        
        # Build checklist items for AI evaluation
        checklist_items_for_ai = []
        if checklist and checklist.get('sections'):
            for section in checklist.get('sections', []):
                section_title = section.get('title', 'General')
                for item in section.get('items', []):
                    checklist_items_for_ai.append({
                        'section': section_title,
                        'id': item.get('id', ''),
                        'text': item.get('text', ''),
                        'required': item.get('required', False)
                    })
        
        # Ask AI to evaluate each item and return JSON
        items_text = "\n".join([f"- {item['id']}: {item['text']}" for item in checklist_items_for_ai])
        
        eval_prompt = f"""Evaluate each checklist item based on the plan analysis.

PLAN ANALYSIS:
{vision_results[:6000] if vision_results else 'No vision analysis available'}

EXTRACTED TEXT:
{sample_text[:3000]}

CHECKLIST ITEMS TO EVALUATE:
{items_text}

Return a JSON array with your evaluation of each item. Format:
[
  {{"id": "30-GEN-001", "status": "PASS", "comment": "Verified - north arrow present on all sheets"}},
  {{"id": "30-GEN-002", "status": "FAIL", "comment": "ADA ramps not identified in scope"}},
  {{"id": "30-GEN-003", "status": "REVIEW", "comment": "Cannot verify utility locations from available sheets"}},
  {{"id": "30-GEN-004", "status": "N/A", "comment": "Not applicable to this project type"}}
]

Status options: PASS, FAIL, REVIEW, N/A
- PASS = Clearly verified/met in the plans
- FAIL = Not met, missing, or has issues
- REVIEW = Cannot determine, needs manual verification
- N/A = Does not apply to this project

Provide specific, actionable comments. Return ONLY the JSON array."""

        try:
            client = OpenAI(api_key=api_key)
            
            # Get AI evaluation
            eval_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a civil engineering QA/QC reviewer. Evaluate checklist items and return JSON only."},
                    {"role": "user", "content": eval_prompt}
                ],
                max_tokens=8000,
                temperature=0.1
            )
            
            eval_text = eval_response.choices[0].message.content.strip()
            # Clean markdown if present
            if eval_text.startswith('```'):
                eval_text = eval_text.split('\n', 1)[1] if '\n' in eval_text else eval_text[3:]
            if eval_text.endswith('```'):
                eval_text = eval_text[:-3]
            eval_text = eval_text.strip()
            if eval_text.startswith('json'):
                eval_text = eval_text[4:].strip()
            
            evaluations = json.loads(eval_text)
            eval_dict = {e['id']: e for e in evaluations}
            
        except Exception as e:
            print(f"AI evaluation error: {e}")
            # Default all to REVIEW if AI fails
            eval_dict = {item['id']: {'id': item['id'], 'status': 'REVIEW', 'comment': 'Requires manual verification'} for item in checklist_items_for_ai}
        
        # Count statuses
        counts = {'PASS': 0, 'FAIL': 0, 'REVIEW': 0, 'N/A': 0}
        for item in checklist_items_for_ai:
            status = eval_dict.get(item['id'], {}).get('status', 'REVIEW')
            if status in counts:
                counts[status] += 1
            else:
                counts['REVIEW'] += 1
        
        # Build HTML report
        html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
body {{ font-family: 'Segoe UI', Calibri, Arial, sans-serif; margin: 0; padding: 40px; color: #333; background: #fff; }}
.report {{ max-width: 900px; margin: 0 auto; }}
.header {{ text-align: center; border-bottom: 4px solid #C8102E; padding-bottom: 25px; margin-bottom: 30px; }}
.header .company {{ font-size: 12px; letter-spacing: 3px; color: #666; margin-bottom: 15px; }}
.header h1 {{ color: #1B365D; font-size: 28px; margin: 0 0 8px 0; }}
.header .phase {{ color: #666; font-size: 16px; margin: 0; }}
.header .date {{ color: #999; font-size: 14px; margin-top: 15px; }}
.info-section {{ margin-bottom: 30px; }}
.info-section h2 {{ font-size: 14px; color: #1B365D; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #1B365D; padding-bottom: 8px; margin-bottom: 15px; }}
.info-table {{ width: 100%; border-collapse: collapse; }}
.info-table td {{ padding: 10px 15px; border: 1px solid #ddd; font-size: 13px; }}
.info-table td:first-child {{ background: #f5f5f5; font-weight: 600; width: 200px; }}
.summary {{ display: flex; justify-content: center; gap: 20px; margin: 30px 0; }}
.summary-box {{ text-align: center; padding: 20px 30px; border-radius: 8px; min-width: 100px; }}
.summary-box.pass {{ background: #d4edda; border: 2px solid #28a745; }}
.summary-box.fail {{ background: #f8d7da; border: 2px solid #dc3545; }}
.summary-box.review {{ background: #fff3cd; border: 2px solid #ffc107; }}
.summary-box.na {{ background: #e9ecef; border: 2px solid #6c757d; }}
.summary-box .count {{ font-size: 36px; font-weight: 700; display: block; }}
.summary-box.pass .count {{ color: #28a745; }}
.summary-box.fail .count {{ color: #dc3545; }}
.summary-box.review .count {{ color: #856404; }}
.summary-box.na .count {{ color: #6c757d; }}
.summary-box .label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
.section {{ margin-bottom: 25px; page-break-inside: avoid; }}
.section-title {{ background: #1B365D; color: white; padding: 12px 18px; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 0; }}
.checklist {{ width: 100%; border-collapse: collapse; }}
.checklist th {{ background: #f0f0f0; padding: 12px; border: 1px solid #ddd; text-align: left; font-size: 12px; font-weight: 600; }}
.checklist td {{ padding: 10px 12px; border: 1px solid #ddd; font-size: 12px; vertical-align: top; }}
.checklist tr:nth-child(even) {{ background: #fafafa; }}
.status-col {{ width: 70px; text-align: center; }}
.id-col {{ width: 100px; font-family: 'Consolas', monospace; color: #666; font-size: 11px; }}
.item-col {{ width: 40%; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; color: white; font-size: 10px; font-weight: 700; text-transform: uppercase; }}
.badge-pass {{ background: #28a745; }}
.badge-fail {{ background: #dc3545; }}
.badge-review {{ background: #ffc107; color: #333; }}
.badge-na {{ background: #6c757d; }}
.findings {{ background: #f8f9fa; padding: 25px; margin: 30px 0; border-left: 5px solid #C8102E; }}
.findings h2 {{ color: #1B365D; font-size: 18px; margin: 0 0 20px 0; }}
.findings h3 {{ font-size: 14px; margin: 20px 0 10px 0; }}
.findings h3.critical {{ color: #dc3545; }}
.findings h3.review {{ color: #856404; }}
.findings h3.general {{ color: #1B365D; }}
.findings ul {{ margin: 0 0 15px 0; padding-left: 25px; }}
.findings li {{ margin: 8px 0; font-size: 13px; }}
.findings p {{ font-size: 13px; color: #555; line-height: 1.6; }}
.signatures {{ display: flex; justify-content: space-between; margin-top: 50px; padding-top: 20px; }}
.sig-block {{ text-align: center; flex: 1; padding: 0 20px; }}
.sig-line {{ border-bottom: 2px solid #333; height: 50px; margin-bottom: 8px; }}
.sig-label {{ font-size: 12px; color: #666; }}
.footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 2px solid #eee; font-size: 11px; color: #999; }}
</style>
</head>
<body>
<div class="report">

<div class="header">
<div class="company">ABONMARCHE</div>
<h1>{checklist_name}</h1>
<p class="phase">{checklist_phase} Design Phase Review</p>
<p class="date">Review Date: {review_date}</p>
</div>

<div class="info-section">
<h2>Project Information</h2>
<table class="info-table">
<tr><td>Project Name</td><td>{project_name}</td></tr>
<tr><td>Project Number</td><td>{project_number}</td></tr>
<tr><td>Location</td><td>{location}</td></tr>
<tr><td>Client / Owner</td><td>{owner}</td></tr>
<tr><td>Engineer of Record</td><td>{engineer}</td></tr>
<tr><td>Total Sheets Reviewed</td><td>{total_sheets}</td></tr>
<tr><td>Reviewed By</td><td>AI-Assisted Review (Redline.ai)</td></tr>
</table>
</div>

<div class="summary">
<div class="summary-box pass"><span class="count">{counts['PASS']}</span><span class="label">Pass</span></div>
<div class="summary-box fail"><span class="count">{counts['FAIL']}</span><span class="label">Fail</span></div>
<div class="summary-box review"><span class="count">{counts['REVIEW']}</span><span class="label">Review</span></div>
<div class="summary-box na"><span class="count">{counts['N/A']}</span><span class="label">N/A</span></div>
</div>
'''
        
        # Group items by section and build tables
        current_section = None
        fail_items = []
        review_items = []
        
        for item in checklist_items_for_ai:
            eval_data = eval_dict.get(item['id'], {'status': 'REVIEW', 'comment': 'Requires manual verification'})
            status = eval_data.get('status', 'REVIEW')
            comment = eval_data.get('comment', '')
            
            # Track fails and reviews for findings
            if status == 'FAIL':
                fail_items.append(f"{item['id']}: {comment}")
            elif status == 'REVIEW':
                review_items.append(f"{item['id']}: {comment}")
            
            # Start new section if needed
            if item['section'] != current_section:
                if current_section is not None:
                    html += '</table></div>\n'
                current_section = item['section']
                html += f'''
<div class="section">
<div class="section-title">{current_section}</div>
<table class="checklist">
<tr><th class="status-col">Status</th><th class="id-col">ID</th><th class="item-col">Checklist Item</th><th>Comments</th></tr>
'''
            
            # Badge class
            badge_class = 'badge-' + status.lower().replace('/', '')
            if status == 'N/A':
                badge_class = 'badge-na'
            
            html += f'''<tr>
<td class="status-col"><span class="badge {badge_class}">{status}</span></td>
<td class="id-col">{item['id']}</td>
<td class="item-col">{item['text']}</td>
<td>{comment}</td>
</tr>
'''
        
        # Close last section
        if current_section is not None:
            html += '</table></div>\n'
        
        # Add findings section
        html += '''
<div class="findings">
<h2>Key Findings & Recommendations</h2>
'''
        
        if fail_items:
            html += '<h3 class="critical">Critical Issues (Action Required)</h3>\n<ul>\n'
            for item in fail_items[:10]:  # Limit to 10
                html += f'<li>{item}</li>\n'
            html += '</ul>\n'
        else:
            html += '<h3 class="critical">Critical Issues</h3>\n<p>No critical issues identified.</p>\n'
        
        if review_items:
            html += '<h3 class="review">Items Requiring Manual Review</h3>\n<ul>\n'
            for item in review_items[:10]:  # Limit to 10
                html += f'<li>{item}</li>\n'
            html += '</ul>\n'
        else:
            html += '<h3 class="review">Items Requiring Manual Review</h3>\n<p>No items flagged for manual review.</p>\n'
        
        html += f'''
<h3 class="general">General Observations</h3>
<p>This {checklist_phase} review evaluated {len(checklist_items_for_ai)} checklist items. 
{counts['PASS']} items passed verification, {counts['FAIL']} items require attention, 
{counts['REVIEW']} items need manual verification, and {counts['N/A']} items were not applicable to this project.</p>
</div>

<div class="signatures">
<div class="sig-block"><div class="sig-line"></div><p class="sig-label">QA/QC Reviewer</p></div>
<div class="sig-block"><div class="sig-line"></div><p class="sig-label">Date</p></div>
<div class="sig-block"><div class="sig-line"></div><p class="sig-label">Project Manager</p></div>
</div>

<div class="footer">
Generated by Redline.ai | Abonmarche QA/QC Review System | {review_date}
</div>

</div>
</body>
</html>'''
        
        return html


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
