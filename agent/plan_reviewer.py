"""
Plan Reviewer - PDF parsing and GPT-4 Vision analysis for civil engineering plans
"""

import os
import io
import base64
import json
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image


class PlanReviewer:
    """
    Extracts and analyzes civil engineering plan sheets using GPT-4 Vision.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the plan reviewer with OpenAI API key."""
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # GPT-4 Vision model
        
    def extract_sheets_from_pdf(self, pdf_path: str, dpi: int = 150) -> List[Dict[str, Any]]:
        """
        Extract each page from a PDF as an image.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image extraction (default 150 for balance of quality/size)
            
        Returns:
            List of dicts with page info and base64-encoded images
        """
        sheets = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to image
            zoom = dpi / 72  # 72 is default PDF DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image and then to base64
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Resize if too large (max 2048px on longest side for API efficiency)
            max_dim = 2048
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Extract text from page as well
            text_content = page.get_text()
            
            sheets.append({
                'page_num': page_num + 1,
                'pdf_name': os.path.basename(pdf_path),
                'image_base64': base64_image,
                'text_content': text_content,
                'width': img.size[0],
                'height': img.size[1]
            })
            
        doc.close()
        return sheets
    
    def extract_sheets_from_multiple_pdfs(self, pdf_paths: List[str], dpi: int = 150) -> List[Dict[str, Any]]:
        """
        Extract sheets from multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            dpi: Resolution for image extraction
            
        Returns:
            Combined list of all sheets from all PDFs
        """
        all_sheets = []
        for pdf_path in pdf_paths:
            sheets = self.extract_sheets_from_pdf(pdf_path, dpi)
            all_sheets.extend(sheets)
        return all_sheets
    
    def classify_sheet(self, sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use GPT-4 Vision to classify what type of plan sheet this is.
        
        Returns sheet type classification (title, site plan, grading, MOT, etc.)
        """
        prompt = """Analyze this civil engineering plan sheet and classify it.

Return a JSON object with:
{
    "sheet_type": "one of: title, general, site_plan, grading, demolition, utility, stormwater, erosion_control, landscape, lighting, structural, mot (maintenance of traffic), cross_section, detail, profile, other",
    "sheet_number": "extracted sheet number if visible",
    "sheet_title": "extracted sheet title if visible",
    "discipline": "civil, structural, landscape, electrical, other",
    "key_elements": ["list of key elements visible on this sheet"],
    "confidence": 0.0 to 1.0
}

Only return the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{sheet['image_base64']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
            else:
                result = {
                    "sheet_type": "unknown",
                    "sheet_number": None,
                    "sheet_title": None,
                    "discipline": "unknown",
                    "key_elements": [],
                    "confidence": 0.0
                }
        
        sheet['classification'] = result
        return sheet
    
    def analyze_sheet_for_checklist_item(
        self, 
        sheet: Dict[str, Any], 
        checklist_item: Dict[str, str],
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a sheet against a specific checklist item using GPT-4 Vision.
        
        Args:
            sheet: Sheet data including base64 image
            checklist_item: Dict with 'id' and 'text' of the checklist item
            custom_instructions: Optional additional instructions for the review
            
        Returns:
            Analysis result with status, comments, and confidence
        """
        custom_context = f"\n\nAdditional review instructions: {custom_instructions}" if custom_instructions else ""
        
        prompt = f"""You are a senior civil engineer performing a QA/QC review of engineering plans.

Analyze this plan sheet and evaluate the following checklist item:

CHECKLIST ITEM: {checklist_item['text']}

Determine if this checklist item:
- YES: The requirement is met/present on this sheet
- NO: The requirement is NOT met/missing and should be on this sheet
- N/A: This item does not apply to this sheet type

Return a JSON object:
{{
    "status": "YES" or "NO" or "N/A",
    "applies_to_sheet": true or false,
    "comments": "Detailed explanation of what you found or didn't find",
    "location_on_sheet": "Where on the sheet you found the relevant info (if applicable)",
    "confidence": 0.0 to 1.0,
    "issues_found": ["list any specific issues or deficiencies"]
}}{custom_context}

Only return the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{sheet['image_base64']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
            else:
                result = {
                    "status": "N/A",
                    "applies_to_sheet": False,
                    "comments": "Unable to parse response",
                    "confidence": 0.0,
                    "issues_found": []
                }
        
        return {
            'checklist_item_id': checklist_item['id'],
            'checklist_item_text': checklist_item['text'],
            'sheet_page': sheet['page_num'],
            'sheet_pdf': sheet['pdf_name'],
            'analysis': result
        }
    
    def extract_quantities(self, sheet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract quantity information from a plan sheet.
        
        Returns quantities found (areas, lengths, counts, etc.)
        """
        prompt = """Analyze this civil engineering plan sheet and extract any quantity information visible.

Look for:
- Areas (square feet, acres)
- Lengths (linear feet)
- Counts (manholes, catch basins, signs, etc.)
- Volumes (cubic yards)
- Any pay items or bid quantities

Return a JSON object:
{
    "quantities_found": [
        {
            "item": "description",
            "quantity": numeric value,
            "unit": "unit of measure",
            "location": "where found on sheet"
        }
    ],
    "has_quantity_table": true/false,
    "notes": "any relevant notes about quantities"
}

Only return the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{sheet['image_base64']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
            else:
                result = {
                    "quantities_found": [],
                    "has_quantity_table": False,
                    "notes": "Unable to parse quantities"
                }
        
        return result
    
    def get_overall_plan_summary(self, sheets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a high-level summary of the entire plan set.
        
        Args:
            sheets: List of classified sheets
            
        Returns:
            Summary of the plan set
        """
        # Use first sheet (usually title) for main summary
        title_sheet = None
        for sheet in sheets:
            if sheet.get('classification', {}).get('sheet_type') == 'title':
                title_sheet = sheet
                break
        
        if not title_sheet and sheets:
            title_sheet = sheets[0]
        
        if not title_sheet:
            return {"error": "No sheets available for summary"}
        
        prompt = """Analyze this civil engineering plan sheet (likely title sheet) and extract project information.

Return a JSON object:
{
    "project_name": "extracted project name",
    "project_number": "extracted project number",
    "client": "client/owner name",
    "location": "project location",
    "engineer_of_record": "engineering firm name",
    "date": "plan date",
    "total_sheets": number if visible,
    "project_description": "brief description of the project",
    "disciplines_included": ["list of engineering disciplines covered"]
}

Only return the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{title_sheet['image_base64']}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            content = response.choices[0].message.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
            else:
                result = {"error": "Unable to extract project summary"}
        
        return result
