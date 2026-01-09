"""
Training Module for Redline.ai Review Agent
Handles:
1. Parsing completed review PDFs to extract training data
2. Storing examples in vector database for RAG
3. Generating few-shot examples for prompts
4. Exporting fine-tuning datasets
"""
import os
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Training data directory
TRAINING_DIR = Path(__file__).parent.parent / 'training'
EXAMPLES_DIR = TRAINING_DIR / 'examples'
EMBEDDINGS_DIR = TRAINING_DIR / 'embeddings'

# Ensure directories exist
TRAINING_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Try to import ChromaDB for vector storage
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available - RAG features disabled")

# Try to import OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TrainingExample:
    """A single training example from a completed review"""
    
    def __init__(
        self,
        checklist_item_id: str,
        checklist_item_text: str,
        status: str,  # PASS, FAIL, REVIEW, N/A
        comment: str,
        project_type: str = "",
        project_name: str = "",
        sheet_context: str = "",  # What was visible on the sheets
        source_file: str = ""
    ):
        self.checklist_item_id = checklist_item_id
        self.checklist_item_text = checklist_item_text
        self.status = status.upper()
        self.comment = comment
        self.project_type = project_type
        self.project_name = project_name
        self.sheet_context = sheet_context
        self.source_file = source_file
        self.created_at = datetime.now().isoformat()
        
        # Generate unique ID
        content = f"{checklist_item_id}:{status}:{comment}:{project_name}"
        self.id = hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'checklist_item_id': self.checklist_item_id,
            'checklist_item_text': self.checklist_item_text,
            'status': self.status,
            'comment': self.comment,
            'project_type': self.project_type,
            'project_name': self.project_name,
            'sheet_context': self.sheet_context,
            'source_file': self.source_file,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        example = cls(
            checklist_item_id=data['checklist_item_id'],
            checklist_item_text=data['checklist_item_text'],
            status=data['status'],
            comment=data['comment'],
            project_type=data.get('project_type', ''),
            project_name=data.get('project_name', ''),
            sheet_context=data.get('sheet_context', ''),
            source_file=data.get('source_file', '')
        )
        example.id = data.get('id', example.id)
        example.created_at = data.get('created_at', example.created_at)
        return example
    
    def to_embedding_text(self) -> str:
        """Generate text for embedding"""
        return f"""Checklist Item: {self.checklist_item_text}
Project Type: {self.project_type}
Status: {self.status}
Comment: {self.comment}
Context: {self.sheet_context[:500] if self.sheet_context else 'N/A'}"""


class ReviewPDFParser:
    """Parse filled-out review PDF forms to extract training data"""
    
    # Common patterns for status indicators in PDFs
    STATUS_PATTERNS = {
        'PASS': [r'\[x\]\s*pass', r'✓\s*pass', r'pass\s*[✓✔☑]', r'\byes\b', r'\bpassed\b', r'\bcomplete\b'],
        'FAIL': [r'\[x\]\s*fail', r'✗\s*fail', r'fail\s*[✗✘☒]', r'\bno\b', r'\bfailed\b', r'\bincomplete\b'],
        'REVIEW': [r'\[x\]\s*review', r'review\s*needed', r'\bn/r\b', r'\btbd\b', r'\bpending\b'],
        'N/A': [r'\[x\]\s*n/?a', r'not\s*applicable', r'\bn/a\b', r'\bna\b']
    }
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.doc = fitz.open(str(self.pdf_path))
        self.examples: List[TrainingExample] = []
    
    def __del__(self):
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()
    
    def extract_form_fields(self) -> Dict[str, Any]:
        """Extract form field values from PDF"""
        fields = {}
        
        for page in self.doc:
            # Get form fields (widgets)
            for widget in page.widgets():
                if widget.field_name:
                    field_name = widget.field_name
                    field_value = widget.field_value or ""
                    field_type = widget.field_type_string
                    
                    fields[field_name] = {
                        'value': field_value,
                        'type': field_type,
                        'page': page.number + 1
                    }
        
        return fields
    
    def extract_text_with_positions(self) -> List[Dict[str, Any]]:
        """Extract text blocks with their positions"""
        text_blocks = []
        
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        text = " ".join([span["text"] for span in line["spans"]])
                        if text.strip():
                            text_blocks.append({
                                'text': text.strip(),
                                'page': page_num + 1,
                                'bbox': block["bbox"],
                                'y': line["bbox"][1]  # Y position for sorting
                            })
        
        return text_blocks
    
    def parse_checklist_responses(self, checklist: Dict[str, Any] = None) -> List[TrainingExample]:
        """
        Parse the PDF to extract checklist item responses.
        
        Args:
            checklist: Optional checklist definition to match against
        
        Returns:
            List of TrainingExample objects
        """
        self.examples = []
        
        # First try form fields
        form_fields = self.extract_form_fields()
        
        if form_fields:
            self._parse_from_form_fields(form_fields, checklist)
        else:
            # Fall back to text extraction
            self._parse_from_text(checklist)
        
        return self.examples
    
    def _parse_from_form_fields(self, fields: Dict[str, Any], checklist: Dict[str, Any] = None):
        """Parse responses from PDF form fields"""
        
        # Group fields by checklist item (often named like "30-GEN-001_status", "30-GEN-001_comment")
        item_fields = {}
        
        for field_name, field_data in fields.items():
            # Try to extract item ID from field name
            match = re.match(r'^(\d+-[A-Z]+-\d+)', field_name)
            if match:
                item_id = match.group(1)
                if item_id not in item_fields:
                    item_fields[item_id] = {}
                
                if 'status' in field_name.lower() or 'pass' in field_name.lower() or 'fail' in field_name.lower():
                    item_fields[item_id]['status'] = field_data['value']
                elif 'comment' in field_name.lower() or 'note' in field_name.lower():
                    item_fields[item_id]['comment'] = field_data['value']
        
        # Create examples from parsed fields
        for item_id, data in item_fields.items():
            status = self._normalize_status(data.get('status', ''))
            comment = data.get('comment', '')
            
            if status:
                # Get item text from checklist if available
                item_text = self._get_item_text(item_id, checklist)
                
                example = TrainingExample(
                    checklist_item_id=item_id,
                    checklist_item_text=item_text,
                    status=status,
                    comment=comment,
                    source_file=self.pdf_path.name
                )
                self.examples.append(example)
    
    def _parse_from_text(self, checklist: Dict[str, Any] = None):
        """Parse responses from PDF text (for non-fillable PDFs)"""
        
        text_blocks = self.extract_text_with_positions()
        full_text = "\n".join([b['text'] for b in text_blocks])
        
        # Look for patterns like "30-GEN-001: PASS - Comment here"
        pattern = r'(\d+-[A-Z]+-\d+)[:\s]+([A-Za-z/]+)[\s\-:]*(.+?)(?=\d+-[A-Z]+-\d+|$)'
        matches = re.findall(pattern, full_text, re.DOTALL)
        
        for match in matches:
            item_id = match[0]
            status_text = match[1].strip()
            comment = match[2].strip()
            
            status = self._normalize_status(status_text)
            if status:
                item_text = self._get_item_text(item_id, checklist)
                
                example = TrainingExample(
                    checklist_item_id=item_id,
                    checklist_item_text=item_text,
                    status=status,
                    comment=comment[:500],  # Limit comment length
                    source_file=self.pdf_path.name
                )
                self.examples.append(example)
    
    def _normalize_status(self, status_text: str) -> str:
        """Normalize status text to PASS/FAIL/REVIEW/N/A"""
        if not status_text:
            return ""
        
        status_text = status_text.lower().strip()
        
        for status, patterns in self.STATUS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, status_text, re.IGNORECASE):
                    return status
        
        # Direct matches
        if status_text in ['pass', 'p', 'y', 'yes', '1', 'true', 'ok']:
            return 'PASS'
        elif status_text in ['fail', 'f', 'n', 'no', '0', 'false']:
            return 'FAIL'
        elif status_text in ['review', 'r', 'tbd', 'pending']:
            return 'REVIEW'
        elif status_text in ['n/a', 'na', 'not applicable', '-']:
            return 'N/A'
        
        return ""
    
    def _get_item_text(self, item_id: str, checklist: Dict[str, Any] = None) -> str:
        """Get checklist item text from checklist definition"""
        if not checklist:
            return ""
        
        for section in checklist.get('sections', []):
            for item in section.get('items', []):
                if item.get('id') == item_id:
                    return item.get('text', '')
        
        return ""


class TrainingDataStore:
    """Store and retrieve training examples"""
    
    def __init__(self):
        self.examples_file = EXAMPLES_DIR / 'training_examples.json'
        self.examples: List[TrainingExample] = []
        self._load_examples()
        
        # Initialize ChromaDB if available
        self.chroma_client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(EMBEDDINGS_DIR),
                anonymized_telemetry=False
            ))
            self.collection = self.chroma_client.get_or_create_collection(
                name="review_examples",
                metadata={"description": "Training examples for review agent"}
            )
            logger.info(f"ChromaDB initialized with {self.collection.count()} examples")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _load_examples(self):
        """Load examples from JSON file"""
        if self.examples_file.exists():
            try:
                with open(self.examples_file, 'r') as f:
                    data = json.load(f)
                self.examples = [TrainingExample.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.examples)} training examples")
            except Exception as e:
                logger.error(f"Failed to load training examples: {e}")
                self.examples = []
    
    def _save_examples(self):
        """Save examples to JSON file"""
        try:
            with open(self.examples_file, 'w') as f:
                json.dump([e.to_dict() for e in self.examples], f, indent=2)
            logger.info(f"Saved {len(self.examples)} training examples")
        except Exception as e:
            logger.error(f"Failed to save training examples: {e}")
    
    def add_examples(self, examples: List[TrainingExample]) -> int:
        """Add new training examples"""
        added = 0
        existing_ids = {e.id for e in self.examples}
        
        for example in examples:
            if example.id not in existing_ids:
                self.examples.append(example)
                existing_ids.add(example.id)
                added += 1
                
                # Add to vector store
                if self.collection:
                    self._add_to_vector_store(example)
        
        if added > 0:
            self._save_examples()
        
        return added
    
    def _add_to_vector_store(self, example: TrainingExample):
        """Add example to ChromaDB vector store"""
        if not self.collection:
            return
        
        try:
            self.collection.add(
                ids=[example.id],
                documents=[example.to_embedding_text()],
                metadatas=[{
                    'checklist_item_id': example.checklist_item_id,
                    'status': example.status,
                    'project_type': example.project_type
                }]
            )
        except Exception as e:
            logger.error(f"Failed to add to vector store: {e}")
    
    def find_similar_examples(
        self,
        checklist_item_text: str,
        project_type: str = "",
        n_results: int = 3
    ) -> List[TrainingExample]:
        """Find similar examples using vector search"""
        
        if not self.collection or self.collection.count() == 0:
            # Fall back to simple text matching
            return self._find_similar_by_text(checklist_item_text, n_results)
        
        try:
            query_text = f"Checklist Item: {checklist_item_text}\nProject Type: {project_type}"
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Get full examples by ID
            similar = []
            if results['ids'] and results['ids'][0]:
                for example_id in results['ids'][0]:
                    for example in self.examples:
                        if example.id == example_id:
                            similar.append(example)
                            break
            
            return similar
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._find_similar_by_text(checklist_item_text, n_results)
    
    def _find_similar_by_text(self, checklist_item_text: str, n_results: int) -> List[TrainingExample]:
        """Simple text-based similarity search"""
        
        # Find examples with matching keywords
        keywords = set(checklist_item_text.lower().split())
        
        scored = []
        for example in self.examples:
            example_keywords = set(example.checklist_item_text.lower().split())
            overlap = len(keywords & example_keywords)
            if overlap > 0:
                # Prioritize examples with comments (more valuable for training)
                # Also boost examples that aren't just "REVIEW"
                bonus = 0
                if example.comment and len(example.comment) > 10:
                    bonus += 3  # Significant boost for good comments
                if example.status in ('PASS', 'FAIL'):
                    bonus += 1  # Slight boost for decisive statuses
                scored.append((overlap + bonus, example))
        
        # Sort by score and return top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:n_results]]
    
    def get_examples_for_item(self, checklist_item_id: str) -> List[TrainingExample]:
        """Get all examples for a specific checklist item"""
        return [e for e in self.examples if e.checklist_item_id == checklist_item_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about training data"""
        stats = {
            'total_examples': len(self.examples),
            'by_status': {'PASS': 0, 'FAIL': 0, 'REVIEW': 0, 'N/A': 0},
            'by_checklist_item': {},
            'unique_projects': set(),
            'source_files': set()
        }
        
        for example in self.examples:
            stats['by_status'][example.status] = stats['by_status'].get(example.status, 0) + 1
            stats['by_checklist_item'][example.checklist_item_id] = \
                stats['by_checklist_item'].get(example.checklist_item_id, 0) + 1
            if example.project_name:
                stats['unique_projects'].add(example.project_name)
            if example.source_file:
                stats['source_files'].add(example.source_file)
        
        stats['unique_projects'] = len(stats['unique_projects'])
        stats['source_files'] = len(stats['source_files'])
        stats['unique_checklist_items'] = len(stats['by_checklist_item'])
        
        return stats
    
    def export_for_finetuning(self, output_path: str = None) -> str:
        """Export training data in OpenAI fine-tuning format (JSONL)"""
        
        if output_path is None:
            output_path = str(TRAINING_DIR / 'finetuning_data.jsonl')
        
        with open(output_path, 'w') as f:
            for example in self.examples:
                # Create fine-tuning example
                system_msg = "You are a civil engineering QA/QC reviewer. Evaluate checklist items and provide status (PASS/FAIL/REVIEW/N/A) with specific comments."
                
                user_msg = f"""Evaluate this checklist item:
Item ID: {example.checklist_item_id}
Item: {example.checklist_item_text}
Project Type: {example.project_type}
Sheet Context: {example.sheet_context[:500] if example.sheet_context else 'Review the plans'}"""

                assistant_msg = json.dumps({
                    'id': example.checklist_item_id,
                    'status': example.status,
                    'comment': example.comment
                })
                
                fine_tune_example = {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg}
                    ]
                }
                
                f.write(json.dumps(fine_tune_example) + '\n')
        
        logger.info(f"Exported {len(self.examples)} examples to {output_path}")
        return output_path


def generate_few_shot_prompt(
    checklist_item_id: str,
    checklist_item_text: str,
    training_store: TrainingDataStore,
    n_examples: int = 3
) -> str:
    """
    Generate a few-shot prompt section with similar examples.
    
    Args:
        checklist_item_id: ID of the item being evaluated
        checklist_item_text: Text of the item being evaluated
        training_store: TrainingDataStore instance
        n_examples: Number of examples to include
    
    Returns:
        Formatted string with example evaluations
    """
    
    # First try to get examples for this exact item
    examples = training_store.get_examples_for_item(checklist_item_id)
    
    # If not enough, find similar items
    if len(examples) < n_examples:
        similar = training_store.find_similar_examples(checklist_item_text, n_results=n_examples)
        for ex in similar:
            if ex not in examples:
                examples.append(ex)
            if len(examples) >= n_examples:
                break
    
    if not examples:
        return ""
    
    prompt_parts = ["\n--- EXAMPLE EVALUATIONS FROM PAST REVIEWS ---\n"]
    
    for i, example in enumerate(examples[:n_examples], 1):
        prompt_parts.append(f"""
Example {i}:
Item: {example.checklist_item_text}
Status: {example.status}
Comment: {example.comment}
""")
    
    prompt_parts.append("\n--- END EXAMPLES ---\n")
    prompt_parts.append("Use these examples as guidance for your evaluation style and detail level.\n")
    
    return "\n".join(prompt_parts)


# Singleton instance
_training_store = None

def get_training_store() -> TrainingDataStore:
    """Get the singleton training store instance"""
    global _training_store
    if _training_store is None:
        _training_store = TrainingDataStore()
    return _training_store
