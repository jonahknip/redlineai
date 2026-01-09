"""
Project Model - Manages planset projects and their analysis state
"""
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from config import PROJECTS_DIR


@dataclass
class Project:
    """A planset analysis project"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    filename: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Analysis state
    status: str = "pending"  # pending, analyzing, completed, error
    total_sheets: int = 0
    sheets_analyzed: int = 0
    
    # Project info extracted from drawings
    project_number: str = ""
    location: str = ""
    owner: str = ""
    engineer: str = ""
    
    # Analysis results
    sheets: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    elements: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # File paths (relative to project directory)
    pdf_path: str = ""
    marked_pdf_path: str = ""
    
    @property
    def project_dir(self) -> Path:
        """Get the project's storage directory"""
        return PROJECTS_DIR / self.id
    
    @property
    def full_pdf_path(self) -> Path:
        """Get full path to original PDF"""
        return self.project_dir / self.pdf_path if self.pdf_path else None
    
    @property
    def full_marked_pdf_path(self) -> Path:
        """Get full path to marked-up PDF"""
        return self.project_dir / self.marked_pdf_path if self.marked_pdf_path else None
    
    def create_directory(self) -> Path:
        """Create the project directory"""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        return self.project_dir
    
    def save_pdf(self, source_path: Path, filename: str) -> str:
        """Copy PDF to project directory"""
        self.create_directory()
        dest_path = self.project_dir / filename
        shutil.copy2(source_path, dest_path)
        self.pdf_path = filename
        self.filename = filename
        return str(dest_path)
    
    def save(self) -> None:
        """Save project metadata to JSON"""
        self.updated_at = datetime.now().isoformat()
        self.create_directory()
        
        metadata_path = self.project_dir / 'project.json'
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, project_id: str) -> Optional['Project']:
        """Load a project by ID"""
        project_dir = PROJECTS_DIR / project_id
        metadata_path = project_dir / 'project.json'
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def list_all(cls) -> List['Project']:
        """List all projects"""
        projects = []
        
        if not PROJECTS_DIR.exists():
            return projects
        
        for project_dir in PROJECTS_DIR.iterdir():
            if project_dir.is_dir():
                project = cls.load(project_dir.name)
                if project:
                    projects.append(project)
        
        # Sort by updated_at descending
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects
    
    def delete(self) -> bool:
        """Delete the project and all its files"""
        if self.project_dir.exists():
            shutil.rmtree(self.project_dir)
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary for listing"""
        # Count findings by severity
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0, 'info': 0}
        for finding in self.findings:
            sev = finding.get('severity', 'info')
            if sev in severity_counts:
                severity_counts[sev] += 1
        
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'status': self.status,
            'total_sheets': self.total_sheets,
            'sheets_analyzed': self.sheets_analyzed,
            'total_findings': len(self.findings),
            'findings_by_severity': severity_counts,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
