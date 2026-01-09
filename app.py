"""
RedlineAI - Flask Web Application
AI-powered civil engineering plan review
"""

import os
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename

from agent.checklist_engine import ChecklistEngine
from agent.report_generator import ReportGenerator


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'redlineai-dev-key-change-in-prod')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Upload configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'redlineai_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

# Store review results in memory (use Redis/DB in production)
review_results_store = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or session."""
    return os.environ.get('OPENAI_API_KEY') or session.get('api_key')


@app.route('/')
def index():
    """Main page with upload form."""
    try:
        engine = ChecklistEngine(api_key=None)  # Don't need API key for listing checklists
        checklists = engine.get_available_checklists()
    except Exception as e:
        # Fallback checklists if loading fails
        checklists = [
            {'phase': '30%', 'name': '30% Engineering QA/QC Review', 'description': 'Early design validation', 'item_count': 45},
            {'phase': '60%', 'name': '60% Engineering QA/QC Review', 'description': 'Mid-design review', 'item_count': 55},
            {'phase': '90%', 'name': '90% Engineering QA/QC Review', 'description': 'Final design review', 'item_count': 75},
            {'phase': 'CADD', 'name': 'CADD Drawing Review', 'description': 'CADD standards review', 'item_count': 35},
        ]
    return render_template('index.html', checklists=checklists)


@app.route('/api/checklists', methods=['GET'])
def get_checklists():
    """Get available checklists."""
    try:
        engine = ChecklistEngine(api_key=get_api_key() or 'dummy')
        checklists = engine.get_available_checklists()
        return jsonify({'success': True, 'checklists': checklists})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review', methods=['POST'])
def start_review():
    """Start a plan review."""
    try:
        # Check for API key
        api_key = get_api_key()
        if not api_key:
            return jsonify({
                'success': False, 
                'error': 'OpenAI API key not configured. Set OPENAI_API_KEY environment variable.'
            }), 400
        
        # Get phase
        phase = request.form.get('phase')
        if not phase:
            return jsonify({'success': False, 'error': 'Phase is required'}), 400
        
        # Get custom instructions
        custom_instructions = request.form.get('custom_instructions', '')
        
        # Handle file uploads
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Save uploaded files
        review_id = str(uuid.uuid4())
        review_folder = UPLOAD_FOLDER / review_id
        review_folder.mkdir(exist_ok=True)
        
        pdf_paths = []
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = review_folder / filename
                file.save(str(filepath))
                pdf_paths.append(str(filepath))
        
        if not pdf_paths:
            return jsonify({'success': False, 'error': 'No valid PDF files uploaded'}), 400
        
        # Initialize engine and run review
        engine = ChecklistEngine(api_key=api_key)
        report_gen = ReportGenerator()
        
        # Run the review
        results = engine.run_review(
            pdf_paths=pdf_paths,
            phase=phase,
            custom_instructions=custom_instructions if custom_instructions else None
        )
        
        # Add form metadata from user input
        results['form_project_name'] = request.form.get('project_name', '')
        results['form_project_number'] = request.form.get('project_number', '')
        results['form_project_manager'] = request.form.get('project_manager', '')
        results['form_reviewer'] = request.form.get('reviewer', '') or 'RedlineAI'
        
        # Generate HTML report
        html_report = report_gen.generate_html_report(results)
        
        # Store results
        review_results_store[review_id] = {
            'results': results,
            'html_report': html_report,
            'created_at': datetime.now().isoformat()
        }
        
        # Cleanup uploaded files
        for path in pdf_paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            review_folder.rmdir()
        except:
            pass
        
        return jsonify({
            'success': True,
            'review_id': review_id,
            'summary': results.get('summary', {}),
            'project': results.get('project_summary', {})
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/review/<review_id>/report', methods=['GET'])
def get_report(review_id: str):
    """Get HTML report for a review."""
    if review_id not in review_results_store:
        return jsonify({'success': False, 'error': 'Review not found'}), 404
    
    return review_results_store[review_id]['html_report']


@app.route('/api/review/<review_id>/results', methods=['GET'])
def get_results(review_id: str):
    """Get JSON results for a review."""
    if review_id not in review_results_store:
        return jsonify({'success': False, 'error': 'Review not found'}), 404
    
    return jsonify({
        'success': True,
        'results': review_results_store[review_id]['results']
    })


@app.route('/api/review/<review_id>/export/word', methods=['GET'])
def export_word(review_id: str):
    """Export review as Word document."""
    if review_id not in review_results_store:
        return jsonify({'success': False, 'error': 'Review not found'}), 404
    
    try:
        results = review_results_store[review_id]['results']
        report_gen = ReportGenerator()
        
        buffer = report_gen.generate_word_report(results)
        
        project_name = results.get('project_summary', {}).get('project_name', 'Review')
        filename = f"RedlineAI_{project_name}_{results.get('phase', '')}_{datetime.now().strftime('%Y%m%d')}.docx"
        
        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/<review_id>/export/excel', methods=['GET'])
def export_excel(review_id: str):
    """Export quantities as Excel."""
    if review_id not in review_results_store:
        return jsonify({'success': False, 'error': 'Review not found'}), 404
    
    try:
        results = review_results_store[review_id]['results']
        report_gen = ReportGenerator()
        
        buffer = report_gen.generate_excel_quantities(results)
        
        project_name = results.get('project_summary', {}).get('project_name', 'Quantities')
        filename = f"RedlineAI_Quantities_{project_name}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/<review_id>/next-phase-items', methods=['GET'])
def get_next_phase_items(review_id: str):
    """Get items required for next phase."""
    if review_id not in review_results_store:
        return jsonify({'success': False, 'error': 'Review not found'}), 404
    
    results = review_results_store[review_id]['results']
    items = results.get('summary', {}).get('items_for_next_phase', [])
    next_phase = results.get('next_phase')
    
    return jsonify({
        'success': True,
        'next_phase': next_phase,
        'items': items,
        'count': len(items)
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '0.1.0',
        'api_key_configured': bool(os.environ.get('OPENAI_API_KEY'))
    })


@app.route('/api/debug')
def debug():
    """Debug endpoint to check configuration."""
    return jsonify({
        'api_key_set': bool(os.environ.get('OPENAI_API_KEY')),
        'api_key_prefix': os.environ.get('OPENAI_API_KEY', '')[:8] + '...' if os.environ.get('OPENAI_API_KEY') else None,
        'upload_folder': str(UPLOAD_FOLDER),
        'upload_folder_exists': UPLOAD_FOLDER.exists(),
        'checklists_dir': str(Path(__file__).parent / 'checklists'),
        'checklists_exist': (Path(__file__).parent / 'checklists').exists()
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 100MB.'}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
