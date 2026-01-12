# Training Data Directory

This directory contains all training resources for Redline.AI's QA/QC review system.

## Directory Structure

```
training_data/
├── reference_checklists/     # Original PDF checklists (for reference)
│   ├── 30_Checklist_Fillable.pdf
│   ├── 60_Checklist_Fillable.pdf
│   ├── 90_Checklist_Fillable.pdf
│   ├── CADD_Drawing_Checklist.pdf
│   └── QA_QC_Plan.pdf
│
├── completed_reviews/        # Completed review examples for training
│   ├── 30_percent/          # 30% phase reviews
│   ├── 60_percent/          # 60% phase reviews
│   ├── 90_percent/          # 90% phase reviews
│   └── full_review/         # Full reviews (all phases)
│
├── planset_library/         # Reference plansets categorized by type
│   ├── road_projects/
│   ├── utility_projects/
│   ├── site_development/
│   └── demo/                # Demo plansets for testing
│
├── measurement_examples/    # Q&A pairs for measurement questions
│   └── (JSON files by category)
│
└── few_shot_examples/       # Few-shot learning examples
    └── training_examples.json
```

## Adding Training Data

### Completed Reviews
1. Create a folder: `completed_reviews/{phase}/project_name/`
2. Add the planset PDF
3. Add `review_results.json` with the final eval_data

### Plansets
Add PDF plansets to the appropriate category folder in `planset_library/`

### Measurement Examples
Add JSON files to `measurement_examples/` with format:
```json
{
  "checklist_item_id": "90-UTL-002",
  "question_type": "distance",
  "question_text": "What is the separation at STA 12+50?",
  "user_answer": "12.5 feet",
  "result_status": "PASS",
  "project_type": "road"
}
```
