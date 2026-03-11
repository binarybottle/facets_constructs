# Item Assessment Study

A jsPsych-based forced-choice experiment for understanding which student qualities teachers consider most important to assess.

**Author**: Arno Klein (arnoklein.info)

**License**: Apache v2.0


## Overview

This study presents teachers with pairs of student qualities (e.g., "Self-Control" vs. "Empathy") and asks them to choose which quality is more important for them to understand when reflecting on their students.

**Research Focus**: Understanding teacher priorities for student assessment to inform educational tool development.


## Study Design

- **Task**: Forced binary choice between two qualities
- **Prompt**: "When reflecting on your students, which quality is more important for you to understand/assess?"
- **Items**: 104 student qualities with 3 synonyms each (416 total terms)
- **Pair generation**: Random selection of one item paired with another item or any synonym


## Data Collected

Each trial records:
- `left_term`, `right_term`: The two options presented
- `left_is_item`, `right_is_item`: Whether each is an original item or synonym
- `left_source`, `right_source`: The source item (for synonyms)
- `chosen_term`, `unchosen_term`: The participant's choice
- `response_time`: Time to make choice (ms)


## Configuration

Edit `src/experiment.js` to change:

```javascript
let experimentConfig = {
  numTrials: 10,              // Number of pairs to present
  itemsFile: 'data/items.csv' // CSV with items and synonyms
};
```


## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```


## Deployment

The study integrates with:
- **Prolific**: For participant recruitment (reads `PROLIFIC_PID` from URL)
- **OSF**: For data storage (requires `token.json` with API key)


## File Structure

```
study/
├── data/
│   └── items.csv          # 104 items with 3 synonyms each
├── src/
│   ├── experiment.js      # Main experiment code
│   └── style.css          # Styling
├── index.html
└── vite.config.js
```
