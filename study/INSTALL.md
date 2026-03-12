# Installation

## Prerequisites

- Node.js >= 18.x (on your local machine only)

## Local Development

```bash
git clone git@github.com:binarybottle/facets_constructs.git
cd facets_constructs/study
npm install
npm run dev
```

Visit the URL shown (e.g., `http://localhost:8082/...`) to preview.

## Remote Production Deployment

Build locally, then upload static files (no Node.js needed on server):

```bash
# First-time setup: create the directory and upload data/token
export STUDY='binarybottle@arnoklein.info:/home/binarybottle/arnoklein.info/facets/facets_study'
ssh binarybottle@arnoklein.info "mkdir -p ${STUDY#*:}"
scp token.json $STUDY/
scp -r data $STUDY/
```

```bash
# Regenerate subsets (run whenever items.csv or k/n parameters change)
cd /Users/arno/Software/facets_constructs/study
python3 generate_subsets.py --k 26 --n 400
scp data/subsets.json binarybottle@arnoklein.info:/home/binarybottle/arnoklein.info/facets/facets_study/data/
```

```bash
# Every code deploy: build and upload (run from the study/ directory)
cd /Users/arno/Software/facets_constructs/study
npm run build && ssh binarybottle@arnoklein.info "rm -rf /home/binarybottle/arnoklein.info/facets/facets_study/assets" && scp -r dist/* binarybottle@arnoklein.info:/home/binarybottle/arnoklein.info/facets/facets_study/
```

## Vite Configuration

If deploying to a subdirectory, update `vite.config.js`:

```javascript
export default {
  base: '/facets/facets_study/',
}
```

The `base` path should match your server's URL path.

## Files Required on Server

After deployment, the server directory should contain:
```
facets_study/
├── index.html
├── assets/          # Built JS/CSS
├── data/
│   └── items.csv
└── token.json       # OSF API token (for data upload)
```
