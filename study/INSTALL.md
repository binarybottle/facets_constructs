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
# 1. Build locally
cd /Users/arno/Software/facets_constructs/study
npm run build

# 2. Set remote path
export STUDY='/home/binarybottle/arnoklein.info/facets/facets_study'

# 3. Create directory and upload
ssh binarybottle@arnoklein.info "mkdir -p $STUDY"
scp -r dist/* binarybottle@arnoklein.info:$STUDY/
scp token.json binarybottle@arnoklein.info:$STUDY/
scp -r data binarybottle@arnoklein.info:$STUDY/
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
