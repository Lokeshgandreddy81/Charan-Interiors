# Charan-Interiors

Designing A generative model

---

## Overview

This repository contains the full application suite and tools for **Charan Interiors**, featuring the **Expense & Receipts Analytics** platform, benchmarking suites, and document processing pipelines.

## Project Structure

```
.
├── .gitignore
├── README.md
├── screenshots/
│   └── leaderboard-dropdown-open.png
└── expense-&-receipts-analytics/
    ├── package.json
    ├── server.ts
    ├── vite.config.ts
    ├── index.html
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   └── utils/
    ├── scripts/
    │   ├── pdf_chapter_extractor.py
    │   └── pdf_roadmap_generator.py
    └── .env.example
```

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18+ recommended)
- [npm](https://www.npmjs.com/) or [Bun](https://bun.sh/)
- [Python 3.8+](https://www.python.org/) (for PDF extraction utilities)

### Setup & Run Locally

1. Navigate to the application directory:
   ```bash
   cd expense-&-receipts-analytics
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Provide your `GEMINI_API_KEY` in the `.env` file.

4. Start the development server:
   ```bash
   npm run dev
   ```

## Security & Best Practices

- Environment variables and API keys (`.env`) are strictly excluded from version control.
- Build artifacts (`dist/`) and dependencies (`node_modules/`) are managed via `.gitignore`.
