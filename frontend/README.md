# Finvarta Frontend

React UI for sending company analysis requests to the FastAPI backend.

## Prerequisites

- Node.js 18+ and npm
- FastAPI server running locally at `http://localhost:8000`

## Setup

```bash
cd frontend
npm install
```

## Run the Dev Server

```bash
npm run dev
```

Open the printed Vite URL (defaults to `http://localhost:5173`) in a browser.

## Usage

1. Ensure the FastAPI backend is running and exposes `POST /analyze`.
2. Enter a company name in the input field.
3. Click **Submit** to fetch the analysis. Results (raw JSON) render on the right panel.

Any errors returned by the API will be displayed below the form.
