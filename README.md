# AdaBoost Practice — Front-end Dataset Viewer

This is a minimal, front-end-only web app (no frameworks) to help students practice concepts around AdaBoost. It generates a small synthetic binary classification dataset (2 or sometimes 3 features) and renders it in a table when the page loads.

## How to run

Open `index.html` in a browser. If your environment restricts file URLs, you can serve the folder locally with a simple HTTP server.

Optional, using Python 3:

```bash
python3 -m http.server 8000
```

Then visit http://localhost:8000 in your browser and open `index.html`.

## Notes

- Labels are derived from a linear boundary on f1 and f2 with a bit of noise; a third feature (f3) may be included as an extra weak signal.
- The generator includes slight overlap and a small random label flip to reduce cases of perfect separation (which can lead to ±∞ say on very small datasets).
- Use the "Regenerate dataset" button to get a new random sample. The dataset always contains exactly 15 rows; each regeneration randomly chooses either a 2-feature or 3-feature dataset.