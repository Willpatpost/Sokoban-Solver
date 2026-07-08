# Sokomind Web

The web edition is a responsive, dependency-free browser game ready for GitHub
Pages. It includes five levels from Ultra Tiny through Huge, keyboard play,
undo/reset, a first-move timer, hints, and animated solving in a background Web
Worker.

## How to play

Push every box onto its matching goal:

- `X` boxes go on `S` goals.
- Lettered boxes such as `A` go on the matching lowercase goal, such as `a`.
- Boxes can be pushed but cannot be pulled.

Use the arrow keys or WASD to move. Press U to undo and R to reset. The timer
starts on the first legal move.

## Local preview

From the repository root:

```powershell
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## GitHub Pages deployment

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Choose **GitHub Actions** as the source.
4. Push to `main` or manually run the included Pages workflow.

The `.github/workflows/pages.yml` workflow publishes this `docs/` directory.
No build command or package installation is required.
