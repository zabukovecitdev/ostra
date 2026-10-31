from nicegui import ui

PRIMARY_COLOR = "#0f766e"
# Quasar props that give every field a flat outline in place of a filled box.
FIELD_PROPS = "outlined dense"
BUTTON_PROPS = "unelevated no-caps"

# Flat design: no shadows, thin borders, one accent color. Colors are variables, so dark mode changes only the values.
STYLE = """
:root {
  --bg: #fafaf9; --surface: #ffffff; --text: #1c1917; --muted: #78716c; --border: #e7e5e4;
  --accent: #0f766e; --accent-soft: #ccfbf1; --accent-text: #115e59;
  color-scheme: light;
}
body.body--dark {
  --bg: #141413; --surface: #1c1b1a; --text: #f5f5f4; --muted: #a8a29e; --border: #2e2c2a;
  --accent: #2dd4bf; --accent-soft: #134e4a; --accent-text: #99f6e4;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--text);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
.q-header { background: var(--surface) !important; color: var(--text) !important;
  border-bottom: 1px solid var(--border); box-shadow: none !important; }
.q-card { background: var(--surface); color: var(--text); box-shadow: none !important;
  border: 1px solid var(--border); border-radius: 12px; }
.q-btn { box-shadow: none !important; border-radius: 8px; }
.q-field--outlined .q-field__control { border-radius: 8px; }
.ostra-page { width: 100%; max-width: 760px; margin: 0 auto; padding: 24px 16px; gap: 16px; }
.ostra-brand { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }
.ostra-nav { color: var(--muted); text-decoration: none; padding: 4px 10px; border-radius: 8px; }
.ostra-nav:hover { background: var(--bg); color: var(--text); }
.ostra-nav--active { background: var(--accent-soft); color: var(--accent-text); font-weight: 600; }
.ostra-muted { color: var(--muted); font-size: 0.85rem; }
.ostra-score { background: var(--accent-soft); color: var(--accent-text); font-weight: 600; font-size: 0.8rem;
  padding: 2px 10px; border-radius: 999px; }
.ostra-text { white-space: pre-wrap; line-height: 1.55; }
"""


def apply_theme() -> None:
    """Set the colors and the stylesheet of the current page. Call it at the start of each page function."""
    ui.colors(primary=PRIMARY_COLOR)
    ui.add_css(STYLE)
