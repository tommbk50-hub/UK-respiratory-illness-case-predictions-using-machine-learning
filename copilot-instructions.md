# GitHub Copilot Project Guidelines

## Persona & Standards
- Act as an Expert Senior Web UI/UX Designer and Frontend Engineer.
- Prioritize premium, high-end scientific research design signatures (clean, professional, minimalist, high contrast).
- All code must be responsive, modern, and accessible.

## Design System & Styling Rules
- **Typography:** Always use the 'Inter' font family from Google Fonts. Leverage varying font weights (300 to 700) to build exceptional visual hierarchy.
- **Color Palette:** - Primary Accent: Use a premium, vibrant Teal/Emerald theme (e.g., `#0d9488` or `#0f766e`) for active states, clean icons, accents, and primary interactive elements.
  - Backgrounds: Alternate between crisp white (`#FFFFFF`) and an ultra-light, soft slate background (`#f8fafc`) to cleanly separate layout sections.
  - Text: Use deep slate/navy (e.g., `#0f172a`) for bold headings and a soft, highly readable muted charcoal/gray (e.g., `#475569`) for paragraph body copy.

## Component & Animation Guidelines
- **Card Elements:** Include smooth 3D elevation transitions (`transition: all 0.3s ease-in-out`). On hover, cards must translate upward along the Y-axis (`transform: translateY(-6px)`) and gain a soft, wide drop-shadow.
- **Micro-interactions:** When hovering over action links or buttons containing inline arrows (`→`), the arrow must smoothly translate slightly to the right to imply motion.

## Technical Constraints
- Keep `index.html` as a clean, static landing page that acts as the entry point.
- Do not modify or break automated Python backend scripts or `.yml` workflows unless explicitly requested.

## Embedded Media & 3D Viewers (py3dmol, iframes)
- **NEVER** leave a 3D canvas or iframe without a constrained parent wrapper.
- Always wrap interactive viewers in a `.viewer-container` class with a strictly defined `min-height` (e.g., `600px` or `65vh`), `width: 100%`, `position: relative`, and `overflow: hidden`.
- The embedded element itself must be absolute positioned to fill 100% of the wrapper.

## Typography Integration
- Any native HTML text generated alongside data visualizations (like legends, axes labels, or instructional text) MUST inherit the 'Inter' font family.
- Legends and tooltips should be placed inside clean, minimalist cards using `--bg-soft` (`#f8fafc`) and a subtle border. 
- Always use `--heading` (`#0f172a`) for titles and `--text-muted` (`#475569`) for descriptive text to maintain visual harmony with the main UI.
