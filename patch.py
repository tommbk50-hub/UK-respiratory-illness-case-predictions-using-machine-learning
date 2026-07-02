import re

with open('sars-cov-2.html', 'r') as f:
    content = f.read()

# Add viewer-container styles
style_insertion = """
        .viewer-container {
            min-height: 760px;
            width: 100%;
            position: relative;
            overflow: hidden;
            border-radius: 0.75rem;
            background-color: var(--bg-white);
        }
        .viewer-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: 0;
        }"""

content = content.replace('.viz-card {', style_insertion + '\n\n        .viz-card {')

# Replace the Visualization section
old_viz = """    <!-- ================= Visualization ================= -->
    <section class="section section-soft">
        <div class="container">
            <div class="viz-card">
                <iframe class="viz-frame" src="SARS-CoV-2_Helicase_hotspot_visualization.html"
                        title="SARS-CoV-2 helicase ligand hotspot 3D visualization" loading="lazy"></iframe>
                <p class="viz-note" style="font-size: 0.85rem; color: #475569;">
                    <i class="fas fa-info-circle me-1" style="color: #0d9488;"></i>
                    Drag to rotate, scroll to zoom, and hover over the structure to explore predicted binding
                    hotspots. Colour indicates ligand occupancy, from low (green) to high (red).
                </p>
            </div>
        </div>
    </section>"""

new_viz = """    <!-- ================= Visualization ================= -->
    <section class="section section-soft">
        <div class="container">
            <div class="viz-card" style="margin-bottom: 2rem;">
                <h2 style="color: var(--heading); margin-bottom: 1rem; font-size: 1.5rem; font-family: 'Inter', sans-serif;">SARS-CoV-2 Helicase Hotspot Visualization</h2>
                <div class="viewer-container">
                    <iframe class="viz-frame" src="SARS-CoV-2_Helicase_hotspot_visualization.html"
                            title="SARS-CoV-2 helicase ligand hotspot 3D visualization" loading="lazy"></iframe>
                </div>
                <p class="viz-note" style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: var(--text-muted);">
                    <i class="fas fa-info-circle me-1" style="color: var(--accent);"></i>
                    Drag to rotate, scroll to zoom, and hover over the structure to explore predicted binding
                    hotspots. Colour indicates ligand occupancy, from low (green) to high (red).
                </p>
            </div>

            <div class="viz-card">
                <h2 style="color: var(--heading); margin-bottom: 1rem; font-size: 1.5rem; font-family: 'Inter', sans-serif;">DBSCAN Clustering Analysis Report</h2>
                <div class="viewer-container" style="min-height: 800px;">
                    <iframe class="viz-frame" src="dbscan_summary_report.html"
                            title="DBSCAN Clustering Analysis Report" loading="lazy"></iframe>
                </div>
                <p class="viz-note" style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: var(--text-muted);">
                    <i class="fas fa-chart-bar me-1" style="color: var(--accent);"></i>
                    Interactive DBSCAN clustering analysis summary report showing parameter space exploration and optimal clustering configurations for structural data.
                </p>
            </div>
        </div>
    </section>"""

content = content.replace(old_viz, new_viz)

with open('sars-cov-2.html', 'w') as f:
    f.write(content)
