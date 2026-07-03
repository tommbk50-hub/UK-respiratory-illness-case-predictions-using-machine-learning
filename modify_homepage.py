import re

with open('sars-cov-2_homepage.html', 'r') as f:
    content = f.read()

# Update title
content = re.sub(
    r'<title>.*?</title>',
    '<title>SARS-CoV-2 | OpenDiscovery AI</title>',
    content
)

# Update page header
new_header = """    <section class="page-header">
        <div class="container text-center">
            <span class="page-eyebrow"><i class="fas fa-virus-covid"></i> Respiratory</span>
            <h1>SARS-CoV-2 <span class="accent">Research</span></h1>
            <p class="lead-copy mx-auto">
                Explore our machine learning-driven insights and interactive tools for SARS-CoV-2.
            </p>
        </div>
    </section>"""

content = re.sub(
    r'    <section class="page-header">.*?</section>',
    new_header,
    content,
    flags=re.DOTALL
)

# Update visualization section
new_section = """    <!-- ================= Main Content ================= -->
    <section class="section section-soft">
        <div class="container text-center">
            <div class="row justify-content-center g-4 py-5">
                <div class="col-md-6 col-lg-5">
                    <a href="sars-cov-2_fragments.html" class="text-decoration-none">
                        <div class="card h-100 shadow-sm border-0" style="border-radius: 1rem; overflow: hidden; transition: transform 0.2s;">
                            <div class="card-body p-5">
                                <i class="fas fa-microscope mb-4" style="font-size: 3rem; color: var(--accent);"></i>
                                <h3 class="h4 mb-3" style="color: var(--heading);">Fragment Cluster Analysis</h3>
                                <p class="text-muted mb-4" style="color: var(--text-muted) !important;">Explore 3D predicted ligand-binding hotspots and clustering analysis on the SARS-CoV-2 helicase.</p>
                                <span class="btn btn-primary px-4 py-2" style="background-color: var(--accent); border: none; border-radius: 2rem;">Fragment Cluster Analysis</span>
                            </div>
                        </div>
                    </a>
                </div>
                <div class="col-md-6 col-lg-5">
                    <a href="dashboard.html" class="text-decoration-none">
                        <div class="card h-100 shadow-sm border-0" style="border-radius: 1rem; overflow: hidden; transition: transform 0.2s;">
                            <div class="card-body p-5">
                                <i class="fas fa-chart-line mb-4" style="font-size: 3rem; color: var(--accent);"></i>
                                <h3 class="h4 mb-3" style="color: var(--heading);">Live COVID and Influenza Case Prediction</h3>
                                <p class="text-muted mb-4" style="color: var(--text-muted) !important;">View real-time predictions for respiratory illness cases using machine learning models.</p>
                                <span class="btn btn-primary px-4 py-2" style="background-color: var(--accent); border: none; border-radius: 2rem;">Live COVID and Influenza Case Prediction</span>
                            </div>
                        </div>
                    </a>
                </div>
            </div>
        </div>
    </section>"""

content = re.sub(
    r'    <!-- ================= Visualization ================= -->.*?</section>',
    new_section,
    content,
    flags=re.DOTALL
)

with open('sars-cov-2_homepage.html', 'w') as f:
    f.write(content)
