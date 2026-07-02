import re

filename = '/home/runner/work/UK-respiratory-illness-case-predictions-using-machine-learning/UK-respiratory-illness-case-predictions-using-machine-learning/SARS-CoV-2_Helicase_hotspot_visualization.html'

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

# Replacement HTML
new_head = """<html>
<head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.viewer-container {
    width: 100%;
    min-height: 600px;
    height: 65vh;
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.legend-card {
    margin-top: 10px;
    padding: 1.25rem;
    border: 1px solid #e2e8f0;
    background-color: #f8fafc;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    font-family: 'Inter', sans-serif;
    margin-bottom: 1.5rem;
}
.legend-title {
    color: #0f172a;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 0.75rem;
}
.legend-label {
    color: #475569;
    font-weight: 500;
    font-size: 0.9rem;
}
</style>
</head>
<body>
<div class="legend-card">
  <h4 class="legend-title">Ligand Hotspot Color Coding</h4>
  <div style="display: flex; align-items: center;">
    <span class="legend-label" style="margin-right: 10px;">Low Occupancy (Green)</span>
    <div style="width: 150px; height: 20px; background: linear-gradient(to right, #00ff00, #ffff00, #ff0000); border-radius: 4px;"></div>
    <span class="legend-label" style="margin-left: 10px;">High Occupancy (Red)</span>
  </div>
</div>
<div class="viewer-container">
<div id="3dmolviewer_17830090239516625" style="width: 100%; height: 100%; position: absolute; top: 0; left: 0;">"""

# We need to replace the original beginning of the file with `new_head`
# We also need to close the .viewer-container div at the end of the file. Wait, does the viewer div close normally?
# Yes, <div id="..."> ... </div>, so we just need to append </div>

# Let's find the old part:
old_part_pattern = r'<html><body>\s*<div style="margin-top: 10px; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">.*?<br><div id="3dmolviewer_17830090239516625"  style="position: relative; width: 800px; height: 600px;">'

new_content = re.sub(old_part_pattern, new_head, content, flags=re.DOTALL)

# Insert closing div for viewer-container
# The original ends with </body></html>
new_content = new_content.replace('</body>\n</html>', '</div>\n</body>\n</html>')
new_content = new_content.replace('</body></html>', '</div></body></html>')


with open(filename, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("HTML updated successfully.")
