"""
Phase 55 Colab Notebook
Save this as a .ipynb file or copy-paste the cells into Google Colab

Cell 1: Setup (2-3 minutes)
Cell 2: Run Phase 55 (30-40 minutes)

INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. Click "New notebook"
3. Paste this code into cells as shown below
4. Run each cell
"""

# ============================================================================
# CELL 1: SETUP - Clone DELTA and install dependencies (2-3 min)
# ============================================================================
# Paste this entire block into Cell 1 of your Colab notebook:

"""
!git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
%cd /content/DELTA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
print("✅ Setup complete! DELTA cloned and dependencies installed.")
"""


# ============================================================================
# CELL 2: RUN PHASE 55 - Execute the experiment (30-40 min)
# ============================================================================
# Paste this entire block into Cell 2 of your Colab notebook:

"""
!python phase55_colab_launcher.py \
    --seeds 42 \
    --epochs 150 \
    --eval_every 30 \
    --target_density 0.02
"""


# ============================================================================
# CELL 3 (OPTIONAL): Download results to your machine
# ============================================================================
# Paste this if you want to download results (after Phase 55 completes):

"""
from google.colab import files
import json

# Download the results file
files.download('/content/DELTA/phase55_output.json')

# Also download the text output
!cp /content/DELTA/phase55_output.txt . 2>/dev/null || echo "Output text not yet available"
files.download('/content/DELTA/phase55_output.txt') 2>/dev/null || print("Text output will be available after run")

print("✅ Results downloaded to your personal machine")
"""


# ============================================================================
# QUICK START (Copy-paste these exact commands)
# ============================================================================

"""
TO LAUNCH PHASE 55 ON COLAB IN 3 STEPS:

Step 1: Open https://colab.research.google.com

Step 2: Create new notebook and click the first cell

Step 3: Paste and run this (2-3 minutes for setup):
---
!git clone https://github.com/bdbrown4/DELTA.git /content/DELTA
%cd /content/DELTA
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
---

Step 4: In a new cell, paste and run this (30-40 minutes for experiment):
---
!python phase55_colab_launcher.py --seeds 42 --epochs 150 --eval_every 30 --target_density 0.02
---

Step 5 (Optional): Download results with this final cell:
---
from google.colab import files
files.download('/content/DELTA/phase55_output.json')
---

DONE! Results will be in your Downloads folder.
"""
