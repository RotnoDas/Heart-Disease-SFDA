$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================"
Write-Host "Heart SFDA Final Analysis"
Write-Host "============================================"
Write-Host ""

Write-Host "[1/4] Hierarchical bootstrap"
python scripts\run_hierarchical_bootstrap.py

Write-Host ""
Write-Host "[2/4] Final statistical tables"
python scripts\build_final_statistical_tables.py

Write-Host ""
Write-Host "[3/4] Manuscript tables"
python scripts\build_manuscript_tables.py

Write-Host ""
Write-Host "[4/4] Publication forest plots"
python scripts\plot_publication_forest_v2.py

Write-Host ""
Write-Host "============================================"
Write-Host "Final analysis complete."
Write-Host "============================================"