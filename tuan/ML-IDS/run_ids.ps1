# Two-tier IDS test (see "FLow test.md").
#   tier 1 : binary train fold (benign vs attack), the same for every configuration
#   tier 2 : A original / B multiclass / C multiclass + CTGAN / D multiclass + TVAE
#   test   : binary test fold (benign + attack), never trained on
# Usage:  .\run_ids.ps1                    (fold 1, GPU)
#         .\run_ids.ps1 -Folds 1,2,3,4,5   (all folds, mean +/- std)
#         .\run_ids.ps1 -Trees 100         (quick check)
#         .\run_ids.ps1 -Cpu               (CPU only)
param(
    [int[]]$Folds   = @(1),
    [string]$OutDir = "ids_report",
    [int]$Trees     = 300,
    [switch]$Cpu
)

$bin   = "IDS_Data/IDS-binary-classification_MIA"
$multi = "IDS_Data/IDS-multiclass-classification/unsw_nb15_attacks_smote_10k.csv"
$ctgan = "Synthetic Data (for test)/CTGAN/synthetic_ctgan_unsw_nb15_smote_10k_smote10k_v2.csv"
$tvae  = "Synthetic Data (for test)/TVAE/synthetic_tvae_perclass_smote_10k.csv"
$orig  = "Traffic Flow/unsw_nb15_attacks_original.csv"

$argsList = @(
    "ids_pipeline.py",
    "--train", "$bin/train/NF-UNSW-NB15-v3_fold_{fold}.csv",
    "--test",  "$bin/test/NF-UNSW-NB15-v3_fold_{fold}.csv",
    "--folds") + ($Folds | ForEach-Object { "$_" }) + @(
    "--config", "A_original=$orig",
    "--config", "B_multiclass=$multi",
    "--config", "C_multiclass+CTGAN=$multi,$ctgan",
    "--config", "D_multiclass+TVAE=$multi,$tvae",
    "--baseline", "B_multiclass",
    "--trees", "$Trees",
    "--out-dir", $OutDir
)
if (-not $Cpu) { $argsList += "--gpu" }

Set-Location $PSScriptRoot
python @argsList
exit $LASTEXITCODE
