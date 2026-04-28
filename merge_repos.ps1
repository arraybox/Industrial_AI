# ============================================================
#  Industrial_AI 하위 폴더 병합 스크립트 v5
# ============================================================

$account = "arraybox"
$baseDir  = "D:\99.GITHUB\Industrial_AI"

$repos = @(
    "Apprentice-1",
    "Apprentice-2",
    "Apprentice-3",
    "Apprentice-4",
    "CBLand",
    "image_labeling_tool",
    "Industrial_AI",
    "numpy_student",
    "PJT_11-01",
    "PJT_11-02",
    "PJT_11-03",
    "PJT_11-04",
    "PJT_21-01",
    "PJT_21-02",
    "PJT_21-03",
    "PJT_21-103",
    "PJT_21-105",
    "PJT_21-106",
    "PJT_21-107"
)

if (-not (Test-Path $baseDir)) {
    Write-Host "[ERROR] Path not found: $baseDir" -ForegroundColor Red
    exit 1
}
Set-Location $baseDir
Write-Host ""
Write-Host "=== Merging repos into Industrial_AI ===" -ForegroundColor Cyan
Write-Host "Path: $baseDir"
Write-Host ""

$success = @()
$failed  = @()

foreach ($repo in $repos) {
    $url = "https://github.com/$account/$repo.git"
    Write-Host "[$repo] merging..." -ForegroundColor Yellow

    git subtree add --prefix=$repo $url main --squash 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK (main)" -ForegroundColor Green
        $success += $repo
        continue
    }

    Write-Host "  main failed -> retry master..." -ForegroundColor DarkYellow
    git subtree add --prefix=$repo $url master --squash 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK (master)" -ForegroundColor Green
        $success += $repo
    } else {
        Write-Host "  FAIL" -ForegroundColor Red
        $failed += $repo
    }
}

Write-Host ""
Write-Host "=== Applying original repo creation dates ===" -ForegroundColor Cyan

foreach ($repo in $success) {
    $apiUrl = "https://api.github.com/repos/$account/$repo"
    try {
        $resp       = Invoke-RestMethod -Uri $apiUrl -Headers @{ "User-Agent" = "ps-script" } -ErrorAction Stop
        $createdAt  = [datetime]$resp.created_at
        $folderPath = Join-Path $baseDir $repo
        if (Test-Path $folderPath) {
            $item = Get-Item $folderPath
            $item.CreationTime  = $createdAt
            $item.LastWriteTime = $createdAt
            Write-Host "  [$repo] date set: $createdAt" -ForegroundColor Green
        }
    } catch {
        Write-Host "  [$repo] API failed: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================" -ForegroundColor Cyan
Write-Host "Result summary"
Write-Host "OK : $($success.Count)" -ForegroundColor Green

if ($failed.Count -gt 0) {
    Write-Host "FAIL: $($failed.Count)" -ForegroundColor Red
    foreach ($r in $failed) {
        Write-Host "  - $r" -ForegroundColor Red
    }
}
Write-Host "============================" -ForegroundColor Cyan

if ($success.Count -gt 0) {
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    git push origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Push OK!" -ForegroundColor Green
    } else {
        Write-Host "Push failed -> run manually: git push origin main" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================" -ForegroundColor Cyan
Write-Host "Delete commands (GitHub CLI)"
Write-Host "If gh not installed: winget install GitHub.cli  then  gh auth login"
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

foreach ($repo in $success) {
    Write-Host "gh repo delete $account/$repo --yes"
}
Write-Host ""