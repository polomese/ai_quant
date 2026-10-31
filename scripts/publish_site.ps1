<#
.SYNOPSIS
  Build the results site for one run and publish it to GitHub Pages (the gh-pages branch).

.DESCRIPTION
  The site (`pitlab site`) is a single HTML file with aggregate results only. It is pushed
  to an orphan `gh-pages` branch as a single commit, force-pushed each time, so the ~5 MB
  page never piles up in the code history on main. GitHub Pages serves that branch at
  https://<user>.github.io/<repo>/.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\publish_site.ps1
  powershell -ExecutionPolicy Bypass -File scripts\publish_site.ps1 -RunId real_v3
#>
param(
    [string]$RunId = "real_v2",
    [string]$Remote = "origin"
)
# "Continue": git writes progress to stderr, which Windows PowerShell 5.1 would turn into
# terminating errors under "Stop". Failures are caught from $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$proj = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\python.exe"
$work = Join-Path ([IO.Path]::GetTempPath()) "pitlab-gh-pages"
$page = Join-Path $work "site.html"

Push-Location $proj
try {
    if (Test-Path $work) { git worktree remove --force $work 2>$null; Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory -Force $work | Out-Null
    $env:PYTHONIOENCODING = "utf-8"
    & $python -m pitlab.cli site --run-id $RunId --out $page
    if ($LASTEXITCODE) { throw "site build failed" }
    $sha = (git rev-parse --short HEAD).Trim()

    $tree = Join-Path $work "tree"
    git branch -D pages-tmp 2>$null | Out-Null   # leftover from an interrupted run
    git worktree add --detach $tree | Out-Null
    if ($LASTEXITCODE) { throw "git worktree add failed" }
    Push-Location $tree
    try {
        git checkout --orphan pages-tmp | Out-Null
        if ($LASTEXITCODE) { throw "git checkout --orphan failed" }
        git rm -rf --quiet . | Out-Null
        Move-Item $page (Join-Path $tree "index.html")
        New-Item -ItemType File ".nojekyll" | Out-Null   # serve the file as-is
        git add index.html .nojekyll
        git commit --quiet -m "Results site for run $RunId (code $sha)"
        if ($LASTEXITCODE) { throw "git commit failed" }
        git push --force $Remote HEAD:gh-pages
        if ($LASTEXITCODE) { throw "push failed" }
    }
    finally { Pop-Location }
    git worktree remove --force $tree
    git branch -D pages-tmp 2>$null | Out-Null
    Write-Host "published run $RunId to the gh-pages branch"
}
finally { Pop-Location; Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
