[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [ValidateSet('hotpotqa','2wiki','musique','all')][string]$Dataset = 'all',
    [string]$Name = 'public_reproduction',
    [string]$Python = 'python',
    [switch]$Execute,
    [switch]$AllowModelApi
)
$ErrorActionPreference = 'Stop'
$supplement = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$arguments = @((Join-Path $supplement 'tools/reproduce.py'),
    '--profile','hyper-gpt4omini','--dataset',$Dataset,'--data-root',$DataRoot,
    '--run-name',$Name,'--phase','run')
if ($Execute) { $arguments += '--execute' }
if ($AllowModelApi) { $arguments += '--allow-model-api' }
& $Python @arguments
exit $LASTEXITCODE
