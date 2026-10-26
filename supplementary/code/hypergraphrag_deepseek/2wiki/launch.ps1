[CmdletBinding()]
param(
    [ValidateSet('plan','prepare','run','evaluate','verify')]
    [string]$Action = 'plan',
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [string]$Python = 'python',
    [switch]$AllowModelApi
)
$ErrorActionPreference = 'Stop'
$supplement = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$dataset = Split-Path -Leaf $PSScriptRoot
$arguments = @((Join-Path $supplement 'tools/reproduce.py'),
    '--profile','hyper-deepseek','--dataset',$dataset,'--data-root',$DataRoot,
    '--phase',$Action)
if ($Action -ne 'plan') { $arguments += '--execute' }
if ($AllowModelApi) { $arguments += '--allow-model-api' }
& $Python @arguments
exit $LASTEXITCODE
