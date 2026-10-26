[CmdletBinding()]
param(
    [int]$Port = 11436
)

$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) {
    throw 'Expected an unprivileged local port.'
}
$hyperRepo = Split-Path -Parent $PSScriptRoot
$serviceRoot = Join-Path $hyperRepo ('monitoring/embedding_service_' + $Port)
$ollamaBinary = Join-Path $env:LOCALAPPDATA 'Programs/Ollama/ollama.exe'
$ollamaModels = Join-Path $env:USERPROFILE '.ollama/models'
if (-not (Test-Path -LiteralPath $ollamaBinary) -or
    -not (Test-Path -LiteralPath $ollamaModels)) {
    throw 'Existing Ollama executable/model store is missing.'
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw ('Port already has a listener; inspect ownership before reuse: ' + $Port)
}
New-Item -ItemType Directory -Path $serviceRoot -Force | Out-Null
$settings = @{
    OLLAMA_HOST = ('127.0.0.1:' + $Port)
    OLLAMA_MODELS = $ollamaModels
    OLLAMA_MAX_LOADED_MODELS = '1'
    OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_KEEP_ALIVE = '24h'
}
$previous = @{}
try {
    foreach ($name in $settings.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    $service = Start-Process -FilePath $ollamaBinary -ArgumentList @('serve') `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $serviceRoot 'stdout.log') `
        -RedirectStandardError (Join-Path $serviceRoot 'stderr.log')
} finally {
    foreach ($name in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}
$record = @{
    pid = $service.Id
    host = ('127.0.0.1:' + $Port)
    endpoint = ('http://127.0.0.1:' + $Port + '/api/embed')
    started_local = (Get-Date -Format o)
    executable = $ollamaBinary
    executable_sha256 = (Get-FileHash -LiteralPath $ollamaBinary -Algorithm SHA256).Hash.ToLowerInvariant()
    scoped_environment = $settings
    model_store_scope = 'Read existing pinned model; no pull/create/delete requested.'
    purpose = 'HyperGraphRAG isolated BGE inference; fixed context and compute batch.'
}
$encoded = $record | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText(
    (Join-Path $serviceRoot 'service.json'), $encoded,
    [System.Text.UTF8Encoding]::new($false))
Write-Output $encoded
