# ==============================================================================
# INTEGRATED CHRONOGRAPH & KURAMOTO ORDER PARAMETER ENGINE (PS 5.1 NATIVE)
# ==============================================================================
$ErrorActionPreference = "Stop"
Push-Location "C:\RobdoeDeepSeekN-V3"

$hosts = @("time.google.com", "8.8.8.8", "8.8.4.4", "google.com", "1.1.1.1", "9.9.9.9")
$N = $hosts.Count
$baseFreq = 0.05208333
$periodSec = 1.0 / $baseFreq
$K = 3.5

Write-Host "================================================================================" -ForegroundColor DarkCyan
Write-Host "  MASTER CHRONOGRAPH & KURAMOTO ORDER PARAMETER (R) ENGINE" -ForegroundColor Cyan
Write-Host "  Oscillator Freq: $baseFreq Hz | Sweep Period: $([math]::Round($periodSec, 2))s" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor DarkCyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$ping = New-Object System.Net.NetworkInformation.Ping
$phases = New-Object double[] $N
$latencies = New-Object double[] $N

# 1. Asynchronous Network Socket Latency Sampling
for ($i = 0; $i -lt $N; $i++) {
    try {
        $reply = $ping.Send($hosts[$i], 1200)
        $rtt = if ($reply -and $reply.Status -eq "Success") { [double]$reply.RoundtripTime } else { 80.0 }
        if ($rtt -le 0) { $rtt = 1.0 }
    } catch {
        $rtt = 120.0
    }
    $latencies[$i] = $rtt
    $phases[$i] = (($rtt % 360.0) * [Math]::PI / 180.0)
}

# 2. Chronograph Sweep & Phase Coupling Calculation
$newPhases = New-Object double[] $N
$sumCos = 0.0
$sumSin = 0.0

Write-Host "`n--- LIVE NODE ESCAPEMENT & PHASE MATRIX ---" -ForegroundColor Yellow

for ($i = 0; $i -lt $N; $i++) {
    $interaction = 0.0
    for ($j = 0; $j -lt $N; $j++) {
        $interaction += [Math]::Sin($phases[$j] - $phases[$i])
    }
    $dTheta = $baseFreq + ($latencies[$i] / 1000.0) + ($K / $N) * $interaction
    $newPhases[$i] = ($phases[$i] + $dTheta * 0.1) % (2.0 * [Math]::PI)
    
    $sumCos += [Math]::Cos($newPhases[$i])
    $sumSin += [Math]::Sin($newPhases[$i])

    $deg = [int]($newPhases[$i] * 180.0 / [Math]::PI)
    $barLength = [Math]::Max(1, [int]($deg / 10))
    $bar = "█" * $barLength
    $pad = " " * (36 - $barLength)

    Write-Host ("{0,-16} | RTT: {1,4:F0} ms | Phase: {2,5:F1}° | [{3}{4}]" -f $hosts[$i], $latencies[$i], $deg, $bar, $pad) -ForegroundColor Green
}

$sw.Stop()

# 3. Compute Coherence Order Parameter R(t)
$R = [Math]::Sqrt(($sumCos * $sumCos) + ($sumSin * $sumSin)) / $N
$meanAngleRad = [Math]::Atan2($sumSin, $sumCos)
if ($meanAngleRad -lt 0) { $meanAngleRad += 2.0 * [Math]::PI }

$rPct = [int]($R * 100)
$gaugeFilled = "■" * [int]($rPct / 4)
$gaugeEmpty  = "░" * (25 - [int]($rPct / 4))
$color = if ($R -gt 0.7) { "Green" } elseif ($R -gt 0.4) { "Yellow" } else { "Red" }

Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor DarkCyan
Write-Host ("ORDER PARAMETER R(t) : {0,6:F4}  [{1}{2}] ({3}%)" -f $R, $gaugeFilled, $gaugeEmpty, $rPct) -ForegroundColor $color
Write-Host ("MEAN CLUSTER PHASE  : {0,6:F4} rad ({1,5:F1}°)" -f $meanAngleRad, ($meanAngleRad * 180.0 / [Math]::PI)) -ForegroundColor Cyan
Write-Host ("TOTAL SWEEP TIME    : {0:F3} ms" -f $sw.Elapsed.TotalMilliseconds) -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor DarkCyan

# 4. Automatic Git Commit
git add chronograph_engine.ps1
$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
git commit -m "feat(core): integrate order parameter R and phase matrix into chronograph engine [$timestamp]" --quiet
Write-Host "`n[✓] Script updated and committed to Git." -ForegroundColor Green

Pop-Location
