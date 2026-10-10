# 监控 Flickr2K.zip 下载完成 -> 自动解压到 D:/datasets/flickr2k  (Windows PowerShell 5.1 兼容)
$zip    = "E:\flickr2k\Flickr2K.zip"
$target = 11645004887
$log    = "E:\flickr2k\monitor.log"
$curlPid = 39564   # curl 进程 (断点续传, 同 PID 重试)

function Log($m) {
    "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m | Out-File $log -Append -Encoding utf8
}

Log "monitor start target=$target"

while ($true) {
    if (Test-Path $zip) { $size = (Get-Item $zip).Length } else { $size = 0 }
    $curlAlive = $null -ne (Get-Process -Id $curlPid -ErrorAction SilentlyContinue)
    Log "size=$size curlAlive=$curlAlive"

    if ($size -ge $target) {
        Log "DOWNLOAD_COMPLETE size=$size"
        break
    }
    if (-not $curlAlive -and $size -lt $target) {
        Log "CURL_GONE_INCOMPLETE size=$size (下载中断, 需断点续传)"
        exit 2
    }
    Start-Sleep -Seconds 60
}

Log "start extraction -> D:/datasets/flickr2k"
$env:PYTHONUTF8 = "1"
& "D:\Python\python.exe" "D:\tmp_timing\extract_flickr2k.py" --zip $zip --out "D:\datasets\flickr2k" > "E:\flickr2k\extract.log" 2>&1
Log "EXTRACT_EXIT=$LASTEXITCODE"
