param([ValidateSet('Backup','Protect','Release','Clean')][string]$Mode='Backup')
$ErrorActionPreference='Stop'
$repo='C:\生成式图像增强轻量模型设计挑战'
$archive='D:\project_archive_20261010'
$report=Join-Path $repo 'docs\handoff\CLEANUP_20261010'
$sourceRoot=Join-Path $report 'external_sources'
$api='https://api.github.com/repos/hxxdxwzys/Design-of-a-Lightweight-Generative-Image-Enhancement-Model'
$tag='project-archive-20261010'
$roots=[ordered]@{
    C_exports='C:\models\exports'; C_final_base='C:\models\finals'; D_models='D:\models';
    D_timing='D:\tmp_timing'; D_tmp='D:\tmp'; D_datasets='D:\datasets';
    C_submissions=Join-Path $repo '提交历史';
    C_workspace=Join-Path $repo '生成式图像增强轻量模型设计挑战\workspace'
}
function Files([string]$root) {
    $stack=[Collections.Generic.Stack[string]]::new();$stack.Push($root)
    while($stack.Count){
        $dir=$stack.Pop()
        foreach($item in Get-ChildItem -LiteralPath $dir -Force){
            if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){continue}
            if($item.PSIsContainer){
                if($item.Name -notin '.git','__pycache__','.venv','node_modules'){$stack.Push($item.FullName)}
            }else{$item}
        }
    }
}
function SaveJson($value,[string]$path) {
    $parent=Split-Path -Parent $path
    [IO.Directory]::CreateDirectory($parent)|Out-Null
    [IO.File]::WriteAllText($path,($value|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
}
function Sha([string]$path){(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
function SafeTarget([string]$path){
    $p=[IO.Path]::GetFullPath($path).TrimEnd('\')
    $allowed=@('C:\models\exports\','D:\models\','D:\datasets\','D:\tmp\','D:\tmp_timing\',($repo+'\提交历史\'))
    $ok=$false
    foreach($a in $allowed){if($p.StartsWith($a,[StringComparison]::OrdinalIgnoreCase)){$ok=$true}}
    if($p -in 'D:\tmp','D:\tmp_timing','D:\venv_faithdiff','D:\venv_osediff','D:\venv_tadsr'){$ok=$true}
    if($p -match '^D:\\tmp_(g1_case17|render_A|render_B|s1b|s1b2|v4)\.npy$'){$ok=$true}
    if(!$ok -or $p -in 'C:\','D:\',$repo,'C:\models','D:\models','D:\datasets'){throw "Unsafe deletion target: $p"}
    $protectedRoots=@('D:\models\upload','D:\models\vosr_lora','D:\models\final_compliant','D:\models\finals','D:\models\finals_20260927','D:\models\review_V3_20260926','D:\models\quality27_20260927','D:\models\v3_fidelity27_20260927','D:\datasets\div2k','D:\datasets\flickr2k')
    foreach($keep in $protectedRoots){
        if($p -eq $keep -or $p.StartsWith($keep+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Protected asset target: $p"}
    }
    if($p -in 'D:\models\tiny_v1','D:\models\refine_v1'){throw "Protected checkpoint root: $p"}
    $ancestor=$p
    while($ancestor -and $ancestor.Length -gt 3){
        if(Test-Path -LiteralPath $ancestor){
            if((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Reparse point in deletion path: $p"}
        }
        $ancestor=Split-Path -Parent $ancestor
    }
    return $p
}
function ApiHeaders {
    $inputText="protocol=https`nhost=github.com`n`n"
    $lines=@($inputText|git -c credential.interactive=never credential fill)
    $line=$lines|Where-Object {$_ -like 'password=*'}|Select-Object -First 1
    if(!$line){throw 'GitHub credentials unavailable'}
    return @{Authorization=('Bearer '+$line.Substring(9));'User-Agent'='Project-Archive';Accept='application/vnd.github+json'}
}

if($Mode -eq 'Backup'){
    if(Test-Path -LiteralPath (Join-Path $archive 'backup_ready.json')){throw 'Backup already frozen; use existing manifest.'}
    [IO.Directory]::CreateDirectory($archive)|Out-Null
    [IO.Directory]::CreateDirectory($sourceRoot)|Out-Null
    $before=@(Get-PSDrive C,D|Select-Object Name,Used,Free)
    SaveJson $before (Join-Path $report 'disk_before.json')
    $index=[Collections.Generic.List[object]]::new()
    $weights=[Collections.Generic.List[object]]::new()
    $smallWeightDir=Join-Path $archive 'small_weights\blobs'
    [IO.Directory]::CreateDirectory($smallWeightDir)|Out-Null
    $textExt=@('.py','.pyi','.json','.jsonl','.yaml','.yml','.toml','.cfg','.ini','.md','.txt','.log','.csv','.tsv','.sh','.bat','.ps1','.mjs','.js','.err','.out','.rst')
    $weightExt=@('.pt','.pth','.ckpt','.safetensors','.pkl','.onnx')
    $secretPattern='github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----|https?://[^\s/@:]+:[^\s/@]+@'
    foreach($label in $roots.Keys){
        $root=$roots[$label]
        if(!(Test-Path -LiteralPath $root)){continue}
        Write-Output "BACKUP_SCAN $label"
        foreach($f in Files $root){
            $rel=$f.FullName.Substring($root.TrimEnd('\').Length+1)
            $row=[ordered]@{root=$label;relative_path=$rel;bytes=$f.Length;mtime_utc=$f.LastWriteTimeUtc.ToString('o');sha256=$null;retention='metadata_only'}
            $isText=($f.Extension.ToLowerInvariant() -in $textExt -or $f.Name -match '^(LICENSE|COPYING|NOTICE|requirements)') -and $f.Length -le 8MB
            if($isText){
                $content=[IO.File]::ReadAllText($f.FullName)
                $row.sha256=Sha $f.FullName
                if($content -match $secretPattern -or $f.Name -match '(?i)^\.?(secrets?|credentials?|\.env)'){
                    $dest=Join-Path $archive ('private_sources\'+$label+'\'+$rel)
                    $row.retention='private_local_only'
                }else{
                    $dest=Join-Path $sourceRoot ($label+'\'+$rel)
                    $row.retention='github_text_snapshot'
                }
                [IO.Directory]::CreateDirectory((Split-Path -Parent $dest))|Out-Null
                Copy-Item -LiteralPath $f.FullName -Destination $dest
                if((Sha $dest) -ne $row.sha256){throw "Text copy mismatch: $label/$rel"}
            }elseif($f.Extension.ToLowerInvariant() -in $weightExt -and $f.Length -le 32MB){
                $row.sha256=Sha $f.FullName
                $blob=Join-Path $smallWeightDir ($row.sha256+$f.Extension.ToLowerInvariant())
                if(!(Test-Path -LiteralPath $blob)){Copy-Item -LiteralPath $f.FullName -Destination $blob}
                if((Sha $blob) -ne $row.sha256){throw 'Checkpoint copy mismatch'}
                $row.retention='github_release_small_weights'
                $weights.Add([PSCustomObject]@{source=$f.FullName;sha256=$row.sha256;bytes=$f.Length;blob=('blobs/'+[IO.Path]::GetFileName($blob))})
            }elseif($f.Extension -eq '.zip'){
                $row.sha256=Sha $f.FullName
            }
            $index.Add([PSCustomObject]$row)
        }
    }
    $venvs=@('D:\venv_faithdiff','D:\venv_osediff','D:\venv_tadsr')
    foreach($v in $venvs){
        if(!(Test-Path -LiteralPath $v)){continue}
        $label=Split-Path -Leaf $v
        $dir=Join-Path $sourceRoot ('environment\'+$label)
        [IO.Directory]::CreateDirectory($dir)|Out-Null
        Copy-Item -LiteralPath (Join-Path $v 'pyvenv.cfg') -Destination $dir
        $p=Join-Path $v 'Scripts\python.exe'
        $freeze=@(& $p -m pip freeze 2>&1|ForEach-Object {[string]$_})
        [IO.File]::WriteAllLines((Join-Path $dir 'pip-freeze.txt'),$freeze,[Text.UTF8Encoding]::new($false))
    }
    $mainFreeze=@(& D:\Python\python.exe -m pip freeze 2>&1|ForEach-Object {[string]$_})
    [IO.File]::WriteAllLines((Join-Path $sourceRoot 'environment\main-python-freeze.txt'),$mainFreeze,[Text.UTF8Encoding]::new($false))
    SaveJson $index.ToArray() (Join-Path $report 'file_inventory.json')
    SaveJson $weights.ToArray() (Join-Path $archive 'small_weights\checkpoint_index.json')
    Copy-Item -LiteralPath (Join-Path $archive 'small_weights\checkpoint_index.json') -Destination (Join-Path $report 'checkpoint_index.json')
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipIndex=[Collections.Generic.List[object]]::new()
    foreach($label in @('C_submissions','D_models')){
        foreach($r in $index|Where-Object {$_.root -eq $label -and $_.relative_path -like '*.zip'}){
            $path=Join-Path $roots[$label] $r.relative_path
            $z=[IO.Compression.ZipFile]::OpenRead($path)
            try{
                $members=@($z.Entries|ForEach-Object {[PSCustomObject]@{name=$_.FullName;bytes=$_.Length}})
                $zipIndex.Add([PSCustomObject]@{source=$path;bytes=$r.bytes;sha256=$r.sha256;members=$members})
            }finally{$z.Dispose()}
        }
    }
    SaveJson $zipIndex.ToArray() (Join-Path $report 'historical_zip_members.json')
    $smallZip=Join-Path $archive 'research-small-checkpoints.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory((Join-Path $archive 'small_weights'),$smallZip,[IO.Compression.CompressionLevel]::Optimal,$false)
    $a2Source='C:\models\exports\DELIV02_20260918_073810\Track2_A2_fp16_v1——42.2145.zip'
    $a2Dest=Join-Path $archive 'Track2_A2_fp16_v1_42.2145.zip'
    $expectedA2='49bd0e7e2df18cbad6e490998b76ad69031d43e33afc913843a912db424a1a67'
    if((Sha $a2Source) -ne $expectedA2){throw 'A2 original hash mismatch'}
    Copy-Item -LiteralPath $a2Source -Destination $a2Dest
    if((Sha $a2Dest) -ne $expectedA2){throw 'A2 backup hash mismatch'}
    $assets=@(
        @{name='Track2_TinyRefV3_original.zip';path='D:\models\upload\Track2_TinyRefV3.zip';expected='d3bf0bf602e9d2ec6459ace57753fd0188189aaacf7b1ca36fda5e1d590ca712'},
        @{name='Track2_TinyRefV4t1_original.zip';path='D:\models\upload\Track2_TinyRefV4t1.zip';expected='9502384034af5fb4a87d9c89eeedc191eb89008c52dbbef75fa854193cc94bad'},
        @{name='Track2_TinyRefV3_Review_Final.zip';path='D:\models\review_V3_20260926\Track2_TinyRefV3_Review_Final.zip';expected='b7d71e71bbf08439c2a66707286b1fb489c2817697bc990de5cc1fddac3b027b'},
        @{name='Track2_V4t1_EdgeFix_Review_20260927.zip';path='D:\models\quality27_20260927\Track2_V4t1_EdgeFix_Review_20260927.zip';expected='8a9c23ebf788494eb7b623a576038c8c6c9f22f7c269ac19c380e98c8124c58d'},
        @{name='research-small-checkpoints.zip';path=$smallZip;expected=$null}
    )
    foreach($a in $assets){
        $sha=Sha $a.path
        if($a.expected -and $sha -ne $a.expected){throw ('Artifact hash mismatch: '+$a.name)}
        $a.sha256=$sha;$a.bytes=(Get-Item -LiteralPath $a.path).Length
    }
    SaveJson $assets (Join-Path $archive 'release_assets.json')
    Copy-Item -LiteralPath (Join-Path $archive 'release_assets.json') -Destination (Join-Path $report 'release_assets.json')
    $plan=[Collections.Generic.List[object]]::new()
    $protected=@('upload','vosr_lora','final_compliant','finals','finals_20260927','review_V3_20260926','quality27_20260927','v3_fidelity27_20260927','tiny_v1','refine_v1')
    foreach($f in Get-ChildItem -LiteralPath 'D:\models' -Force){
        if($f.Attributes -band [IO.FileAttributes]::ReparsePoint){continue}
        if($f.PSIsContainer -and $f.Name -notin $protected){$plan.Add(@{path=SafeTarget $f.FullName;reason='historical experiment or downloadable model; sources and small checkpoints archived'})}
        elseif(!$f.PSIsContainer -and $f.Extension -eq '.zip'){$plan.Add(@{path=SafeTarget $f.FullName;reason='historical ZIP identity/member inventory retained; current originals protected'})}
    }
    foreach($r in @('C:\models\exports',(Join-Path $repo '提交历史'))){
        foreach($f in Get-ChildItem -LiteralPath $r -Force){if($f.Attributes -band [IO.FileAttributes]::ReparsePoint){continue};$plan.Add(@{path=SafeTarget $f.FullName;reason='external text evidence archived; A2 original ZIP preserved separately'})}
    }
    foreach($r in @('D:\models\tiny_v1','D:\models\refine_v1')){
        foreach($d in Get-ChildItem -LiteralPath $r -Directory){if($d.Attributes -band [IO.FileAttributes]::ReparsePoint){continue};$plan.Add(@{path=SafeTarget $d.FullName;reason='generated training/inference image cache; source checkpoints retained'})}
    }
    foreach($r in @('D:\tmp','D:\tmp_timing','D:\venv_faithdiff','D:\venv_osediff','D:\venv_tadsr')){if(Test-Path -LiteralPath $r){$plan.Add(@{path=SafeTarget $r;reason='archived external tools and obsolete experiment environments'})}}
    foreach($n in @('distill_hypir','distill256','holdout_v2','tiny_patches')){
        $p=Join-Path 'D:\datasets' $n
        if(Test-Path -LiteralPath $p){$plan.Add(@{path=SafeTarget $p;reason='derived training cache; raw DIV2K/Flickr2K retained'})}
    }
    foreach($n in @('tmp_g1_case17.npy','tmp_render_A.npy','tmp_render_B.npy','tmp_s1b.npy','tmp_s1b2.npy','tmp_v4.npy')){
        $p=Join-Path 'D:\' $n
        if(Test-Path -LiteralPath $p){$plan.Add(@{path=SafeTarget $p;reason='temporary inference array'})}
    }
    SaveJson $plan.ToArray() (Join-Path $report 'cleanup_plan.json')
    SaveJson @{created_utc=[DateTime]::UtcNow.ToString('o');files=$index.Count;small_checkpoint_paths=$weights.Count;archive=$archive;report=$report;a2_sha256=$expectedA2;release_assets=$assets.Count;targets=$plan.Count} (Join-Path $archive 'backup_ready.json')
    Copy-Item -LiteralPath (Join-Path $archive 'backup_ready.json') -Destination (Join-Path $report 'backup_ready.json')
    Write-Output ('BACKUP_READY files='+$index.Count+' checkpoints='+$weights.Count+' targets='+$plan.Count)
}

if($Mode -eq 'Protect'){
    $pattern='(?i)(Authorization\s*[:=]\s*[\x22\x27]?\s*(Bearer|Basic)\s+[A-Za-z0-9+/_=-]{16,}|[?&](token|access_token|signature|x-amz-signature|x-goog-signature|x-amz-credential)=[^&\s\x22\x27]{16,}|(password|passwd|api_key|api_token|access_token|secret_key)\s*[:=]\s*[\x22\x27][^\x22\x27\s]{12,}[\x22\x27])'
    $index=Get-Content -LiteralPath (Join-Path $report 'file_inventory.json') -Raw|ConvertFrom-Json
    $withheld=[Collections.Generic.List[object]]::new()
    foreach($f in Files $sourceRoot){
        if(!(Select-String -LiteralPath $f.FullName -Pattern $pattern -Quiet)){continue}
        $rel=$f.FullName.Substring($sourceRoot.Length+1)
        $dest=Join-Path $archive ('private_sources\'+$rel)
        $sha=Sha $f.FullName
        [IO.Directory]::CreateDirectory((Split-Path -Parent $dest))|Out-Null
        Copy-Item -LiteralPath $f.FullName -Destination $dest
        if((Sha $dest) -ne $sha){throw 'Private log copy mismatch'}
        $parts=$rel -split '\\',2
        foreach($row in $index){if($row.root -eq $parts[0] -and $row.relative_path -eq $parts[1]){$row.retention='private_local_only'}}
        if(!([IO.Path]::GetFullPath($f.FullName)).StartsWith($sourceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Snapshot path escaped'}
        Remove-Item -LiteralPath $f.FullName
        $withheld.Add(@{path=$rel;sha256=$sha;reason='credential-like value or signed URL; kept only in local private_sources'})
    }
    SaveJson $index (Join-Path $report 'file_inventory.json')
    SaveJson $withheld.ToArray() (Join-Path $report 'additional_sensitive_withheld.json')
    Write-Output ('SNAPSHOT_PROTECTED withheld='+$withheld.Count)
}

if($Mode -eq 'Release'){
    if(!(Test-Path -LiteralPath (Join-Path $archive 'backup_ready.json'))){throw 'Backup not ready'}
    $headers=ApiHeaders
    $all=@(Invoke-RestMethod -Uri ($api+'/releases?per_page=100') -Headers $headers)
    $release=$all|Where-Object tag_name -EQ $tag|Select-Object -First 1
    if(!$release){
        $payload=@{tag_name=$tag;target_commitish=(git -C $repo rev-parse HEAD);name='Project preservation archive 2026-10-10';draft=$false;prerelease=$false;body='Preservation of original V3/V4t1 packages, review materials, and historical small research checkpoints. Research checkpoints are not approved submissions. Scores and review status are documented in the repository. This archive is not a new competition submission.'}|ConvertTo-Json
        $release=Invoke-RestMethod -Method Post -Uri ($api+'/releases') -Headers $headers -Body $payload -ContentType 'application/json'
    }
    $receipts=[Collections.Generic.List[object]]::new()
    foreach($a in Get-Content -LiteralPath (Join-Path $archive 'release_assets.json') -Raw|ConvertFrom-Json){
        if((Sha $a.path) -ne $a.sha256){throw 'Asset changed after backup'}
        $existing=@(Invoke-RestMethod -Uri $release.assets_url -Headers $headers)|Where-Object name -EQ $a.name|Select-Object -First 1
        if($existing){$asset=$existing}else{
            Write-Output ('UPLOAD_START '+$a.name+' bytes='+$a.bytes)
            $upload=($release.upload_url -replace '\{.*$','')+'?name='+[Uri]::EscapeDataString($a.name)
            $asset=Invoke-RestMethod -Method Post -Uri $upload -Headers $headers -InFile $a.path -ContentType 'application/zip' -TimeoutSec 3600
        }
        if($asset.size -ne $a.bytes -or $asset.state -ne 'uploaded'){throw 'Remote asset size/state mismatch'}
        if(!$asset.digest -or $asset.digest -ne ('sha256:'+$a.sha256)){throw 'Remote SHA-256 digest missing or mismatched'}
        $receipts.Add(@{name=$a.name;id=$asset.id;bytes=$asset.size;sha256=$a.sha256;url=$asset.browser_download_url;remote_digest=$asset.digest;verified=$true})
        SaveJson @{release_url=$release.html_url;assets=$receipts.ToArray()} (Join-Path $report 'release_upload_receipt.json')
        Write-Output ('UPLOAD_VERIFIED '+$a.name)
    }
    Copy-Item -LiteralPath (Join-Path $report 'release_upload_receipt.json') -Destination (Join-Path $archive 'release_upload_receipt.json')
    Write-Output ('RELEASE_READY '+$release.html_url)
}

if($Mode -eq 'Clean'){
    if(!(Test-Path -LiteralPath (Join-Path $archive 'release_upload_receipt.json'))){throw 'Verified release missing'}
    $receipt=Get-Content -LiteralPath (Join-Path $archive 'release_upload_receipt.json') -Raw|ConvertFrom-Json
    if(@($receipt.assets|Where-Object {!$_.verified}).Count -or $receipt.assets.Count -ne 5){throw 'Release incomplete'}
    $local=git -C $repo rev-parse HEAD
    $remote=((git -C $repo ls-remote --heads origin main) -split '\s+')[0]
    if($local -ne $remote){throw 'Repository not synchronized before cleanup'}
    if((Sha (Join-Path $archive 'Track2_A2_fp16_v1_42.2145.zip')) -ne '49bd0e7e2df18cbad6e490998b76ad69031d43e33afc913843a912db424a1a67'){throw 'A2 backup missing'}
    $plan=Get-Content -LiteralPath (Join-Path $report 'cleanup_plan.json') -Raw|ConvertFrom-Json
    $records=[Collections.Generic.List[object]]::new()
    foreach($entry in $plan){
        $p=SafeTarget $entry.path
        if(!(Test-Path -LiteralPath $p)){continue}
        $item=Get-Item -LiteralPath $p -Force
        $bytes=if($item.PSIsContainer){(Get-ChildItem -LiteralPath $p -Recurse -Force -File|Where-Object {!($_.Attributes -band [IO.FileAttributes]::ReparsePoint)}|Measure-Object Length -Sum).Sum}else{$item.Length}
        $descendantLinks=if($item.PSIsContainer){@(Get-ChildItem -LiteralPath $p -Recurse -Force|Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint})}else{@()}
        if($descendantLinks.Count){
            $records.Add(@{path=$p;status='SKIPPED_REPARSE_POINT';bytes=$bytes;reason=$entry.reason})
            Write-Output ('SKIPPED_LINK '+$p)
            continue
        }
        Remove-Item -LiteralPath $p -Recurse -Force
        $records.Add(@{path=$p;status='DELETED';bytes=$bytes;reason=$entry.reason;finished_utc=[DateTime]::UtcNow.ToString('o')})
        SaveJson $records.ToArray() (Join-Path $report 'cleanup_execution.json')
        Write-Output ('DELETED '+$p+' logical_MiB='+[math]::Round($bytes/1MB,1))
    }
    SaveJson @(Get-PSDrive C,D|Select-Object Name,Used,Free) (Join-Path $report 'disk_after.json')
    Copy-Item -LiteralPath (Join-Path $report 'cleanup_execution.json') -Destination $archive
    Copy-Item -LiteralPath (Join-Path $report 'disk_after.json') -Destination $archive
    Write-Output 'CLEANUP_COMPLETE'
}
