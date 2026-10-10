@echo off
rem 一键恢复 1.4B 自动验证链 (2026-09-05)
rem 用法: 开机后双击运行; watchdog 会自动拉起 autoval (120s 内)
echo [1/2] 启动 watchdog (p100_watch_local.py) ...
start "watchdog" /min D:\Python\python.exe D:\tmp_timing\p100_watch_local.py
echo [2/2] 等待 watchdog 拉起 autoval ...
timeout /t 5 /nobreak >nul
echo 完成。验证: 心跳文件 D:\tmp_timing\p100_val\autoval14.hb 应在 ~2 分钟内更新;
echo       历史 D:\tmp_timing\p100_val\autoval14_history.txt 会显示 [boot] 新行。
echo 若 5 分钟后无 [boot], 手动运行: D:\Python\python.exe D:\tmp_timing\p100_autoval_14b.py
pause
