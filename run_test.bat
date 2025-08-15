@echo off
chcp 65001 > nul

cd /d "d:\Cursor\多Agent智能分流 + 长程记忆优化方案"

echo 运行简化测试脚本...
echo =========================

python demos/simple_test.py

if %errorlevel% equ 0 (
    echo.
    echo 测试完成！
) else (
    echo.
    echo 测试失败，错误代码: %errorlevel%
)

echo.
pause