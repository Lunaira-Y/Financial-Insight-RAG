import sys
import io
import contextlib

def safe_python_executor(code: str):
    """
    安全执行 Python 代码并返回 stdout 输出
    专门用于财务计算
    """
    # 简单的安全过滤，禁止危险模块
    forbidden_keywords = ["os.", "sys.", "subprocess", "open(", "eval(", "exec(", "requests", "import "]
    for kw in forbidden_keywords:
        if kw in code:
            return f"❌ 安全拦截：代码包含禁用关键字 {kw}"

    # 捕获输出
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            # 预置一个数学计算环境
            exec_globals = {"__builtins__": __builtins__, "math": __import__("math")}
            exec(code, exec_globals)
    except Exception as e:
        return f"❌ 代码执行出错: {str(e)}"
    
    return f.getvalue().strip()

if __name__ == "__main__":
    # 测试代码
    test_code = "print((803964958 / 777102455 - 1) * 100)"
    print(f"测试计算结果: {safe_python_executor(test_code)}%")
