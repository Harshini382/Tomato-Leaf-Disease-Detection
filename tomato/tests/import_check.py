import sys, os
sys.path.insert(0, os.path.abspath('.'))
print('CWD:', os.getcwd())
try:
    from methods import method2
    print('imported method2 OK')
    print('has run_method2:', hasattr(method2, 'run_method2'))
except Exception as e:
    print('import error:', e)
    import traceback
    traceback.print_exc()
