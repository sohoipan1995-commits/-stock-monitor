KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError
The above exception was the direct cause of the following exception:
File "/mount/src/-stock-monitor/app.py", line 982, in <module>
    f"{(df_all['cycle_status'].str.startswith(('🟢', '🟡'))).sum()} 隻",
        ~~~~~~^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
File "/home/adminuser/venv/lib/python3.14/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
