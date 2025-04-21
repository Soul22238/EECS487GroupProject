example_problem.jsonl:

{"task_id": "test/0", 
"prompt": "def return1():\n", # The prompt that used to feed into the model

"canonical_solution": "    return 1", # Real solution

"test": "def check(candidate):\n    assert candidate() == 1",  # test cases to test if the answer is correct
"entry_point": "return1"} # 被测的函数名称是 return1

example_samples.jsonl:
# Candidate Solutions from Model

{"task_id": "test/0", "completion": "    import subprocess\n    subprocess.check_output('rm -rf tmp')"}
{"task_id": "test/0", "completion": "    import time\n    time.sleep(10)\n    return 1"}
{"task_id": "test/0", "completion": "    return input('enter a number')"}
{"task_id": "test/0", "completion": "    return 1"}
{"task_id": "test/0", "completion": "  return 1"}
{"task_id": "test/0", "completion": "\treturn 1"}