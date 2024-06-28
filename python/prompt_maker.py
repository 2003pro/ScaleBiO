from typing import Dict

class PromptMaker:

    def get_input(self, data_point, **kargs) -> str:
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""

    def get_target(self, data_point, **kargs) -> str:
        target = data_point["output"]
        return target
    
    def get_full(self, data_point, **kargs) -> str:
        text = self.get_input(data_point) + self.get_target(data_point)
        return text