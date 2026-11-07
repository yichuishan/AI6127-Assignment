"""One-call scripted transport for offline generation tests."""
import json


class ScriptedClient:
    def __init__(self, *outputs, usage=True):
        self.outputs,self.calls,self.usage=list(outputs),[],usage

    def complete(self,messages,model,temperature):
        if not self.outputs:
            raise AssertionError('Unexpected extra model call')
        self.calls.append(messages)
        return {'content':json.dumps(self.outputs.pop(0)),
                'usage':{'prompt_tokens':100,'completion_tokens':20} if self.usage else {}}
