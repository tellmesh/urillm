def register(runtime):
    runtime.register('llm://{host}/vision/query/analyze', 'python://urillm.handlers:vision_analyze', kind='query', operation='llm.vision.analyze')
