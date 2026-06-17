def register(runtime):
    runtime.register('llm://{host}/vision/query/analyze', 'python://urillm.handlers:vision_analyze', kind='query', operation='llm.vision.analyze')
    runtime.register('llm://{host}/text/query/plan', 'python://urillm.handlers:text_plan', kind='query', operation='llm.text.plan')
