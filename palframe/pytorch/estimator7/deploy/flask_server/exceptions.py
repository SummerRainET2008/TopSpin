"""
各种异常汇总
"""


class ExampleListNoneException(Exception):
    def __str__(self):
        return 'example list is None'


class ExampleProcessError(Exception):
    """
    example 处理出现错误
    """
    def __init__(self, example, example_index,processor,error_info):
        """

        :param example: 出错当前的example
        :param example_index: example 的索引，
        :param processor: 出错位置的processor
        :param error_info: 错误信息
        """
        self.example = example
        self.processor = processor
        self.example_index = example_index
        self.error_info = error_info
        self.error_traceback = []  # [(),(),..]

    def format_traceback_info(self):
        traceback_info = []
        if self.error_traceback:
            for item_name,item_value in self.error_traceback:
                traceback_info.append(f'{item_name}:\n {item_value}')
        join_char = '\n' + '-*'*50 + '\n'
        return join_char.join(traceback_info)

    def __str__(self):
        return f'example 在processor: {self.processor} 中计算出错，' \
               f'错误信息为: \n {self.error_info}\n' \
               f'example 处理的回溯信息为:\n {self.format_traceback_info()}'



class ModelStartError(Exception):
    def __init__(self, model_id, error_info):
        self.model_id = model_id
        self.error_info = error_info
    def __str__(self):
        return f'模型: {self.model_id} 启动失败, 错误信息为: {self.error_info}'



if __name__ == '__main__':
    pass
