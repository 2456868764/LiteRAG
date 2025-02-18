import hashlib
import os
from typing import (
    List,
    Callable,
    Generator,
    Dict,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
nltk.data.path = [os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")] \
                + nltk.data.path
print("nltk data path", nltk.data.path)

def md5_encryption(data):
    md5 = hashlib.md5()
    md5.update(data.encode('utf-8'))
    return md5.hexdigest()


def get_env_var(key, default=None, cast=str):
    """
    获取环境变量值，支持类型转换和默认值。
    """
    value = os.getenv(key, default)
    if value is not None and cast:
        try:
            if cast == bool:
                return value.lower() in ["true", "1", "yes"]
            return cast(value)
        except ValueError:
            raise ValueError(f"Environment variable {key} must be of type {cast.__name__}")
    return value


def validate_knowledge_name(name: str) -> bool:
    if "../" in name:
        return False
    return True


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    """
    在线程池中执行函数。

    本函数通过线程池异步执行多个函数调用，并在所有调用完成时返回结果。
    使用线程池可以有效地并行化处理多个任务，提高程序的执行效率。

    参数:
    - func: Callable, 需要在线程池中执行的函数。
    - params: List[Dict], 一个包含多个字典的列表，每个字典包含func所需的参数。
              默认为空列表，表示不传递任何参数。

    返回:
    - Generator, 一个生成器，按任务完成的顺序依次返回每个任务的结果。
    """
    # 初始化一个空的任务列表，用于存储所有提交到线程池的任务。
    tasks = []

    # 使用上下文管理器创建一个线程池。
    with ThreadPoolExecutor() as pool:
        # 遍历参数列表，每个参数是一个字典。
        for kwargs in params:
            # 在线程池中提交一个任务，使用提交的参数调用func函数。
            thread = pool.submit(func, **kwargs)
            # 将提交的任务添加到任务列表中。
            tasks.append(thread)

        # 使用as_completed函数监控任务列表中的任务完成情况。
        for obj in as_completed(tasks):
            # 当一个任务完成时，通过调用result方法获取任务的返回值，并通过yield返回。
            yield obj.result()
