from functools import wraps
from contextlib import contextmanager

from comps import CustomLogger
from rag.connector.database.base import SessionLocal
from sqlalchemy.orm import Session

logger = CustomLogger(__name__)

@contextmanager
def transaction_scope() -> Session:
    """
    创建一个事务范围的上下文管理器。

    该函数的作用是为数据库操作提供一个事务性的上下文环境。在进入上下文时，
    自动开始一个新的数据库会话，并在退出上下文时根据情况提交或回滚事务。

    使用该函数的主要目的是简化数据库事务管理，确保在发生异常时能够自动回滚，
    并在操作完成后自动关闭数据库会话，防止资源泄露。

    Yields:
        Session: 一个数据库会话对象，用于执行数据库操作。
    """
    # 创建一个数据库会话实例
    session = SessionLocal()
    try:
        # 使用yield语法将控制权交还给调用者，调用者可以使用session执行数据库操作
        yield session
        # 如果在上下文中的代码执行成功，提交事务
        session.commit()
    except Exception as e:
        # 如果在上下文中发生异常，回滚事务
        session.rollback()
        # 记录错误日志，包含异常信息和堆栈跟踪
        logger.error(f"Database session error: {e}")
        # 将异常重新抛出，以便调用者可以捕获和处理
        raise
    finally:
        # 无论是否发生异常，都关闭数据库会话
        session.close()



def with_session(f):
    """
    为函数调用自动管理数据库会话的装饰器。
    该装饰器负责在函数调用前创建一个数据库会话，并在函数执行成功后提交会话。
    若函数执行过程中抛出异常，会自动回滚会话。

    参数:
    f: 被装饰的函数或方法。

    返回:
    wrapper: 包装后的函数，自动管理会话。
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        """
        会话管理的包装函数。
        该函数确保在被装饰的函数执行时自动开启一个数据库会话，
        并根据执行情况提交或回滚事务。

        参数:
        *args: 位置参数，允许接受不定数量的参数。
        **kwargs: 关键字参数，允许接受不定数量的参数。

        返回:
        result: 被装饰函数的执行结果。
        """
        # 使用上下文管理器创建一个事务会话
        with transaction_scope() as session:
            try:
                # 检测是否是类方法，传递 self 参数
                if hasattr(f, "__self__") or (len(args) > 0 and hasattr(args[0], "__dict__")):
                    result = f(args[0], session, *args[1:], **kwargs)
                else:
                    result = f(session, *args, **kwargs)
                # 提交会话
                session.commit()
                return result
            except Exception as e:
                # 回滚会话
                session.rollback()
                # 记录错误日志
                logger.error(f"Error in function {f.__name__}: {e}")
                # 重新抛出异常
                raise

    return wrapper

