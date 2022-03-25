# -----------------------------
__file__ = 'subsys_messaging.py'
__doc__ = """
Contains the messaging subsystem that takes messages from other subsystems and handles their 
logging and distribution.
"""
# -----------------------------
import multiprocessing
import logging


def message_handler(log_queue: multiprocessing.Queue) -> None:
    """

    :param log_queue:
    :return:
    """
    # TODO: connect to the log database
    while True:
        new_message = log_queue.get(block=True, timeout=None)
        print("Received new message: {}".format(new_message))
        # SEND THE LOG MESSAGE INSTEAD OF PRINT


if __name__ == '__main__':
    print("NO CODE TO RUN")
