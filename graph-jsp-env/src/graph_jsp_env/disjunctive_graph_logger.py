import logging
import shutil

from rich.logging import RichHandler

from graph_jsp_env.wzl_ima_banner import big_banner, small_banner

# print banner when logger is imported
w, h = shutil.get_terminal_size((80, 20))

# print(small_banner if w < 140 else big_banner)              #关闭没有必要的开头图标和版本号，虽然我知道是借鉴了他的方法！！！！！！！！！

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False)]
)

# create file handler which logs messages
# fh = logging.FileHandler(PATHS.LOGS_FILE_PATH)
# fh.setLevel(logging.INFO)

log = logging.getLogger("rich")
# log.addHandler(fh)
