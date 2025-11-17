# flake8: noqa
import importlib.metadata
__version__ = importlib.metadata.version("graph_jsp_env")

VERSION = __version__

CRWTH_BLUE = '\033[38;2;34;83;154m'
CRWTH_BLUE_GB = '\033[48;2;34;83;154m'
CRWTH_LIGHT_BLUE = '\033[38;2;151;185;255m'

CIMA_BLUE = '\033[38;2;68;153;162m'
CIMA_BLUE_BG = '\033[48;2;68;153;162m'
B = '\033[38;2;68;153;162m'

CIMA_WHITE = '\033[38;2;255;255;255m'
CIMA_WHITE_BG = '\033[48;2;255;255;255m'
W = '\033[38;2;255;255;255m'

CEND = '\33[0m'
CBOLD = '\33[1m'
CITALIC = '\33[3m'
CURL = '\33[4m'
CBLINK = '\33[5m'
CBLINK2 = '\33[6m'
CSELECTED = '\33[7m'

CBLACK = '\33[30m'
CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE = '\33[36m'
CWHITE = '\33[37m'

CBLACKBG = '\33[40m'
CREDBG = '\33[41m'
CGREENBG = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG = '\33[46m'
CWHITEBG = '\33[47m'

CGREY = '\33[90m'
CRED2 = '\33[91m'
CGREEN2 = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2 = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2 = '\33[96m'
CWHITE2 = '\33[97m'

CGREYBG = '\33[100m'
CREDBG2 = '\33[101m'
CGREENBG2 = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2 = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2 = '\33[106m'
CWHITEBG2 = '\33[107m'


# noqa: E501, W291
big_banner = f"""          

     {CEND}{CIMA_BLUE_BG}                      {CEND}{CRWTH_BLUE}                                                 ┃                              {CRWTH_LIGHT_BLUE}                                {CBLUE}
     {CEND}{CIMA_BLUE_BG}                      {CEND}{CRWTH_BLUE}     ▐███▌         ▟███▛▟███████▛▐███▌           ┃       ▐█▀▜▙█▙   ███████▐█ ▐█ {CRWTH_LIGHT_BLUE}   ▟█▙   ▟█▙  ▟███▐█ ▐█▐█▀▀▐█▙ █{CBLUE}
     {CEND}{CIMA_BLUE_BG}                      {CEND}{CRWTH_BLUE}     ▐███▌        ▟███▛▟███████▛ ▐███▌           ┃       ▐█▄▟▛▜█▙▟▙██ ▐█  ▐████ {CRWTH_LIGHT_BLUE}  ▟▛ ▜▙ ▟▛ ▜▙ █▍  ▐████▐█▀▀▐██▙█{CBLUE}
     {CEND}{CIMA_BLUE_BG}  ▐███▐██▙▟██  ▟█▙    {CEND}{CRWTH_BLUE}     ▐███▌ ▟███  ▟███▛    ▟███▛  ▐███▌           ┃       ▐█ ▜▙ ▜█▛▜██ ▐█  ▐█ ▐█ {CRWTH_LIGHT_BLUE} ▟█▛▀▜█▙█▛▀▜█▙▜███▐█ ▐█▐█▆▆▐█ ▜█{CBLUE}
     {CEND}{CIMA_BLUE_BG}   ▐█ ▐█ ▜▛▐█ ▟█▀█▙   {CEND}{CRWTH_BLUE}     ▐███▌▟████ ▟███▛    ▟███▛   ▐███▌           ┃             {CRWTH_LIGHT_BLUE}▐█  ▐█▐█▙ █▐███▜█▙ ▟███▀▀▐█▀▜▙▟█▀▜█▐███▐█████▙ ▟▛
     {CEND}{CIMA_BLUE_BG}  ▐███▐█   ▐█▟█▛▀▀█▙  {CEND}{CRWTH_BLUE}     ▐██████▛▐█████▛    ▟████████▐█████████▛     ┃             {CRWTH_LIGHT_BLUE}▐█  ▐█▐██▙█ ▐█  ▜█▄█▛▐█▀▀▐█▄▟▛▜█▆▆▄ ▐█  ▐█  ▜█▄▛ 
     {CEND}{CIMA_BLUE_BG}                      {CEND}{CRWTH_BLUE}     ▐█████▛ ▐████▛    ▟█████████▐████████▛      ┃             {CRWTH_LIGHT_BLUE} ▜███▛▐█ ▜█▐███  ▜█▛ ▐███▐█ ▜▙▐█▆▆▛▐███ ▐█   ██  

{CEND}
    {CBLUE}{CBOLD}
    Disjunctive Graph Job Shop Problem Environment
    {CEND}     

    Version:    {CGREEN}{VERSION}{CEND}

    Developer:  {CBLUE}{CBOLD}Alexander Nasuta{CEND}

    E-mail:     {CBLUE}{CBOLD}alexander.nasuta@ima.rwth-aachen.de{CEND},
                {CGREY}alexander@nasuta.eu{CEND}
                
"""

small_banner = f"""          

     {B}{CIMA_BLUE_BG}0100100100100000001111{CEND}{CRWTH_BLUE}                                            
     {B}{CIMA_BLUE_BG}0000110011001000000100{CEND}{CRWTH_BLUE}     ▐███▌         ▟███▛▟███████▛▐███▌      
     {B}{CIMA_BLUE_BG}1111011011000111100101{CEND}{CRWTH_BLUE}     ▐███▌        ▟███▛▟███████▛ ▐███▌      
     {B}{CIMA_BLUE_BG}10{W}▐███▐██▙▟██{B}00{W}▟█▙{B}0100{CEND}{CRWTH_BLUE}     ▐███▌ ▟███  ▟███▛    ▟███▛  ▐███▌             
     {B}{CIMA_BLUE_BG}101{W}▐█{B}1{W}▐█{B}1{W}▜▛▐█{B}0{W}▟█▀█▙{B}001{CEND}{CRWTH_BLUE}     ▐███▌▟████ ▟███▛    ▟███▛   ▐███▌
     {B}{CIMA_BLUE_BG}00{W}▐███▐█{B}000{W}▐█▟█▛▀▀█▙{B}01{CEND}{CRWTH_BLUE}     ▐██████▛▐█████▛    ▟████████▐█████████▛ 
     {B}{CIMA_BLUE_BG}0010010010000000111100{CEND}{CRWTH_BLUE}     ▐█████▛ ▐████▛    ▟█████████▐████████▛ 


     {CRWTH_BLUE}▐█▀▜▙█▙   ███████▐█ ▐█ {CRWTH_LIGHT_BLUE}   ▟█▙   ▟█▙  ▟███▐█ ▐█▐█▀▀▐█▙ █{CBLUE}
     {CRWTH_BLUE}▐█▄▟▛▜█▙▟▙██ ▐█  ▐████ {CRWTH_LIGHT_BLUE}  ▟▛ ▜▙ ▟▛ ▜▙ █▍  ▐████▐█▀▀▐██▙█{CBLUE}
     {CRWTH_BLUE}▐█ ▜▙ ▜█▛▜██ ▐█  ▐█ ▐█ {CRWTH_LIGHT_BLUE} ▟█▛▀▜█▙█▛▀▜█▙▜███▐█ ▐█▐█▆▆▐█ ▜█{CBLUE}
           {CRWTH_LIGHT_BLUE}▐█  ▐█▐█▙ █▐███▜█▙ ▟███▀▀▐█▀▜▙▟█▀▜█▐███▐█████▙ ▟▛
           {CRWTH_LIGHT_BLUE}▐█  ▐█▐██▙█ ▐█  ▜█▄█▛▐█▀▀▐█▄▟▛▜█▆▆▄ ▐█  ▐█  ▜█▄▛ 
           {CRWTH_LIGHT_BLUE} ▜███▛▐█ ▜█▐███  ▜█▛ ▐███▐█ ▜▙▐█▆▆▛▐███ ▐█   ██  
{CEND}
    {CBLUE}{CBOLD}
    Disjunctive Graph Job Shop Problem Environment
    {CEND}     

    Version:    {CGREEN}{VERSION}{CEND}

    Developer:  {CBLUE}{CBOLD}Alexander Nasuta{CEND}

    E-mail:     {CBLUE}{CBOLD}alexander.nasuta@ima.rwth-aachen.de{CEND},
                {CGREY}alexander@nasuta.eu{CEND}

"""
