import json
import logging
import logging.config
import os
import platform
import time
from plotly.tools import set_credentials_file
import configparser

parser = configparser.ConfigParser()
# check if in notebook mode then set path for config files
def is_in_ipynb():
    '''
    Check wether the working environment is Jupyter Notebook / Ipython or not
    :return: Boolean (True or False)
    '''
    from ipykernel.zmqshell import ZMQInteractiveShell as Zshell
    try:
        if get_ipython():
            return True
        else:
            return False
    except NameError:
        return False

if is_in_ipynb():
    import os, sys
    FLAG_Ipynb = True
    module_path = os.path.abspath(os.path.join('./pda'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        # print(module_path + ' is set for notebook!')
    else:
        print(module_path + ' found in sys.path')
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
        # print(module_path + ' is set for notebook!')
    else:
        print(module_path + ' found in sys.path')
else:
    FLAG_Ipynb = False

if parser.read('config_general.ini')==[]:
    if FLAG_Ipynb:
        if parser.read('pda/config_general.ini') == []:
            raise ValueError ('could not find config_general file!')
    if  parser.read('pda/config_general.ini')!=[]:
        pass
    else:
        raise ValueError ('could not find config_general file!')


if platform.system() == 'Windows':
    FLAG_OS = 'win'
    model_path = parser['win']['model_path']
    data_path = parser['win']['data_path']
elif (platform.system() == 'Darwin'):
    FLAG_OS = 'mac'
    model_path = parser['mac']['model_path']
    data_path = parser['mac']['data_path']
if not os.path.exists(model_path):
    os.makedirs(model_path)
dataset_file = parser['default']['dataset_file']
from art import DATA_PATH
DATA_PATH = model_path

set_credentials_file(username=parser['default']['plotly_user'],
                     api_key=parser['default']['plotly_api_key'])

logger = logging.getLogger(__name__)
milisecond = int(round(time.time() * 1000))
logging.basicConfig(
    level=logging.INFO,
    filename="logfilename.log",
    format='%(asctime)s %(levelname)s %(message)s',
    handlers = [
        logging.FileHandler("{0}/{1}.log".format(model_path, 'logger_'+str(milisecond))),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)



