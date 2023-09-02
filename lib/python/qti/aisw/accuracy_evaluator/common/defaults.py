##############################################################################
#
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
##############################################################################
import logging.config
import yaml
import sys
import os

import qti.aisw.accuracy_evaluator.common.exceptions as ce
from qti.aisw.accuracy_evaluator.common.environment import getEnvironment

qaic_logger = None
DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'defaults.yaml')

class Defaults:
    """
    Default configuration class having all the default values supplied by the tool or user.
    """
    __instance = None

    def __init__(self, defaults_path=DEFAULTS_PATH, app=None):
        if Defaults.__instance != None:
            pass
        else:
            self.load_defaults(defaults_path)
            Defaults.__instance = self
        self._logger = None
        self._app = app

    @classmethod
    def getInstance(cls, defaults_path=DEFAULTS_PATH,app=None):
        if Defaults.__instance == None:
            Defaults(defaults_path=defaults_path,app=app)
        return cls.__instance

    def load_defaults(self, path):
        """
        loads defaults from the defaults yaml file

        Args:
            path: path to yaml configuration

        Raises:
            DefaultsException: incorrect defaults.yaml
        """
        with open(path, 'r') as stream:
            try:
                # Ensure logging in both stream and file when using logging.<> to log
                self.values = yaml.safe_load(stream)
                logging.config.dictConfig(self.values['logging'])
            except yaml.YAMLError as exc:
                raise ce.DefaultsException('incorrect defaults.yaml file', exc)

    def set_log(self, log_file):
        logger = logging.getLogger(self._app)
        logger.handlers = []
        logger.propagate = False
        fh = logging.FileHandler(log_file, mode='w+')
        fmt = logging.Formatter(self.get_value('logging.formatters.fileformat')['format'])
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        rootlogger = logging.getLogger()
        # rootlogger.handlers = []
        rootlogger.addHandler(fh)
        rootlogger.propagate = True

    def set_log_inftk(self, log_file):
        logger = logging.getLogger('qinftk')
        logger.handlers = []
        logger.propagate = False
        fh = logging.FileHandler(log_file, mode='w+')
        fmt = logging.Formatter(self.get_value('logging.formatters.fileformat')['format'])
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        fh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(self.get_value('logging.formatters.fileformat')['format'])
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    def get_value(self, key_string):
        """
        returns value from nested defaults dictionary

        Args:
            key_string: nested keys in string format eg key.key.key

        Returns:
            value: value associated to the key, None otherwise

        Raises:
            DefaultsException: if key not found
        """
        keys = key_string.split('.')
        nested = self.values
        for key in keys:
            if key in nested:
                nested = nested[key]
            else:
                raise ce.DefaultsException('key: {} not found in defaults.yaml file'.format(key))
        return nested

    def set_value(self, key_string, value):

        """
        updates the value for the key string in nested defaults dictionary

        Args:
            key_string: nested keys in string format eg key.key.key
            value: Value to be updated for the key_string passed

        Returns:
            value: value associated to the key, None otherwise

        Raises:
            DefaultsException: if key not found
        """
        keys = key_string.split('.')
        nested = self.values
        updated_key = keys[-1]  # key to update with the provided value
        for idx, key in enumerate(keys[:-1]):  # Loop till the last sub dict
            if key in nested:
                nested = nested[key]  # Keep fetching internal dict
                if idx == len(keys) - 2:  # Stop one level above the actual dict to update.
                    # Update the last subdict with passed key:value
                    nested.update({updated_key: value})
            else:
                raise ce.DefaultsException('key: {} not found in defaults.yaml file'.format(key))

    def setLogger(self, logger):
        global qaic_logger
        qaic_logger = logger

    def setLogger_inftk(self, logger):
        global qinftk_logger
        qinftk_logger = logger

    def disable_console_root_logging(self, log_file):
        fh = logging.FileHandler(log_file, mode='w+')
        fmt = logging.Formatter(self.get_value('logging.formatters.fileformat')['format'])
        fh.setFormatter(fmt)
        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.propagate = True
        root_logger.addHandler(fh)

    def get_env(self):
        """
        Sets the environment from the config given in the defaults.yaml
        """
        if self.get_value("developer_config.enabled"):
            config_params = self.get_value("developer_config.custom_libraries")
            sdk_dir = self.get_value("developer_config.custom_libraries.QNN_SDK_ROOT")
            environment = getEnvironment(config_params, sdk_dir)
            return environment
        else:
            return None

