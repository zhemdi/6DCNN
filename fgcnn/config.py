'''
Copyright (C) 2021  Dmitrii Zhemchuzhnikov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import shutil

class load_config:
  def __init__(self):
    # Must be in the same folder as three_dl
    # loads the configuration from config file
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config"), 'r')
    config = f.read()
    f.close()

    config = config.split('\n')
    
    for line in config:
      if line != '' and line[0] != '#':
        [name,var] = line.split('=')
        name, var = name.replace(' ', ''), var.replace(' ', '')
        self.__dict__[name] = var
   
    # flushing the tensorboard repository
    folder = self.TENSORBOARD_PATH
    for the_file in os.listdir(folder):
      file_path = os.path.join(folder, the_file)
      try:
        if os.path.isfile(file_path):
          os.unlink(file_path)
      except Exception as e:
        print(e)
