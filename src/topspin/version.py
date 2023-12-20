__version__ = '1.2.0'

def display_TopSpin_version():
  from topspin.tools.helper import Logger
  Logger.info(f"TopSpin version={__version__}")
