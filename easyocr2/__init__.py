import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from easyocr2.model3 import Solution


if __name__ == '__main__':
    Solution().solution()