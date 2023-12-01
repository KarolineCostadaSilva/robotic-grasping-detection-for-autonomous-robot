import sys
sys.path.append('C:\\Users\\kjcs\\Documents\\GitHub\\robotic-grasping-detection-for-autonomous-robot')

import unittest
from unittest.mock import patch
from available_gpus import get_available_gpus

class TestGetAvailableGpus(unittest.TestCase):

    @patch('subprocess.check_output')
    def test_no_gpus_available(self, mock_check_output):
        mock_check_output.return_value = "0\n0\n0"  # Simula uma saída onde todas as GPUs têm 0 MB livres
        self.assertEqual(get_available_gpus(1024), [])

    @patch('subprocess.check_output')
    def test_all_gpus_available(self, mock_check_output):
        mock_check_output.return_value = "2048\n3072\n4096"  # Simula uma saída onde todas as GPUs têm mais de 1024 MB livres
        self.assertEqual(get_available_gpus(1024), ['0', '1', '2'])

    @patch('subprocess.check_output')
    def test_some_gpus_available(self, mock_check_output):
        mock_check_output.return_value = "500\n1500\n2500"  # Simula uma saída onde apenas algumas GPUs têm mais de 1024 MB livres
        self.assertEqual(get_available_gpus(1024), ['1', '2'])

    @patch('subprocess.check_output')
    def test_memory_limit(self, mock_check_output):
        mock_check_output.return_value = "1025\n999\n1500"  # Simula uma saída com diferentes valores de memória livre
        self.assertEqual(get_available_gpus(1000), ['0', '2'])

# Esta linha permite executar os testes se o script for executado diretamente
if __name__ == '__main__':
    unittest.main()
