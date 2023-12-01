import sys
sys.path.append('C:\\Users\\kjcs\\Documents\\GitHub\\robotic-grasping-detection-for-autonomous-robot')

import unittest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
# Importe seus módulos e funções aqui
from available_gpus import get_available_gpus
from grasp_dataset import GraspDataset
from network import GraspNet
from demo import load_model
from demo import vis_detections
from demo import Rotate2D

class TestGraspDetectionSystem(unittest.TestCase):

    @patch('subprocess.check_output')
    @patch('available_gpus.get_available_gpus')
    def test_get_available_gpus(self, mock_check_output):
        # Teste a função get_available_gpus com diferentes saídas simuladas
        # Simular a saída do comando 'nvidia-smi'
        mock_check_output.return_value = "4096\n2048\n1024"
        expected = ['0', '1']
        result = get_available_gpus(mem_lim=1024)
        self.assertEqual(result, expected)

    # @patch('torch.load')
    # @patch('network.GraspNet')
    # def test_load_model(self, mock_graspnet, mock_torch_load):
    #     # Simula a carga do modelo com pesos que estão no mesmo dispositivo que os dados de entrada
    #     # Teste a função load_model com diferentes números de GPUs disponíveis
    #     mock_model = MagicMock()
    #     mock_model.to = MagicMock(return_value=mock_model)  # Garante que o método to() retorne o próprio mock
    #     mock_torch_load.return_value = {'model': mock_model}

    #     gpu_ids = ['0']
    #     device = torch.device("cuda:0" if gpu_ids else "cpu")
    #     img_tensor = torch.randn(1, 3, 224, 224).to(device)
        
    #     model = load_model(gpu_ids)
    #     model(img_tensor)  # Testa a passagem de um tensor pelo modelo

    #     mock_graspnet.assert_called_once()
    #     mock_model.to.assert_called_with(device)  # Verifica se o modelo foi movido para o dispositivo correto

    # def test_image_processing(self):
    #     # Teste as transformações e manipulações de imagem
    #     pass
    
    @patch('demo.rotate2d')
    def test_rotate2d(self):
        pts = np.array([[1, 0], [0, 1]])
        cnt = np.array([0, 0])
        ang = np.pi / 2  # 90 degrees
        expected = np.array([[0, -1], [1, 0]])
        result = Rotate2D(pts, cnt, ang)
        np.testing.assert_almost_equal(result, expected)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.imshow')
    @patch('demo.vis_detections')
    def test_visualization(self, mock_imshow, mock_subplots):
        # Teste a função de visualização (plot)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        im = MagicMock()
        score = 1
        dets = [0, 0, 10, 10]
        vis_detections(mock_ax, im, score, dets)
        mock_imshow.assert_called_with(im, aspect='equal')

# Esta linha permite executar os testes se o script for executado diretamente
if __name__ == '__main__':
    unittest.main()
