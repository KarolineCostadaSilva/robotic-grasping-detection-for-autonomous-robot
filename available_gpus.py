import subprocess

def get_available_gpus(mem_lim=1024):
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    
    Esta função verifica a memória disponível nas GPUs e retorna aquelas que têm mais memória livre do que o limite especificado.
    Parâmetros
    ----------
    mem_lim : int
        O limite mínimo de memória livre (em MB) para uma GPU ser considerada disponível.
    Retorna
    -------
    list
        Uma lista de strings representando os IDs das GPUs que têm mais memória livre do que o limite especificado.
    """

    # Executa o comando 'nvidia-smi' para consultar a memória livre das GPUs e retorna o resultado como uma string.
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free', #memory.used | # Consulta a quantidade de memória livre das GPUs.
            '--format=csv,nounits,noheader' # Formata a saída do comando como CSV sem unidades nem cabeçalho, para que seja mais fácil de ser processada.
        ], encoding='utf-8')
    # Convert lines into a dictionary
    # Transforma a string de resultado em uma lista de inteiros, representando a memória livre de cada GPU.
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    # Cria um dicionário onde as chaves são os IDs das GPUs (começando de 0) e os valores são a memória livre correspondente.
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    # Imprime a quantidade de memória livre disponível em cada GPU.
    print("GPUs memory available: {}".format(gpu_memory_map))
    # Retorna uma lista de strings representando os IDs das GPUs que têm mais memória livre do que o limite especificado (mem_lim).
    gpus_available = [str(i) for i in range(len(gpu_memory_map)) if gpu_memory_map[i] > mem_lim]
    # Imprime os IDs das GPUs que atendem ao critério de memória livre.
    print("GPUs memory available > {} MB: {}".format(mem_lim, gpus_available))
    # Retorna a lista de strings representando os IDs das GPUs que têm mais memória livre do que o limite especificado.
    return gpus_available
    
if __name__ == '__main__':
    # Chama a função definida acima e imprime as GPUs disponíveis com mais de 1024 MB de memória livre.
    gpus_available = get_available_gpus(mem_lim=1024)   # 1024 MB = 1 GB
    

