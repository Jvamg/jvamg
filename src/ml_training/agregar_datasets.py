# agregar_datasets.py
# Uma ferramenta dedicada para consolidar múltiplos CSVs de padrões em um único dataset.

import pandas as pd
import glob  # Para encontrar arquivos usando padrões (wildcards)
import os    # Para manipular nomes de arquivos e caminhos

def agregar_e_enriquecer_datasets(pasta_onde_estao_os_csvs: str = '.', arquivo_de_saida: str = 'dataset_agregado_hns.csv'):
    """
    Encontra todos os CSVs de padrões, carrega-os, enriquece com metadados
    do nome do arquivo e os concatena em um único DataFrame.

    Args:
        pasta_onde_estao_os_csvs (str): O caminho para a pasta que contém os CSVs.
                                        O padrão é '.', que significa 'a pasta atual'.
        arquivo_de_saida (str): O nome do arquivo CSV consolidado que será gerado.
    """
    print(f"--- Iniciando Agregação de Datasets na pasta '{os.path.abspath(pasta_onde_estao_os_csvs)}' ---")

    # Passo 1: Encontrar todos os arquivos CSV que seguem o padrão de nome
    # O padrão 'padroes_hns_*.csv' usa o * como um curinga para encontrar qualquer nome entre os prefixo e o sufixo.
    padrao_de_busca = os.path.join(pasta_onde_estao_os_csvs, "padroes_hns_*.csv")
    lista_de_arquivos_csv = glob.glob(padrao_de_busca)

    if not lista_de_arquivos_csv:
        print("⚠️  Aviso: Nenhum arquivo CSV com o padrão 'padroes_hns_*.csv' foi encontrado.")
        print("Certifique-se de que este script está na mesma pasta que os seus 12 arquivos CSV.")
        return

    print(f"Encontrados {len(lista_de_arquivos_csv)} arquivos para agregar:")
    for f in lista_de_arquivos_csv:
        print(f"  - {os.path.basename(f)}")

    lista_de_dataframes = []
    # Passo 2: Ler cada arquivo e enriquecê-lo
    for arquivo in lista_de_arquivos_csv:
        try:
            # Esta é a "mágica": extrair informação valiosa do nome do arquivo
            nome_base = os.path.basename(arquivo)  # Ex: 'padroes_hns_BTC-USD_4h.csv'
            
            # Remove o prefixo e o sufixo para isolar as partes importantes
            partes_importantes = nome_base.replace('padroes_hns_', '').replace('.csv', '')
            
            # Divide o resto pelo '_' para separar ticker e intervalo
            ticker, intervalo = partes_importantes.split('_')
            
            # Carrega o CSV para um DataFrame do Pandas
            df_temp = pd.read_csv(arquivo)
            
            # Passo 3: Enriquece o DataFrame adicionando as colunas de contexto
            df_temp['ticker'] = ticker
            df_temp['intervalo'] = intervalo
            
            lista_de_dataframes.append(df_temp)
            print(f"  -> Processado e enriquecido: '{nome_base}'")

        except Exception as e:
            print(f"  -> ❌ Erro ao processar o arquivo '{arquivo}': {e}. Pulando este arquivo.")

    if not lista_de_dataframes:
        print("Nenhum DataFrame pôde ser carregado. A agregação foi cancelada.")
        return

    # Passo 4: Concatenar (juntar) todos os DataFrames da lista em um único DataFrame "mestre"
    print("\nConcatenando todos os DataFrames...")
    df_master = pd.concat(lista_de_dataframes, ignore_index=True)
    
    # Reorganiza as colunas para melhor visualização, colocando o contexto primeiro
    colunas_contexto = ['ticker', 'intervalo']
    colunas_dados = [col for col in df_master.columns if col not in colunas_contexto]
    df_master = df_master[colunas_contexto + colunas_dados]

    # Passo 5: Salvar o DataFrame mestre no arquivo de saída final
    df_master.to_csv(arquivo_de_saida, index=False, float_format='%.2f', date_format='%Y-%m-%d %H:%M:%S')
    
    print(f"\n🚀 Sucesso! {len(df_master)} registros de {len(lista_de_dataframes)} arquivos foram agregados em:")
    print(f"'{os.path.abspath(arquivo_de_saida)}'")


if __name__ == "__main__":
    # Simplesmente chama a função principal.
    agregar_e_enriquecer_datasets()