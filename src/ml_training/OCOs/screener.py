import sys
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import init, Fore, Style

# Importa a lógica refatorada e as constantes
from scan_OCO import analisar_ativo_para_screener, MODELO_PATH, C_GREEN, C_YELLOW, C_BOLD, LIMIAR_CONFIANCA


def carregar_tickers(caminho_arquivo="src/ml_training/OCOs/tickers.txt"):
    """Lê uma lista de tickers de um arquivo de texto."""
    try:
        with open(caminho_arquivo, 'r') as f:
            # Remove linhas em branco e espaços
            tickers = [line.strip() for line in f if line.strip()]
        if not tickers:
            print(
                f"{C_YELLOW}Atenção: O arquivo '{caminho_arquivo}' está vazio ou não contém tickers válidos.")
            return []
        return tickers
    except FileNotFoundError:
        print(f"{Fore.RED}ERRO: Arquivo de tickers '{caminho_arquivo}' não encontrado.")
        return None


def main():
    """
    Ponto de entrada principal do screener.
    """
    init(autoreset=True)

    # 1. Carregar Ativos de Machine Learning (UMA ÚNICA VEZ)
    print("Carregando modelo de ML...")
    try:
        artefatos = joblib.load(MODELO_PATH)
        model = artefatos['model']
        scaler = artefatos['scaler']
        expected_features = artefatos['features']
        print(f"{C_GREEN}✔ Modelo carregado com sucesso.")
    except Exception as e:
        print(f"{Fore.RED}ERRO CRÍTICO ao carregar o modelo: {e}")
        sys.exit(1)

    # 2. Carregar Lista de Tickers
    tickers_para_analisar = carregar_tickers()
    if tickers_para_analisar is None or not tickers_para_analisar:
        sys.exit(1)

    total_tickers = len(tickers_para_analisar)
    print(f"\nIniciando varredura em {total_tickers} ativos...")

    todos_os_alertas = []

    # 3. Executar Análise em Paralelo
    # Use um número razoável de workers para não sobrecarregar a rede/API
    MAX_WORKERS = 10

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submete cada ticker para análise e armazena o "futuro"
        futures = {executor.submit(analisar_ativo_para_screener, ticker, model,
                                   scaler, expected_features): ticker for ticker in tickers_para_analisar}

        # Processa os resultados conforme eles ficam prontos
        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                # Pega o resultado (uma lista de alertas para aquele ticker)
                resultados_ticker = future.result()
                if resultados_ticker:
                    todos_os_alertas.extend(resultados_ticker)
            except Exception as e:
                print(f"{Fore.RED}Erro ao processar o ticker {ticker}: {e}")

            # Imprime o progresso
            progresso = f"Progresso: {i + 1}/{total_tickers} ({((i + 1) / total_tickers):.1%})"
            sys.stdout.write(f"\r{C_YELLOW}{progresso}")
            sys.stdout.flush()

    print("\n\n--- Varredura Concluída ---")

    # 4. Exibir Relatório Final
    if not todos_os_alertas:
        print(f"{C_GREEN}Nenhum padrão de ALTA confiança (>= {LIMIAR_CONFIANCA:.0%}) foi encontrado nos {total_tickers} ativos analisados.")
    else:
        # Ordena os resultados pela confiança, do maior para o menor
        alertas_ordenados = sorted(
            todos_os_alertas, key=lambda x: x['confianca'], reverse=True)

        print(f"{C_GREEN}{C_BOLD}RELATÓRIO DE ALERTAS DE ALTA CONFIANÇA ({len(alertas_ordenados)} encontrados):")
        print("-" * 60)
        print(f"{'Ativo':<12} | {'Timeframe':<10} | {'Padrão':<8} | {'Confiança':<15}")
        print("-" * 60)

        for alerta in alertas_ordenados:
            conf_str = f"{alerta['confianca']:.2%}"
            print(
                f"{C_BOLD}{alerta['ticker']:<12}{Style.RESET_ALL} | {alerta['timeframe']:<10} | {alerta['padrao']:<8} | {C_GREEN}{conf_str:<15}")
        print("-" * 60)


if __name__ == "__main__":
    main()
