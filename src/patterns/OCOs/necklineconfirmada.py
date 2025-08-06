# --- gerador_de_dataset.py (v20 - Lógica de Estratégias) ---
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from typing import List, Dict, Any, Optional
import os
import time
from colorama import Fore, Style, init

# Inicializa o Colorama
init(autoreset=True)


class Config:
    TICKERS = [
        'AAVE-USD',   # Aave
        'ADA-USD',    # Cardano
        'ALGO-USD',   # Algorand
        'AVAX-USD',   # Avalanche
        'BCH-USD',    # Bitcoin Cash
        'BNB-USD',    # BNB
        'BTC-USD',    # Bitcoin
        'BSV-USD',    # Bitcoin SV
        'CHZ-USD',    # Chiliz
        'CRO-USD',    # Cronos
        'DOGE-USD',   # Dogecoin
        'DOT-USD',    # Polkadot
        'EGLD-USD',   # MultiversX
        'EOS-USD',    # EOS
        'ETC-USD',    # Ethereum Classic
        'ETH-USD',    # Ethereum
        'FIL-USD',    # Filecoin
        'FLOW-USD',   # Flow
        'HBAR-USD',   # Hedera
        'ICP-USD',    # Internet Computer
        'LDO-USD',    # Lido DAO
        'LINK-USD',   # Chainlink
        'LTC-USD',    # Litecoin
        'MANA-USD',   # Decentraland
        'MKR-USD',    # Maker
        'NEAR-USD',   # NEAR Protocol
        'NEO-USD',    # Neo
        'OP-USD',     # Optimism
        'QNT-USD',    # Quant
        'SHIB-USD',   # Shiba Inu
        'SNX-USD',    # Synthetix
        'SOL-USD',    # Solana
        'THETA-USD',  # Theta Network
        'TRX-USD',    # TRON
        'VET-USD',    # VeChain
        'XLM-USD',    # Stellar
        'XMR-USD',    # Monero
        'XRP-USD',    # XRP
        'XTZ-USD',    # Tezos
        'ZIL-USD',    # Zilliqa
    ]
    DATA_PERIOD = '5y'

    # <<< ALTERAÇÃO 1: ZIGZAG_STRATEGIES agora é a fonte da verdade >>>
    # As antigas variáveis INTERVALS e TIMEFRAME_PARAMS foram removidas.
    ZIGZAG_STRATEGIES = {
        # ------- SCALPING (micro-estruturas) ----------
        'scalping_aggressive': {
            '5m':  {'depth': 3, 'deviation': 0.25}
        },
        'scalping_moderate': {
            '5m':  {'depth': 4, 'deviation': 0.40},
            '15m': {'depth': 5, 'deviation': 0.60}
        },
        'scalping_conservative': {
            '5m':  {'depth': 5, 'deviation': 0.55},
            '15m': {'depth': 6, 'deviation': 0.75}
        },

        # ------- INTRADAY (15m-1h) ----------
        'intraday_momentum': {
            '5m':  {'depth': 6, 'deviation': 0.80},
            '15m': {'depth': 7, 'deviation': 1.10},
            '1h':  {'depth': 8, 'deviation': 1.60}
        },
        'intraday_range': {
            '5m':  {'depth': 7, 'deviation': 1.00},
            '15m': {'depth': 8, 'deviation': 1.30},
            '1h':  {'depth': 9, 'deviation': 1.90}
        },

        # ------- SWING (hor. de horas a dias) ----------
        'swing_short': {
            '15m': {'depth': 8,  'deviation': 2.0},
            '1h':  {'depth': 10, 'deviation': 2.8},
            '4h':  {'depth': 12, 'deviation': 4.0}
        },
    }

    # --- SISTEMA DE REGRAS E PONTUAÇÃO (Sem alterações) ---
    SCORE_WEIGHTS = {
        # Regras obrigatórias
        'valid_extremo_cabeca': 20, 'valid_contexto_cabeca': 15,
        'valid_simetria_ombros': 10, 'valid_neckline_plana': 5,
        'valid_base_tendencia': 5,
        'valid_neckline_retest_p6': 15,
        # Regras opicionais
        'valid_divergencia_rsi': 15,
        'valid_divergencia_macd': 10, 'valid_proeminencia_cabeca': 10,
        'valid_ombro_direito_fraco': 5, 'valid_perfil_volume': 5
    }
    MINIMUM_SCORE_TO_SAVE = 70

    # Parâmetros de validação (Sem alterações)
    HEAD_SIGNIFICANCE_RATIO = 1.1
    SHOULDER_SYMMETRY_TOLERANCE = 0.30
    NECKLINE_FLATNESS_TOLERANCE = 0.25
    HEAD_EXTREME_LOOKBACK_FACTOR = 3

    RECENT_PATTERNS_LOOKBACK_COUNT = 1

    ZIGZAG_EXTEND_TO_LAST_BAR = True

    MAX_DOWNLOAD_TENTATIVAS, RETRY_DELAY_SEGUNDOS = 3, 5
    # <<< ALTERAÇÃO 2: Novo diretório de saída para organizar os resultados >>>
    OUTPUT_DIR = 'data/datasets/datasets_hns_by_strategy'
    FINAL_CSV_PATH = os.path.join(
        OUTPUT_DIR, 'dataset_hns_by_strategy_final.csv')

# --- FUNÇÕES AUXILIARES (Sem alterações no corpo das funções) ---
# buscar_dados, calcular_zigzag_oficial, is_head_extreme, check_rsi_divergence,
# check_macd_divergence, check_volume_profile, validate_and_score_hns_pattern,
# identificar_padroes_hns continuam exatamente as mesmas.
# O código delas não precisa mudar, pois elas recebem os parâmetros que precisam.


def buscar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Busca dados históricos ajustando o período solicitado para respeitar
    os limites da API do Yahoo Finance.
    """
    original_period = period

    # <<< CORREÇÃO: Respeitando os limites da API do yfinance >>>
    # A API do Yahoo Finance permite buscar no máximo 7-8 dias de dados
    # quando a granularidade (interval) é em minutos.
    if 'mo' in interval:
        period = 'max'
    elif 'm' in interval:
        period = '7d'  # Alterado de '60d' para '7d'
    elif 'h' in interval:
        period = '2y'

    if period != original_period:
        print(f"{Fore.YELLOW}Aviso: Período padrão '{original_period}' ajustado para '{period}' para o intervalo '{interval}' para respeitar limites da API.{Style.RESET_ALL}")

    for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
        try:
            df = yf.download(tickers=ticker, period=period,
                             interval=interval, auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [col.lower() for col in df.columns]
                return df
            else:
                raise ValueError("Download retornou um DataFrame vazio.")
        except Exception as e:
            if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1:
                print(
                    f"{Fore.YELLOW}Tentativa {tentativa + 1} falhou. Tentando novamente em {Config.RETRY_DELAY_SEGUNDOS}s...{Style.RESET_ALL}")
                time.sleep(Config.RETRY_DELAY_SEGUNDOS)
            else:
                # Lança uma exceção mais específica para ser capturada no loop principal
                raise ConnectionError(
                    f"Download falhou para {ticker}/{interval} após {Config.MAX_DOWNLOAD_TENTATIVAS} tentativas. Erro: {e}")

    # Este ponto não deveria ser alcançado, mas é uma boa prática ter um fallback.
    raise ConnectionError(
        f"Falha inesperada no download para {ticker}/{interval}.")


def calcular_zigzag_oficial(df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
    # ... (código idêntico, usando apenas high/low)
    peak_series, valley_series = df['high'], df['low']
    window_size = 2 * depth + 1
    rolling_max, rolling_min = peak_series.rolling(window=window_size, center=True, min_periods=1).max(
    ), valley_series.rolling(window=window_size, center=True, min_periods=1).min()
    candidate_peaks_df, candidate_valleys_df = df[peak_series ==
                                                  rolling_max], df[valley_series == rolling_min]
    candidates = []
    for idx, row in candidate_peaks_df.iterrows():
        candidates.append(
            {'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
    for idx, row in candidate_valleys_df.iterrows():
        candidates.append(
            {'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})
    candidates = sorted(
        list({p['idx']: p for p in candidates}.values()), key=lambda x: x['idx'])
    if len(candidates) < 2:
        return []
    confirmed_pivots = [candidates[0]]
    last_pivot = candidates[0]
    for i in range(1, len(candidates)):
        candidate = candidates[i]
        if candidate['tipo'] == last_pivot['tipo']:
            if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                confirmed_pivots[-1], last_pivot = candidate, candidate
            continue
        if last_pivot['preco'] == 0:
            continue
        price_dev = abs(
            candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
        if price_dev >= deviation_percent:
            confirmed_pivots.append(candidate)
            last_pivot = candidate
    if Config.ZIGZAG_EXTEND_TO_LAST_BAR and confirmed_pivots:
        last_confirmed_pivot = confirmed_pivots[-1]
        last_bar = df.iloc[-1]

        # Se o último candle prolongar o movimento na MESMA direção,
        # apenas atualizamos o pivô existente. Caso contrário, criamos um novo pivô.
        if last_confirmed_pivot['tipo'] == 'PICO':
            # Movimento ainda de alta: atualiza o pico existente
            if last_bar['high'] > last_confirmed_pivot['preco']:
                last_confirmed_pivot['preco'] = last_bar['high']
                last_confirmed_pivot['idx'] = df.index[-1]
            else:
                # Inverteu para baixa → cria um VALE
                potential_pivot = {
                    'idx': df.index[-1],
                    'tipo': 'VALE',
                    'preco': last_bar['low']
                }
                if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                    confirmed_pivots.append(potential_pivot)
        else:  # último pivô é VALE
            # Movimento ainda de baixa: atualiza o vale existente
            if last_bar['low'] < last_confirmed_pivot['preco']:
                last_confirmed_pivot['preco'] = last_bar['low']
                last_confirmed_pivot['idx'] = df.index[-1]
            else:
                # Inverteu para alta → cria um PICO
                potential_pivot = {
                    'idx': df.index[-1],
                    'tipo': 'PICO',
                    'preco': last_bar['high']
                }
                if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                    confirmed_pivots.append(potential_pivot)

    return confirmed_pivots


def is_head_extreme(df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_days: int) -> bool:
    # ... (código idêntico)
    lookback_period = int(avg_pivot_dist_days *
                          Config.HEAD_EXTREME_LOOKBACK_FACTOR)
    if lookback_period <= 0:
        return True
    start_date, end_date = head_pivot['idx'] - pd.Timedelta(
        days=lookback_period), head_pivot['idx'] + pd.Timedelta(days=lookback_period)
    context_df = df.loc[start_date:end_date]
    if context_df.empty:
        return True
    if head_pivot['tipo'] == 'PICO':
        return head_pivot['preco'] >= context_df['high'].max()
    else:
        return head_pivot['preco'] <= context_df['low'].min()


def check_rsi_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    # ... (código idêntico)
    try:
        if tipo_padrao == 'OCO':
            rsi_series = ta.rsi(df['high'], length=14)
            if p1_idx not in rsi_series.index or p3_idx not in rsi_series.index:
                return False
            rsi_p1, rsi_p3 = rsi_series.loc[p1_idx], rsi_series.loc[p3_idx]
            return p3_price > p1_price and rsi_p3 < rsi_p1
        elif tipo_padrao == 'OCOI':
            rsi_series = ta.rsi(df['low'], length=14)
            if p1_idx not in rsi_series.index or p3_idx not in rsi_series.index:
                return False
            rsi_p1, rsi_p3 = rsi_series.loc[p1_idx], rsi_series.loc[p3_idx]
            return p3_price < p1_price and rsi_p3 > rsi_p1
    except Exception:
        return False
    return False


def check_macd_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
    # ... (código idêntico)
    try:
        macd = df.ta.macd(fast=12, slow=26, signal=9, append=False)
        hist_col = 'MACDh_12_26_9'
        if p1_idx not in macd.index or p3_idx not in macd.index:
            return False
        hist_p1, hist_p3 = macd[hist_col].loc[p1_idx], macd[hist_col].loc[p3_idx]
        if tipo_padrao == 'OCO':
            return p3_price > p1_price and hist_p3 < hist_p1
        elif tipo_padrao == 'OCOI':
            return p3_price < p1_price and hist_p3 > hist_p1
    except Exception:
        return False
    return False


def check_volume_profile(df: pd.DataFrame, pivots: List[Dict[str, Any]], p1_idx, p3_idx, p5_idx) -> bool:
    # ... (código idêntico)
    try:
        indices = {p['idx']: i for i, p in enumerate(pivots)}
        idx_p1, idx_p3, idx_p5 = indices.get(
            p1_idx), indices.get(p3_idx), indices.get(p5_idx)
        if any(i is None for i in [idx_p1, idx_p3, idx_p5]) or idx_p1 < 2:
            return False
        p0_idx, p2_idx, p4_idx = pivots[idx_p1 -
                                        1]['idx'], pivots[idx_p3-1]['idx'], pivots[idx_p5-1]['idx']
        vol_cabeca = df.loc[p2_idx:p3_idx]['volume'].mean()
        vol_od = df.loc[p4_idx:p5_idx]['volume'].mean()
        return vol_cabeca > vol_od
    except Exception:
        return False
    return False


### MUDANÇA 6: Atualizar a assinatura da função para receber p6 ###
def validate_and_score_hns_pattern(p0, p1, p2, p3, p4, p5, p6, tipo_padrao, df_historico, pivots, avg_pivot_dist_days):
    details = {key: False for key in Config.SCORE_WEIGHTS.keys()}
    ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5

    # --- Lógica de validação original (sem alterações) ---
    altura_cabeca = abs(
        cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_esq = abs(
        ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
    altura_ombro_dir = abs(
        ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))

    details['valid_extremo_cabeca'] = (tipo_padrao == 'OCO' and cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']) or \
                                      (tipo_padrao == 'OCOI' and cabeca['preco'] <
                                       ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco'])
    if not details['valid_extremo_cabeca']:
        return None

    details['valid_contexto_cabeca'] = is_head_extreme(
        df_historico, cabeca, avg_pivot_dist_days)
    if not details['valid_contexto_cabeca']:
        return None

    details['valid_simetria_ombros'] = altura_cabeca > 0 and \
        abs(altura_ombro_esq - altura_ombro_dir) <= altura_cabeca * \
        Config.SHOULDER_SYMMETRY_TOLERANCE
    if not details['valid_simetria_ombros']:
        return None

    details['valid_neckline_plana'] = altura_ombro_esq > 0 and \
        abs(neckline1['preco'] - neckline2['preco']
            ) <= altura_ombro_esq * Config.NECKLINE_FLATNESS_TOLERANCE
    if not details['valid_neckline_plana']:
        return None

    details['valid_base_tendencia'] = (tipo_padrao == 'OCO' and ((p0['preco'] < neckline1['preco'] and p0['preco'] < neckline2['preco']) or (abs(p0['preco'] - neckline1['preco']) < p0['preco'] * 0.05 and abs(p0['preco'] - neckline2['preco']) < p0['preco'] * 0.05))) or \
                                      (tipo_padrao == 'OCOI' and ((p0['preco'] > neckline1['preco'] and p0['preco'] > neckline2['preco']) or (abs(
                                          p0['preco'] - neckline1['preco']) < p0['preco'] * 0.05 and abs(p0['preco'] - neckline2['preco']) < p0['preco'] * 0.05)))
    if not details['valid_base_tendencia']:
        return None

    ### MUDANÇA 7: Implementar a nova lógica de validação para o pivô p6 ###
    # Calcular o preço médio da neckline
    neckline_price = np.mean([neckline1['preco'], neckline2['preco']])

    # --- Tolerância adaptativa baseada no ATR ---
    # Calcula ATR(14) utilizando pandas_ta
    atr_series = df_historico.ta.atr(length=14)
    # Procura o ATR no índice do p6; se não houver valor, usa o último ATR disponível
    if p6['idx'] in atr_series.index and not np.isnan(atr_series.loc[p6['idx']]):
        atr_val = atr_series.loc[p6['idx']]
    else:
        atr_val = atr_series.dropna(
        ).iloc[-1] if not atr_series.dropna().empty else 0

    # Percentual de tolerância: 25 % do ATR relativo ao preço da neckline
    # com piso absoluto de 0,3 % para evitar valores demasiadamente pequenos
    tol_pct = 0.3 * atr_val / neckline_price if neckline_price else 0
    tol_pct = max(tol_pct, 0.01)  # mínimo de 0,3 %

    max_variation = neckline_price * tol_pct
    # max_variation = neckline_price * 0.03

    # Verificar se o preço do p6 está dentro da faixa de tolerância da neckline
    is_close_to_neckline = abs(p6['preco'] - neckline_price) <= max_variation

    # Registra o resultado da validação
    details['valid_neckline_retest_p6'] = is_close_to_neckline

    # Se o reteste do p6 for a regra principal, tornamo-la eliminatória
    if not details['valid_neckline_retest_p6']:
        return None
    # --- Fim da nova lógica ---

    # --- Lógica de pontuação (com a nova regra já adicionada aos "details") ---
    score = 0
    for rule, passed in details.items():
        if passed:
            score += Config.SCORE_WEIGHTS.get(rule, 0)

    # ... (Resto da lógica de pontuação opcional permanece a mesma) ...
    if altura_ombro_esq > 0 and (altura_cabeca / altura_ombro_esq >= Config.HEAD_SIGNIFICANCE_RATIO) and (altura_cabeca / altura_ombro_dir >= Config.HEAD_SIGNIFICANCE_RATIO):
        details['valid_proeminencia_cabeca'] = True
        score += Config.SCORE_WEIGHTS['valid_proeminencia_cabeca']
    if check_rsi_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_rsi'] = True
        score += Config.SCORE_WEIGHTS['valid_divergencia_rsi']
    if check_macd_divergence(df_historico, p1['idx'], p3['idx'], p1['preco'], p3['preco'], tipo_padrao):
        details['valid_divergencia_macd'] = True
        score += Config.SCORE_WEIGHTS['valid_divergencia_macd']
    if (tipo_padrao == 'OCO' and ombro_dir['preco'] < ombro_esq['preco']) or \
       (tipo_padrao == 'OCOI' and ombro_dir['preco'] > ombro_esq['preco']):
        details['valid_ombro_direito_fraco'] = True
        score += Config.SCORE_WEIGHTS['valid_ombro_direito_fraco']
    if check_volume_profile(df_historico, pivots, p1['idx'], p3['idx'], p5['idx']):
        details['valid_perfil_volume'] = True
        score += Config.SCORE_WEIGHTS['valid_perfil_volume']

    if score >= Config.MINIMUM_SCORE_TO_SAVE:
        base_data = {
            'padrao_tipo': tipo_padrao, 'score_total': score,
            'p0_idx': p0['idx'],
            'ombro1_idx': p1['idx'], 'ombro1_preco': p1['preco'],
            'neckline1_idx': p2['idx'], 'neckline1_preco': p2['preco'],
            'cabeca_idx': p3['idx'], 'cabeca_preco': p3['preco'],
            'neckline2_idx': p4['idx'], 'neckline2_preco': p4['preco'],
            'ombro2_idx': p5['idx'], 'ombro2_preco': p5['preco'],
            ### MUDANÇA 8: Adicionar os dados de p6 ao resultado final ###
            'retest_p6_idx': p6['idx'], 'retest_p6_preco': p6['preco']
        }
        base_data.update(details)
        return base_data

    return None


def identificar_padroes_hns(pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
    # ... (código idêntico)
    padroes_encontrados = []
    n = len(pivots)
    if n < 7:
        return []
    avg_pivot_dist_days = np.mean(
        [(pivots[i]['idx'] - pivots[i-1]['idx']).days for i in range(1, n)]) if n > 1 else 0
    start_index = max(0, n - 6 - Config.RECENT_PATTERNS_LOOKBACK_COUNT)

    print(
        f"Analisando apenas os últimos {Config.RECENT_PATTERNS_LOOKBACK_COUNT} pivôs finais possíveis (a partir do índice {start_index}).")

    for i in range(start_index, n - 6):
        janela = pivots[i:i+7]
        # O resto da função continua exatamente igual
        p0, p1, p2, p3, p4, p5 = janela[0], janela[1], janela[2], janela[3], janela[4], janela[5]
        ### MUDANÇA 4: Capturar o pivô p6 para o reteste ###
        p6 = janela[6]

        tipo_padrao = None
        # Verifica se a janela corresponde a um padrão OCO
        # A sequência de tipos de pivô continua a mesma, pois p6 é o reteste
        if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])):
            tipo_padrao = 'OCO'
        # Verifica se a janela corresponde a um padrão OCOI
        elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])):
            tipo_padrao = 'OCOI'

        if tipo_padrao:
            ### MUDANÇA 5: Passar p6 para a função de validação ###
            dados_padrao = validate_and_score_hns_pattern(
                p0, p1, p2, p3, p4, p5, p6, tipo_padrao, df_historico, pivots, avg_pivot_dist_days)
            if dados_padrao:
                padroes_encontrados.append(dados_padrao)

    return padroes_encontrados


def main():
    print(f"{Style.BRIGHT}--- INICIANDO MOTOR DE GERAÇÃO (v20 - Estratégias) ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    todos_os_padroes_finais = []

    # <<< ALTERAÇÃO 3: O loop principal agora itera sobre as estratégias >>>
    for strategy_name, intervals_config in Config.ZIGZAG_STRATEGIES.items():
        print(f"\n{Style.BRIGHT}===== ESTRATÉGIA: {strategy_name.upper()} =====")
        for interval, params in intervals_config.items():
            for ticker in Config.TICKERS:
                print(
                    f"\n--- Processando: {ticker} | Intervalo: {interval} (Estratégia: {strategy_name}) ---")
                try:
                    df_historico = buscar_dados(
                        ticker, Config.DATA_PERIOD, interval)

                    print(
                        f"Calculando ZigZag com depth={params['depth']}, deviation={params['deviation']}%...")
                    pivots_detectados = calcular_zigzag_oficial(
                        df_historico, params['depth'], params['deviation'])

                    if len(pivots_detectados) < 7:
                        print(
                            "ℹ️ Número insuficiente de pivôs para formar um padrão.")
                        continue

                    print("Identificando padrões H&S com regras obrigatórias...")
                    padroes_encontrados = identificar_padroes_hns(
                        pivots_detectados, df_historico)

                    if padroes_encontrados:
                        print(
                            f"{Fore.GREEN}✅ Encontrados {len(padroes_encontrados)} padrões que passaram nas regras e atingiram o score.")
                        for padrao in padroes_encontrados:
                            # <<< ALTERAÇÃO 4: Adiciona a estratégia e o timeframe ao resultado >>>
                            padrao['strategy'] = strategy_name
                            padrao['timeframe'] = interval
                            padrao['ticker'] = ticker
                            todos_os_padroes_finais.append(padrao)
                    else:
                        print(
                            "ℹ️ Nenhum padrão passou nos critérios obrigatórios ou atingiu a pontuação mínima.")
                except Exception as e:
                    print(
                        f"{Fore.RED}❌ Erro ao processar {ticker}/{interval} na estratégia {strategy_name}: {e}")

    print(
        f"\n{Style.BRIGHT}--- Processo finalizado. Salvando dataset... ---{Style.RESET_ALL}")

    if not todos_os_padroes_finais:
        print(f"{Fore.YELLOW}Nenhum padrão foi encontrado em todas as execuções.")
        return

    df_final = pd.DataFrame(todos_os_padroes_finais)

    # <<< ALTERAÇÃO 5: Inclui 'strategy' na chave de identificação de duplicatas >>>
    df_final.drop_duplicates(subset=[
        'ticker', 'timeframe', 'padrao_tipo', 'cabeca_idx'], inplace=True, keep='first')

    # <<< ALTERAÇÃO 6: Adiciona 'strategy' nas colunas de informação >>>
    cols_info = ['ticker', 'timeframe',
                 'strategy', 'padrao_tipo', 'score_total']
    cols_validacao = sorted(
        [col for col in df_final.columns if col.startswith('valid_')])
    cols_pontos = [
        col for col in df_final.columns if col.endswith(('_idx', '_preco'))]

    # Garante que todas as colunas existentes sejam incluídas
    existing_cols = set(df_final.columns)
    ordem_final = [c for c in (
        cols_info + cols_validacao + cols_pontos) if c in existing_cols]

    df_final = df_final.reindex(columns=ordem_final)

    df_final.to_csv(Config.FINAL_CSV_PATH, index=False,
                    date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n{Fore.GREEN}✅ Dataset final com {len(df_final)} padrões únicos salvo em: {Config.FINAL_CSV_PATH}")


if __name__ == "__main__":
    main()
