#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para Predição em Lote que inclui Engenharia de Features.
Versão corrigida para lidar com colunas MultiIndex do yfinance.
"""
import os
import sys
import argparse
from datetime import timedelta
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from colorama import Fore, Style, init

# (O resto do cabeçalho e as funções de cálculo de features permanecem as mesmas)
# ...
# ... (Cole aqui as funções get_price_on_neckline, _calcular_volume_rompimento, e calcular_features_padrao) ...
# Vou omiti-las aqui para manter a resposta focada na correção, mas elas precisam estar no seu script.


# --- FUNÇÕES DE FEATURE ENGINEERING (COLE AS SUAS AQUI) ---
MAX_VELAS_BUSCA_ROMPIMENTO = 20
VELAS_VOLUME_POS_ROMPIMENTO = 5
def get_price_on_neckline(point_idx, p1, p2, slope):
    time_delta = (point_idx - p1['idx']).total_seconds() / (3600 * 24)
    return p1['preco'] + slope * time_delta
def _calcular_volume_rompimento(df, ombro2, neckline_p1, neckline_slope, tipo_padrao):
    try:
        start_loc = df.index.get_loc(ombro2['idx'])
        df_busca = df.iloc[start_loc + 1 : start_loc + 1 + MAX_VELAS_BUSCA_ROMPIMENTO]
        if df_busca.empty: return None
        for idx, vela in df_busca.iterrows():
            time_delta = (idx - neckline_p1['idx']).total_seconds() / (3600 * 24)
            neckline_price_atual = neckline_p1['preco'] + neckline_slope * time_delta
            rompeu = (tipo_padrao == 'OCOI' and vela['close'] > neckline_price_atual) or \
                     (tipo_padrao == 'OCO' and vela['close'] < neckline_price_atual)
            if rompeu:
                loc_rompimento = df.index.get_loc(idx)
                df_volume = df.iloc[loc_rompimento : loc_rompimento + VELAS_VOLUME_POS_ROMPIMENTO]
                if len(df_volume) < VELAS_VOLUME_POS_ROMPIMENTO: return None
                return df_volume['volume'].mean()
        return None
    except Exception: return None
def calcular_features_padrao(candidato_extremos, df_completo, tipo_padrao):
    try:
        ombro1, neckline_p1, cabeca, neckline_p2, ombro2 = candidato_extremos
        delta_tempo_neckline = (neckline_p2['idx'] - neckline_p1['idx']).total_seconds() / (3600 * 24)
        if delta_tempo_neckline == 0: return None
        neckline_slope = (neckline_p2['preco'] - neckline_p1['preco']) / delta_tempo_neckline
        altura_cabeca_rel = abs(cabeca['preco'] - get_price_on_neckline(cabeca['idx'], neckline_p1, neckline_p2, neckline_slope))
        if altura_cabeca_rel == 0: return None
        altura_ombro1_rel = abs(ombro1['preco'] - get_price_on_neckline(ombro1['idx'], neckline_p1, neckline_p2, neckline_slope))
        altura_ombro2_rel = abs(ombro2['preco'] - get_price_on_neckline(ombro2['idx'], neckline_p1, neckline_p2, neckline_slope))
        df_padrao = df_completo.loc[ombro1['idx']:ombro2['idx']]
        vol_medio_padrao = df_padrao['volume'].mean()
        vol_medio_rompimento = _calcular_volume_rompimento(df_completo, ombro2, neckline_p1, neckline_slope, tipo_padrao)
        volume_breakout_ratio = vol_medio_rompimento / vol_medio_padrao if vol_medio_rompimento and vol_medio_padrao > 0 else 1.0
        ruido_padrao = df_padrao['close'].pct_change().std()
        return {
            'altura_rel_cabeca': altura_cabeca_rel, 'ratio_ombro_esquerdo': altura_ombro1_rel / altura_cabeca_rel,
            'ratio_ombro_direito': altura_ombro2_rel / altura_cabeca_rel, 'ratio_simetria_altura_ombros': min(altura_ombro1_rel, altura_ombro2_rel) / max(altura_ombro1_rel, altura_ombro2_rel) if max(altura_ombro1_rel, altura_ombro2_rel) > 0 else 1.0,
            'neckline_slope': neckline_slope, 'volume_breakout_ratio': volume_breakout_ratio, 'intervalo_em_minutos': (df_completo.index[1] - df_completo.index[0]).total_seconds() / 60,
            'duracao_em_velas': len(df_padrao), 'dist_ombro1_cabeca': (cabeca['idx'] - ombro1['idx']).total_seconds() / (3600*24),
            'dist_cabeca_ombro2': (ombro2['idx'] - cabeca['idx']).total_seconds() / (3600*24), 'ratio_simetria_temporal': min((cabeca['idx'] - ombro1['idx']).total_seconds(), (ombro2['idx'] - cabeca['idx']).total_seconds()) / max((cabeca['idx'] - ombro1['idx']).total_seconds(), (ombro2['idx'] - cabeca['idx']).total_seconds()) if max((cabeca['idx'] - ombro1['idx']).total_seconds(), (ombro2['idx'] - cabeca['idx']).total_seconds()) > 0 else 1.0,
            'dif_altura_ombros_rel': (altura_ombro2_rel - altura_ombro1_rel) / altura_cabeca_rel, 'extensao_ombro1': (neckline_p1['idx'] - ombro1['idx']).total_seconds() / (3600*24),
            'extensao_ombro2': (ombro2['idx'] - neckline_p2['idx']).total_seconds() / (3600*24), 'neckline_angle_rad': np.arctan(neckline_slope),
            'ruido_padrao': ruido_padrao if pd.notna(ruido_padrao) else 0.0,
        }
    except Exception: return None

def main():
    # ... (código do parser e carregamento do modelo continua igual) ...
    # O loop principal é onde a mudança acontece
    init(autoreset=True)
    MODELO_PATH = 'data/models/modelo_qualidade_pattens.joblib'
    RESULTADO_CSV_PATH = 'data/reports/resultado_predicoes_com_features.csv'
    LIMIAR_PREVISAO = 0.5
    
    parser = argparse.ArgumentParser(description="Calcula features e prevê a validade de padrões brutos.")
    parser.add_argument("input_csv", type=str, help="Caminho para o CSV com os dados brutos dos padrões.")
    args = parser.parse_args()

    try:
        artefatos = joblib.load(MODELO_PATH)
        model, scaler, expected_features = artefatos['model'], artefatos['scaler'], artefatos['features']
        print(f"{Fore.GREEN}✓ Modelo, Scaler e {len(expected_features)} features carregados.")
    except Exception as e:
        print(f"{Fore.RED}ERRO CRÍTICO ao carregar o modelo: {e}"); sys.exit(1)

    try:
        df = pd.read_csv(args.input_csv)
        date_cols = [col for col in df.columns if '_idx' in col]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
        print(f"Lendo {len(df)} padrões brutos de {args.input_csv}.")
    except Exception as e:
        print(f"{Fore.RED}ERRO ao ler ou processar o CSV de entrada: {e}"); sys.exit(1)

    print("Iniciando cálculo de features e predição...")
    dados_cacheados = {}
    resultados = []

    for index, row in df.iterrows():
        ticker, timeframe, tipo_padrao = row['ticker'], row['timeframe'], row['padrao_tipo']
        cache_key = f"{ticker}_{timeframe}"
        print(f"Processando padrão {index+1}/{len(df)} para {ticker} ({timeframe})... ", end="")

        if cache_key not in dados_cacheados:
            try:
                start_date = row['ombro1_idx'] - timedelta(days=10)
                end_date = row['ombro2_idx'] + timedelta(days=10)
                dados_historicos = yf.download(
                    tickers=ticker, interval=timeframe, start=start_date, end=end_date,
                    progress=False, auto_adjust=True
                )
                
                # --- INÍCIO DA CORREÇÃO ---
                # Adiciona a mesma verificação de MultiIndex que fizemos no outro script
                if isinstance(dados_historicos.columns, pd.MultiIndex):
                    dados_historicos.columns = dados_historicos.columns.get_level_values(0)
                # --- FIM DA CORREÇÃO ---

                dados_historicos.columns = [col.lower() for col in dados_historicos.columns]
                dados_cacheados[cache_key] = dados_historicos

            except Exception as e:
                # Agora esta mensagem de erro será mais precisa
                print(f"{Fore.RED}Falha ao baixar dados para {ticker}. Erro: {e}")
                resultados.append({'confianca_modelo': np.nan, 'previsao_modelo': np.nan})
                continue
        
        df_historico = dados_cacheados[cache_key]
        
        # (O resto do loop para calcular features e prever continua exatamente o mesmo)
        candidato_extremos = [
            {'idx': row['ombro1_idx'], 'preco': row['ombro1_preco'], 'tipo': 'VALE' if tipo_padrao == 'OCOI' else 'PICO'},
            {'idx': row['neckline1_idx'], 'preco': row['neckline1_preco'], 'tipo': 'PICO' if tipo_padrao == 'OCOI' else 'VALE'},
            {'idx': row['cabeca_idx'], 'preco': row['cabeca_preco'], 'tipo': 'VALE' if tipo_padrao == 'OCOI' else 'PICO'},
            {'idx': row['neckline2_idx'], 'preco': row['neckline2_preco'], 'tipo': 'PICO' if tipo_padrao == 'OCOI' else 'VALE'},
            {'idx': row['ombro2_idx'], 'preco': row['ombro2_preco'], 'tipo': 'VALE' if tipo_padrao == 'OCOI' else 'PICO'}
        ]
        features = calcular_features_padrao(candidato_extremos, df_historico, tipo_padrao)
        if features is None:
            print(f"{Fore.YELLOW}Não foi possível calcular features.")
            resultados.append({'confianca_modelo': np.nan, 'previsao_modelo': np.nan})
            continue
        try:
            feature_values = np.array([features[fname] for fname in expected_features]).reshape(1, -1)
            dados_normalizados = scaler.transform(feature_values)
            probabilidades = model.predict_proba(dados_normalizados)
            confianca = probabilidades[0][1]
            previsao = 1 if confianca >= LIMIAR_PREVISAO else 0
            resultados.append({'confianca_modelo': confianca, 'previsao_modelo': previsao})
            print(f"{Fore.GREEN}Confiança: {confianca:.2%}")
        except Exception as e:
            print(f"{Fore.RED}Erro na predição. Erro: {e}")
            resultados.append({'confianca_modelo': np.nan, 'previsao_modelo': np.nan})

    # ... (código de finalização e salvamento do resultado continua igual) ...
    df_resultados = pd.DataFrame(resultados)
    df_final = pd.concat([df.reset_index(drop=True), df_resultados], axis=1)

    print("\n--- " + Fore.GREEN + "Resultado Final das Predições" + Style.RESET_ALL + " ---")
    colunas_display = ['ticker', 'timeframe', 'padrao_tipo', 'confianca_modelo', 'previsao_modelo']
    # Apenas para garantir que não dê erro se alguma coluna estiver faltando no df_final
    colunas_reais = [col for col in colunas_display if col in df_final.columns]
    print(df_final[colunas_reais].to_string(index=False))

    output_dir = os.path.dirname(RESULTADO_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv(RESULTADO_CSV_PATH, index=False)
    print(f"\n{Fore.GREEN}✓ Resultados completos salvos em: {Style.BRIGHT}{RESULTADO_CSV_PATH}")


if __name__ == "__main__":
    main()