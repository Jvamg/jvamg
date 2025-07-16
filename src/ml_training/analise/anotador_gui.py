# anotador_gui.py (v7.5 - Lógica Fiel ao Pine Script)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os
import time

class Config:
    """ Parâmetros de configuração centralizados para a GUI e o detector. """
    TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'HBAR-USD']
    INTERVALS = ['15m', '1h', '4h', '1d']
    DATA_PERIOD = '5y'

    TIMEFRAME_PARAMS = {
        'default': {'deviation': 2.0, 'min_distance': 2},
        '1h': {'deviation': 4.0, 'min_distance': 3},
        '4h': {'deviation': 5.0, 'min_distance': 3},
        '1d': {'deviation': 8.0, 'min_distance': 5}
    }

    # --- PARÂMETROS DE VALIDAÇÃO (AJUSTADOS PARA FICAREM IGUAIS AO PINE SCRIPT) ---
    # Tolerância para a diferença de altura entre os ombros (em relação à altura da cabeça)
    SHOULDER_SYMMETRY_TOLERANCE = 0.10  # <--- MUDANÇA: De 0.30 para 0.10 (muito mais rigoroso)
    # Tolerância para a inclinação da neckline (em relação à altura do ombro esquerdo)
    NECKLINE_FLATNESS_TOLERANCE = 0.10  # <--- MUDANÇA: De 0.25 para 0.10 (muito mais rigoroso)
    # Proporção mínima da altura da cabeça em relação ao ombro esquerdo
    HEAD_SIGNIFICANCE_RATIO = 1.2       # Mantido (Cabeça 20% maior)
    # Fator de lookback para validação de contexto da cabeça
    HEAD_EXTREME_LOOKBACK_FACTOR = 5

    # Saída
    ARQUIVO_ENTRADA = 'data/datasets_hns/hns_final_adaptive_dataset.csv'
    ARQUIVO_SAIDA = 'data/datasets/filtered/dataset_hns_labeled_v7.5.csv'
    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5

class LabelingTool(tk.Tk):
    # O __init__ e outras funções permanecem os mesmos...
    def __init__(self, arquivo_entrada, arquivo_saida):
        super().__init__()
        self.title(f"Ferramenta de Anotação (v7.5 - Lógica Fiel)")
        self.geometry("1300x900")
        self.arquivo_saida = arquivo_saida
        self.df_trabalho = None
        self.indice_atual = 0
        if not self.setup_dataframe(arquivo_entrada, arquivo_saida):
            self.destroy()
            return
        self._setup_ui()
        self.bind('<Key>', self.on_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.carregar_proximo_padrao()

    def _setup_ui(self):
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)
        self.frame_grafico = tk.Frame(m)
        m.add(self.frame_grafico, minsize=400)
        frame_controles = tk.Frame(m, height=100)
        frame_controles.pack_propagate(False)
        m.add(frame_controles, minsize=100)
        self.info_label = tk.Label(frame_controles, text="Carregando...", font=("Arial", 12), justify=tk.LEFT, anchor="w")
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.action_label = tk.Label(frame_controles, text="Teclado: [A]provar | [R]ejeitar | [Q]uit", font=("Arial", 12, "bold"))
        self.action_label.pack(side=tk.RIGHT, padx=10)

    def setup_dataframe(self, arquivo_entrada, arquivo_saida):
        try:
            if os.path.exists(arquivo_saida): self.df_trabalho = pd.read_csv(arquivo_saida)
            else: self.df_trabalho = pd.read_csv(arquivo_entrada)
            rename_map = {'timeframe': 'intervalo', 'padrao_tipo': 'tipo_padrao', 'ombro1_idx': 'data_inicio', 'ombro2_idx': 'data_fim', 'cabeca_idx': 'data_cabeca'}
            self.df_trabalho.rename(columns=rename_map, inplace=True)
            for col in ['data_inicio', 'data_fim', 'data_cabeca']:
                if col in self.df_trabalho.columns: self.df_trabalho[col] = pd.to_datetime(self.df_trabalho[col], errors='coerce')
            if 'label_humano' not in self.df_trabalho.columns: self.df_trabalho['label_humano'] = np.nan
            return True
        except Exception as e:
            messagebox.showerror("Erro Crítico no Setup", f"Não foi possível carregar ou adaptar o dataset.\nErro: {e}")
            return False

    def carregar_proximo_padrao(self):
        indices_pendentes = self.df_trabalho[self.df_trabalho['label_humano'].isnull()].index
        if indices_pendentes.empty:
            messagebox.showinfo("Fim!", "Parabéns! Todos os padrões foram rotulados.")
            self.on_closing()
            return
        self.indice_atual = indices_pendentes[0]
        self.plotar_grafico_com_zigzag()
        self.atualizar_info_label()
        
    def _validate_hns_pattern(
        self, p0: Dict, p1: Dict, p2: Dict, p3: Dict, p4: Dict, p5: Dict, 
        tipo_padrao: str, df_historico: pd.DataFrame, avg_pivot_dist: int
    ) -> Optional[Dict[str, Any]]:
        """ Valida um candidato a OCO/OCOI usando a lógica RIGOROSA do Pine Script. """
        ombro_esq, neckline1, cabeca, neckline2, ombro_dir = p1, p2, p3, p4, p5
        
        # Validação de Estrutura Básica
        if tipo_padrao == 'OCO':
            if not (cabeca['preco'] > ombro_esq['preco'] and cabeca['preco'] > ombro_dir['preco']): return None
            if not (ombro_esq['preco'] > neckline1['preco'] and ombro_dir['preco'] > neckline2['preco']): return None
            # <--- NOVO: Regras estruturais do Pine Script
            if not (ombro_dir['preco'] < ombro_esq['preco']): return None 
            if not (p0['preco'] < neckline1['preco'] and p0['preco'] < neckline2['preco']): return None
        
        elif tipo_padrao == 'OCOI':
            if not (cabeca['preco'] < ombro_esq['preco'] and cabeca['preco'] < ombro_dir['preco']): return None
            if not (ombro_esq['preco'] < neckline1['preco'] and ombro_dir['preco'] < neckline2['preco']): return None
            # <--- NOVO: Regras estruturais do Pine Script
            if not (ombro_dir['preco'] > ombro_esq['preco']): return None
            if not (p0['preco'] > neckline1['preco'] and p0['preco'] > neckline2['preco']): return None
            
        # Validação Geométrica (com tolerâncias rigorosas)
        altura_cabeca = abs(cabeca['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
        altura_ombro_esq = abs(ombro_esq['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
        altura_ombro_dir = abs(ombro_dir['preco'] - np.mean([neckline1['preco'], neckline2['preco']]))
        
        if altura_ombro_esq == 0 or altura_cabeca == 0: return None

        if altura_cabeca / altura_ombro_esq < Config.HEAD_SIGNIFICANCE_RATIO: return None
        if abs(altura_ombro_esq - altura_ombro_dir) > altura_cabeca * Config.SHOULDER_SYMMETRY_TOLERANCE: return None
        if abs(neckline1['preco'] - neckline2['preco']) > altura_ombro_esq * Config.NECKLINE_FLATNESS_TOLERANCE: return None
        
        # Validação de Contexto (lookback)
        if not self.is_head_extreme(df_historico, cabeca, avg_pivot_dist): return None
        
        return {'padrao_tipo': tipo_padrao,'ombro1_idx': ombro_esq['idx'], 'ombro1_preco': ombro_esq['preco'],'neckline1_idx': neckline1['idx'], 'neckline1_preco': neckline1['preco'],'cabeca_idx': cabeca['idx'], 'cabeca_preco': cabeca['preco'],'neckline2_idx': neckline2['idx'], 'neckline2_preco': neckline2['preco'],'ombro2_idx': ombro_dir['idx'], 'ombro2_preco': ombro_dir['preco'],}

    # O resto do código permanece o mesmo, pois as mudanças lógicas estão contidas em Config e _validate_hns_pattern
    def plotar_grafico_com_zigzag(self):
        for widget in self.frame_grafico.winfo_children(): widget.destroy()
        padrao_info = self.df_trabalho.loc[self.indice_atual]
        ticker, intervalo = padrao_info['ticker'], padrao_info['intervalo']
        
        df_full = None
        for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
            try:
                periodo_busca = '5y'
                if 'm' in intervalo: periodo_busca = '60d'
                elif 'h' in intervalo: periodo_busca = '2y'
                print(f"Buscando dados para {ticker} ({intervalo})... Tentativa {tentativa + 1}/{Config.MAX_DOWNLOAD_TENTATIVAS}")
                df_full = yf.download(tickers=ticker, period=periodo_busca, interval=intervalo, auto_adjust=True, progress=False)
                if not df_full.empty:
                    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
                    df_full.index = df_full.index.tz_localize(None)
                    df_full.columns = [col.lower() for col in df_full.columns]
                    break
                else: raise ValueError("Download retornou um DataFrame vazio.")
            except Exception as e:
                print(f"Falha na tentativa {tentativa + 1}: {e}")
                if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1: time.sleep(Config.RETRY_DELAY_SEGUNDOS)
                else:
                    self.marcar_e_avancar(-1)
                    return
        
        data_inicio, data_fim = padrao_info['data_inicio'], padrao_info['data_fim']
        if pd.isna(data_inicio) or pd.isna(data_fim):
            self.marcar_e_avancar(-1)
            return
            
        duracao = data_fim - data_inicio
        buffer = max(pd.Timedelta(days=5), duracao * 1.5)
        df_view = df_full.loc[data_inicio - buffer : data_fim + buffer].copy()

        if df_view.empty:
            self.marcar_e_avancar(-1)
            return

        params = Config.TIMEFRAME_PARAMS.get(intervalo, Config.TIMEFRAME_PARAMS['default'])
        pivots = self._calcular_zigzag_para_plot(df_view, params['deviation'], params['min_distance'])
        zigzag_line = self._preparar_zigzag_plot(pivots, df_view.index)
        ad_plot = [mpf.make_addplot(zigzag_line, color='blue', width=1.2)] if not zigzag_line.empty else []

        fig, axlist = mpf.plot(df_view, type='candle', style='charles', returnfig=True, figsize=(12, 8), addplot=ad_plot, title=f"{ticker} ({intervalo}) - Padrão {self.indice_atual}", warn_too_much_data=10000)
        ax = axlist[0]
        start_pos = df_view.index.get_indexer([data_inicio], method='nearest')[0]
        end_pos = df_view.index.get_indexer([data_fim], method='nearest')[0]
        ax.axvspan(start_pos, end_pos, color='yellow', alpha=0.2)
        if pd.notna(padrao_info['data_cabeca']):
            head_pos = df_view.index.get_indexer([padrao_info['data_cabeca']], method='nearest')[0]
            ax.axvline(x=head_pos, color='dodgerblue', linestyle='--', linewidth=1.2)
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def is_head_extreme(self, df: pd.DataFrame, head_pivot: Dict, avg_pivot_dist_days: int) -> bool:
        lookback_period = int(avg_pivot_dist_days * Config.HEAD_EXTREME_LOOKBACK_FACTOR)
        if lookback_period <= 0: return True
        start_date = head_pivot['idx'] - pd.Timedelta(days=lookback_period)
        end_date = head_pivot['idx'] + pd.Timedelta(days=lookback_period)
        context_df = df.loc[start_date:end_date]
        if context_df.empty: return True
        if head_pivot['tipo'] == 'PICO':
            return head_pivot['preco'] >= context_df['high'].max()
        else:
            return head_pivot['preco'] <= context_df['low'].min()
            
    def identificar_padroes_hns(self, pivots: List[Dict[str, Any]], df_historico: pd.DataFrame) -> List[Dict[str, Any]]:
        padroes_encontrados = []
        n = len(pivots)
        if n < 7: return []
        datas_pivots = [p['idx'] for p in pivots]
        avg_pivot_dist_days = np.mean([(datas_pivots[i] - datas_pivots[i-1]).days for i in range(1, n)]) if n > 1 else 0
        for i in range(n - 6):
            janela = pivots[i:i+7]
            p0, p1, p2, p3, p4, p5, p6 = janela
            tipo_padrao = None
            if all(p['tipo'] == t for p, t in zip(janela, ['VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE'])): tipo_padrao = 'OCO'
            elif all(p['tipo'] == t for p, t in zip(janela, ['PICO', 'VALE', 'PICO', 'VALE', 'PICO', 'VALE', 'PICO'])): tipo_padrao = 'OCOI'
            if tipo_padrao:
                # Agora o validador principal é chamado a partir daqui
                dados_padrao = self._validate_hns_pattern(p0, p1, p2, p3, p4, p5, tipo_padrao, df_historico, avg_pivot_dist_days)
                if dados_padrao:
                    padroes_encontrados.append(dados_padrao)
        return padroes_encontrados
        
    def _calcular_zigzag_para_plot(self, df, deviation_percent, min_distance_bars):
        if len(df) < 2: return []
        pivots = [{'idx': df.index[0], 'preco': df['low'].iloc[0], 'tipo': 'VALE'}]
        trend, peak, valley = 'UP', {'preco': -np.inf}, {'preco': np.inf}
        for i in range(1, len(df)):
            row, idx = df.iloc[i], df.index[i]
            if trend == 'UP':
                if row['high'] > peak['preco']: peak = {'preco': row['high'], 'idx': idx}
                if peak['preco'] > 0 and (peak['preco'] - row['low']) / peak['preco'] > deviation_percent / 100:
                    pivots.append({'idx': peak['idx'], 'preco': peak['preco'], 'tipo': 'PICO'})
                    trend, valley = 'DOWN', {'preco': row['low'], 'idx': idx}
            elif trend == 'DOWN':
                if row['low'] < valley['preco']: valley = {'preco': row['low'], 'idx': idx}
                if valley['preco'] > 0 and (row['high'] - valley['preco']) / valley['preco'] > deviation_percent / 100:
                    pivots.append({'idx': valley['idx'], 'preco': valley['preco'], 'tipo': 'VALE'})
                    trend, peak = 'UP', {'preco': row['high'], 'idx': idx}
        if not pivots: return []
        pivots_filtrados = [pivots[0]]
        for i in range(1, len(pivots)):
            try:
                dist = abs(df.index.get_loc(pivots[i]['idx']) - df.index.get_loc(pivots_filtrados[-1]['idx']))
                if pivots[i]['tipo'] != pivots_filtrados[-1]['tipo'] and dist >= min_distance_bars: pivots_filtrados.append(pivots[i])
            except KeyError: continue
        return pivots_filtrados

    def _preparar_zigzag_plot(self, pivots, df_index):
        if not pivots: return pd.Series(np.nan, index=df_index)
        zigzag_points = pd.Series(np.nan, index=df_index)
        for p in pivots:
            if p['idx'] in zigzag_points.index: zigzag_points.loc[p['idx']] = p['preco']
        return zigzag_points.interpolate(method='linear')

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in ['a', 'r']: self.marcar_e_avancar(1 if key == 'a' else 0)
        elif key == 'q': self.on_closing()

    def marcar_e_avancar(self, label):
        self.df_trabalho.loc[self.indice_atual, 'label_humano'] = label
        self.df_trabalho.to_csv(self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
        self.carregar_proximo_padrao()

    def on_closing(self):
        print("Saindo...")
        plt.close('all')
        self.destroy()

    def atualizar_info_label(self):
        total = len(self.df_trabalho)
        feitos = self.df_trabalho['label_humano'].notna().sum()
        padrao = self.df_trabalho.loc[self.indice_atual]
        info_text = (f"Progresso: {feitos}/{total} | Padrão Índice: {self.indice_atual}\n"
                     f"Ativo: {padrao['ticker']} ({padrao['intervalo']}) | Tipo: {padrao['tipo_padrao']}")
        self.info_label.config(text=info_text)

if __name__ == '__main__':
    # Modifiquei o script do gerador de dados para se chamar 'gerador_hns.py'
    # Se você ainda não o rodou, execute-o primeiro para criar o arquivo de entrada.
    if not os.path.exists(Config.ARQUIVO_ENTRADA):
        print(f"Erro: Arquivo de entrada '{Config.ARQUIVO_ENTRADA}' não encontrado.")
        print("Por favor, execute o script gerador de dados primeiro.")
    else:
        app = LabelingTool(Config.ARQUIVO_ENTRADA, Config.ARQUIVO_SAIDA)
        app.mainloop()