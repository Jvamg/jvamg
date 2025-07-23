# anotador_gui.py (v10.1 - Final com Extender Barra e Sincronia Dinâmica)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import time
from typing import List, Dict, Any, Optional

class Config:
    """ Parâmetros de configuração centralizados para a GUI. """
    # --- MUDANÇA 1: REINTRODUZIDO O INTERRUPTOR PARA EXTENDER A BARRA ---
    ZIGZAG_EXTEND_TO_LAST_BAR = True

    # --- MUDANÇA 2: LISTA COMPLETA DE ESTRATÉGIAS PARA SINCRONIA PERFEITA ---
    # O anotador agora conhece todas as estratégias que o gerador usou.
    STRATEGIES = [
        {
            "name": "Oficial_HighLow",
            "price_source": "high_low",
            "params": {
                'default': {'depth': 2, 'deviation': 2.0},
                '1h':      {'depth': 3, 'deviation': 4.0},
                '4h':      {'depth': 3, 'deviation': 5.0},
                '1d':      {'depth': 5, 'deviation': 8.0}
            }
        },
        {
            "name": "Oficial_ClosePrice",
            "price_source": "close",
            "params": {
                'default': {'depth': 2, 'deviation': 2.0},
                '1h':      {'depth': 3, 'deviation': 4.0},
                '4h':      {'depth': 3, 'deviation': 5.0},
                '1d':      {'depth': 5, 'deviation': 8.0}
            }
        }
    ]
    
    ARQUIVO_ENTRADA = 'data/datasets_hns_completo/dataset_hns_consolidado_completo.csv'
    ARQUIVO_SAIDA = 'data/datasets_hns_completo/dataset_hns_consolidado_analisado.csv'
    
    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5

class LabelingTool(tk.Tk):
    # O __init__ e outras funções permanecem os mesmos...
    def __init__(self, arquivo_entrada: str, arquivo_saida: str):
        super().__init__()
        self.title(f"Ferramenta de Anotação (v10.1 - Sincronizado)")
        self.geometry("1300x900")
        self.arquivo_saida = arquivo_saida
        self.df_trabalho: Optional[pd.DataFrame] = None
        self.indice_atual: int = 0
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

    def setup_dataframe(self, arquivo_entrada: str, arquivo_saida: str) -> bool:
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
        if self.df_trabalho is None: return
        indices_pendentes = self.df_trabalho[self.df_trabalho['label_humano'].isnull()].index
        if indices_pendentes.empty:
            messagebox.showinfo("Fim!", "Parabéns! Todos os padrões foram rotulados.")
            self.on_closing()
            return
        self.indice_atual = indices_pendentes[0]
        self.plotar_grafico_com_zigzag()
        self.atualizar_info_label()
        
    def plotar_grafico_com_zigzag(self):
        for widget in self.frame_grafico.winfo_children(): widget.destroy()
        if self.df_trabalho is None: return
        
        padrao_info = self.df_trabalho.loc[self.indice_atual]
        ticker, intervalo = padrao_info['ticker'], padrao_info['intervalo']
        
        df_full = None # ... (lógica de download idêntica à v9.2)
        for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
            try:
                periodo_busca = '5y'
                if 'm' in intervalo: periodo_busca = '60d'
                elif 'h' in intervalo: periodo_busca = '2y'
                df_full = yf.download(tickers=ticker, period=periodo_busca, interval=intervalo, auto_adjust=True, progress=False)
                if not df_full.empty:
                    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
                    df_full.columns = [col.lower() for col in df_full.columns]
                    break
                else: raise ValueError("Download retornou um DataFrame vazio.")
            except Exception as e:
                if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1: time.sleep(Config.RETRY_DELAY_SEGUNDOS)
                else: self.marcar_e_avancar(-1); return
        
        df_full.index = df_full.index.tz_localize(None)
        data_inicio = pd.to_datetime(padrao_info['data_inicio']).tz_localize(None)
        data_fim = pd.to_datetime(padrao_info['data_fim']).tz_localize(None)
        
        if pd.isna(data_inicio) or pd.isna(data_fim): self.marcar_e_avancar(-1); return
            
        duracao = data_fim - data_inicio
        if 'm' in intervalo: base_buffer = pd.Timedelta(hours=12)
        elif '1h' in intervalo: base_buffer = pd.Timedelta(days=2)
        elif '4h' in intervalo: base_buffer = pd.Timedelta(days=5)
        else: base_buffer = pd.Timedelta(days=15)
        buffer = max(base_buffer, duracao * 1.5)
        
        df_view = df_full.loc[data_inicio - buffer : data_fim + buffer].copy()
        if df_view.empty: self.marcar_e_avancar(-1); return

        # --- MUDANÇA 3: LÓGICA DE VISUALIZAÇÃO 100% DINÂMICA ---
        estrategia_str = padrao_info.get('estrategia_zigzag', 'Oficial_HighLow')
        
        # Encontra a estratégia e os parâmetros corretos na Config
        strategy_config = next((s for s in Config.STRATEGIES if s["name"] in estrategia_str), Config.STRATEGIES[0])
        price_source_visual = strategy_config['price_source']
        params = strategy_config['params'].get(intervalo, strategy_config['params']['default'])
        
        print(f"Visualizando padrão da estratégia '{estrategia_str}'. Usando fonte='{price_source_visual}', depth={params['depth']}, dev={params['deviation']}%")
        
        pivots = self._calcular_zigzag_oficial_para_plot(df_view, params['depth'], params['deviation'], price_source_visual)
        
        zigzag_line = self._preparar_zigzag_plot(pivots, df_view)
        ad_plot = [mpf.make_addplot(zigzag_line, color='blue', width=1.2)] if not zigzag_line.empty else []

        fig, axlist = mpf.plot(df_view, type='candle', style='charles', returnfig=True, figsize=(12, 8), addplot=ad_plot, title=f"{ticker} ({intervalo}) - Padrão {self.indice_atual}", warn_too_much_data=10000)
        ax = axlist[0]

        start_pos = df_view.index.get_indexer([data_inicio], method='nearest')[0]
        end_pos = df_view.index.get_indexer([data_fim], method='nearest')[0]
        ax.axvspan(start_pos, end_pos, color='yellow', alpha=0.2)
        
        if pd.notna(padrao_info['data_cabeca']):
            data_cabeca_naive = pd.to_datetime(padrao_info['data_cabeca']).tz_localize(None)
            head_pos = df_view.index.get_indexer([data_cabeca_naive], method='nearest')[0]
            ax.axvline(x=head_pos, color='dodgerblue', linestyle='--', linewidth=1.2)
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _calcular_zigzag_oficial_para_plot(self, df: pd.DataFrame, depth: int, deviation_percent: float, price_source: str) -> List[Dict[str, Any]]:
        # A função em si não muda, pois já recebe os parâmetros dinamicamente
        # ... (cole a função da v9.1 aqui)
        if price_source == 'high_low': peak_series, valley_series = df['high'], df['low']
        else: peak_series, valley_series = df['close'], df['close']
        window_size = 2 * depth + 1
        rolling_max = peak_series.rolling(window=window_size, center=True, min_periods=1).max()
        rolling_min = valley_series.rolling(window=window_size, center=True, min_periods=1).min()
        candidate_peaks_df, candidate_valleys_df = df[peak_series == rolling_max], df[valley_series == rolling_min]
        candidates = []
        for idx, row in candidate_peaks_df.iterrows(): candidates.append({'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
        for idx, row in candidate_valleys_df.iterrows(): candidates.append({'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})
        candidates = sorted(list({p['idx']: p for p in candidates}.values()), key=lambda x: x['idx'])
        if len(candidates) < 2: return []
        confirmed_pivots = [candidates[0]]
        last_pivot = candidates[0]
        for i in range(1, len(candidates)):
            candidate = candidates[i]
            if candidate['tipo'] == last_pivot['tipo']:
                if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                    confirmed_pivots[-1] = candidate; last_pivot = candidate
                continue
            if last_pivot['preco'] == 0: continue
            price_dev = abs(candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
            if price_dev >= deviation_percent:
                confirmed_pivots.append(candidate); last_pivot = candidate
        return confirmed_pivots

    def _preparar_zigzag_plot(self, pivots: List[Dict[str, Any]], df_view: pd.DataFrame) -> pd.Series:
        # --- MUDANÇA 4: A LÓGICA DE EXTENDER A BARRA FOI REINTRODUZIDA ---
        if df_view.empty: return pd.Series()
        zigzag_points = pd.Series(np.nan, index=df_view.index)
        if not pivots: return zigzag_points
        for p in pivots:
            if p['idx'] in zigzag_points.index: zigzag_points.loc[p['idx']] = p['preco']
        if Config.ZIGZAG_EXTEND_TO_LAST_BAR:
            last_close, last_index = df_view['close'].iloc[-1], df_view.index[-1]
            zigzag_points.loc[last_index] = last_close
        return zigzag_points.interpolate(method='linear')

    def on_key_press(self, event: tk.Event):
        key = event.keysym.lower()
        if key in ['a', 'r']: self.marcar_e_avancar(1 if key == 'a' else 0)
        elif key == 'q': self.on_closing()

    def marcar_e_avancar(self, label: int):
        if self.df_trabalho is None: return
        self.df_trabalho.loc[self.indice_atual, 'label_humano'] = label
        self.df_trabalho.to_csv(self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
        self.carregar_proximo_padrao()

    def on_closing(self):
        print("Saindo...")
        plt.close('all')
        self.destroy()

    def atualizar_info_label(self):
        if self.df_trabalho is None: return
        total = len(self.df_trabalho)
        feitos = self.df_trabalho['label_humano'].notna().sum()
        padrao = self.df_trabalho.loc[self.indice_atual]
        info_text = (f"Progresso: {feitos}/{total} | Padrão Índice: {self.indice_atual}\n"
                     f"Ativo: {padrao['ticker']} ({padrao['intervalo']}) | Tipo: {padrao['tipo_padrao']}\n"
                     f"Estratégia: {padrao.get('estrategia_zigzag', 'N/A')}")
        self.info_label.config(text=info_text)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(Config.ARQUIVO_ENTRADA), exist_ok=True)
    os.makedirs(os.path.dirname(Config.ARQUIVO_SAIDA), exist_ok=True)
    if not os.path.exists(Config.ARQUIVO_ENTRADA):
        messagebox.showerror("Erro de Arquivo", f"Arquivo de entrada não encontrado!\n{Config.ARQUIVO_ENTRADA}\n\nPor favor, execute o script gerador (v14+) primeiro.")
    else:
        app = LabelingTool(Config.ARQUIVO_ENTRADA, Config.ARQUIVO_SAIDA)
        app.mainloop()