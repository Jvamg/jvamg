# anotador_gui.py (v13.2 - Controle de Densidade de Candles)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import pandas_ta as ta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import time
from typing import List, Dict, Any, Optional

class Config:
    """ Parâmetros de configuração centralizados para a GUI. """
    ZIGZAG_EXTEND_TO_LAST_BAR = True

    # <<< MENTOR - NOVO >>>
    # Esta é a nossa nova "regra de ouro" para a visualização.
    # Limitamos o número máximo de candles na tela para garantir que o gráfico
    # seja sempre legível e que o tipo 'candle' possa ser usado sem problemas.
    MAX_CANDLES_IN_VIEW = 450 # Um bom número para telas modernas.

    # <<< MENTOR - REMOVIDO >>>
    # A configuração PLOT_BUFFER_CONFIG foi removida. A nova abordagem baseada
    # em contagem de candles é mais direta e eficaz, tornando a antiga obsoleta.

    ZIGZAG_STRATEGIES = {
        'micro_structure': {
            '1m':  {'depth': 3, 'deviation': 0.2}, 
            '5m':  {'depth': 4, 'deviation': 0.6},
            '15m': {'depth': 5, 'deviation': 0.8}
        },
        'day_trade': {
            '15m': {'depth': 7,  'deviation': 1.0}, 
            '1h':  {'depth': 8,  'deviation': 1.5},
            '4h':  {'depth': 10, 'deviation': 2.5}
        },
        # ... resto das estratégias sem alteração
        'swing_structure': {
            '4h': {'depth': 12, 'deviation': 4.0}, 
            '1d': {'depth': 8,  'deviation': 6.0},
            '1wk': {'depth': 5,  'deviation': 8.0}
        },
        'macro_trend': {
            '1d': {'depth': 15, 'deviation': 10.0}, 
            '1wk': {'depth': 10, 'deviation': 15.0},
            '1mo': {'depth': 5,  'deviation': 20.0}
        },
        'major_structure': {
            '4h': {'depth': 15, 'deviation': 8.0}, 
            '1d': {'depth': 20, 'deviation': 10.0},
            '1wk': {'depth': 18, 'deviation': 15.0}
        }
    }

    ARQUIVO_ENTRADA = 'data/datasets/datasets_hns_by_strategy/dataset_hns_by_strategy_final.csv'
    ARQUIVO_SAIDA = 'data/datasets/datasets_hns_by_strategy/dataset_hns_labeled_final.csv'

    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5
    # Lookback generoso para garantir que tenhamos dados suficientes para o cálculo do buffer
    ZIGZAG_LOOKBACK_DAYS_DEFAULT = 400 
    ZIGZAG_LOOKBACK_DAYS_MINUTE = 5 # Aumentado para ter mais dados de contexto se necessário

class LabelingTool(tk.Tk):
    # __init__ e outras funções que não foram alteradas permanecem aqui...
    def __init__(self, arquivo_entrada: str, arquivo_saida: str):
        super().__init__()
        self.title("Ferramenta de Anotação (v13.2 - Zoom por Densidade)")
        # ... resto do __init__ sem alterações ...
        self.geometry("1300x950")
        self.arquivo_saida = arquivo_saida
        self.df_trabalho: Optional[pd.DataFrame] = None
        self.indice_atual: int = 0
        self.fig: Optional[plt.Figure] = None
        self.regras_map = {
            'valid_extremo_cabeca': 'Cabeça é o Extremo', 'valid_contexto_cabeca': 'Contexto Relevante',
            'valid_divergencia_rsi': 'Divergência RSI', 'valid_divergencia_macd': 'Divergência MACD',
            'valid_proeminencia_cabeca': 'Cabeça Proeminente', 'valid_simetria_ombros': 'Simetria de Ombros',
            'valid_neckline_plana': 'Neckline Plana', 'valid_ombro_direito_fraco': 'Ombro Direito Fraco',
            'valid_base_tendencia': 'Estrutura de Tendência', 'valid_perfil_volume': 'Perfil de Volume'
        }
        if not self.setup_dataframe(arquivo_entrada, arquivo_saida):
            self.destroy()
            return
        self._setup_ui()
        self.bind('<Key>', self.on_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.carregar_proximo_padrao()

    # ... setup_dataframe e _setup_ui sem alterações ...

    def plotar_grafico_com_zigzag(self):
        # ... (limpeza inicial do gráfico sem alterações) ...
        if self.fig is not None: plt.close(self.fig)
        for widget in self.frame_grafico.winfo_children(): widget.destroy()
        if self.df_trabalho is None: return

        padrao_info = self.df_trabalho.loc[self.indice_atual]
        # ... (obtenção de ticker, intervalo, estrategia, params sem alterações) ...
        ticker, intervalo = padrao_info['ticker'], padrao_info['intervalo']
        estrategia = padrao_info.get('estrategia_zigzag')
        if not estrategia or not isinstance(estrategia, str):
            messagebox.showerror("Erro de Dados", f"Padrão {self.indice_atual} não possui uma 'estrategia_zigzag' válida.")
            self.marcar_e_avancar(-1); return

        params = Config.ZIGZAG_STRATEGIES.get(estrategia, {}).get(intervalo)
        if not params:
            messagebox.showerror("Erro de Configuração", f"Não foi encontrada configuração de ZigZag para:\nEstratégia: {estrategia}\nIntervalo: {intervalo}")
            self.marcar_e_avancar(-1); return

        data_inicio_padrao = pd.to_datetime(padrao_info['data_inicio']).tz_localize(None)
        data_fim_padrao = pd.to_datetime(padrao_info['data_fim']).tz_localize(None)

        if pd.isna(data_inicio_padrao) or pd.isna(data_fim_padrao):
            self.marcar_e_avancar(-1); return

        # <<< MENTOR - ALTERADO >>>
        # A lógica de download agora precisa de um lookback maior para garantir que teremos
        # candles suficientes para a nova lógica de buffer posicional.
        lookback_days = (
            Config.ZIGZAG_LOOKBACK_DAYS_MINUTE if 'm' in intervalo
            else Config.ZIGZAG_LOOKBACK_DAYS_DEFAULT
        )
        # Fazemos um download amplo primeiro. O "zoom" será feito depois, com o pandas.
        download_start_date = data_inicio_padrao - pd.Timedelta(days=lookback_days)
        download_end_date = data_fim_padrao + pd.Timedelta(days=1)

        df_full = None
        for _ in range(Config.MAX_DOWNLOAD_TENTATIVAS):
            # ... (lógica de download yfinance sem alterações) ...
            try:
                df_full = yf.download(
                    tickers=ticker, start=download_start_date, end=download_end_date,
                    interval=intervalo, auto_adjust=True, progress=False
                )
                if not df_full.empty:
                    df_full.columns = [col.lower() for col in (df_full.columns.get_level_values(0) if isinstance(df_full.columns, pd.MultiIndex) else df_full.columns)]
                    break
                else: raise ValueError("Download retornou um DataFrame vazio.")
            except Exception:
                time.sleep(Config.RETRY_DELAY_SEGUNDOS)
        else:
            self.marcar_e_avancar(-1); return
            
        df_full.index = df_full.index.tz_localize(None)
        
        # <<< MENTOR - LÓGICA DE ZOOM COMPLETAMENTE REFEITA >>>
        try:
            # 1. Encontrar a POSIÇÃO (índice inteiro) do início e fim do padrão no DataFrame completo.
            start_pos = df_full.index.get_indexer([data_inicio_padrao], method='nearest')[0]
            end_pos = df_full.index.get_indexer([data_fim_padrao], method='nearest')[0]

            # 2. Calcular quantos candles o padrão ocupa.
            pattern_candle_count = end_pos - start_pos + 1
            
            # 3. Calcular o buffer em NÚMERO DE CANDLES.
            # Se o padrão já for maior que o nosso limite, não adicionamos buffer.
            buffer_candles = 0
            if pattern_candle_count < Config.MAX_CANDLES_IN_VIEW:
                buffer_candles = (Config.MAX_CANDLES_IN_VIEW - pattern_candle_count) // 2

            # 4. Calcular as posições de início e fim da nossa janela de visualização.
            view_start_pos = max(0, start_pos - buffer_candles)
            view_end_pos = min(len(df_full), end_pos + buffer_candles)
            
            # 5. Fatiar o DataFrame usando .iloc para criar nossa janela de visualização final.
            df_view = df_full.iloc[view_start_pos:view_end_pos].copy()

        except IndexError:
            # Se as datas do padrão não forem encontradas nos dados baixados, marcamos como erro.
            self.marcar_e_avancar(-1); return
        
        if df_view.empty:
            self.marcar_e_avancar(-1); return
        # --- FIM DA NOVA LÓGICA DE ZOOM ---

        pivots_visuais = self._calcular_zigzag(df_full, params['depth'], params['deviation'])
        zigzag_line = self._preparar_zigzag_plot(pivots_visuais, df_view)
        
        df_view.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_view['rsi_high'] = ta.rsi(df_view['high'], length=14)
        df_view['rsi_low'] = ta.rsi(df_view['low'], length=14)
        
        ad_plots = [mpf.make_addplot(zigzag_line, color='dodgerblue', width=1.2)]
        rsi_plot = mpf.make_addplot(df_view[['rsi_high', 'rsi_low']], panel=1, ylabel='RSI')
        macd_hist = mpf.make_addplot(df_view['MACDh_12_26_9'], type='bar', panel=2, ylabel='MACD Hist')
        macd_lines = mpf.make_addplot(df_view[['MACD_12_26_9', 'MACDs_12_26_9']], panel=2)
        ad_plots.extend([rsi_plot, macd_lines, macd_hist])
        
        # Agora podemos usar 'candle' com confiança para todos os gráficos!
        self.fig, axlist = mpf.plot(df_view, type='candle', style='yahoo', returnfig=True,
                                    figsize=(13, 10), addplot=ad_plots,
                                    panel_ratios=(10, 2, 2, 2),
                                    title=f"{ticker} ({intervalo}) - Padrão {self.indice_atual}",
                                    volume=True, volume_panel=3,
                                    warn_too_much_data=Config.MAX_CANDLES_IN_VIEW + 50) # Avisa só se exceder muito nosso limite

        # A lógica de highlight agora usa o df_view
        ax_price = axlist[0]
        # Precisamos recalcular as posições relativas ao NOVO df_view
        start_pos_view = df_view.index.get_indexer([data_inicio_padrao], method='nearest')[0]
        end_pos_view = df_view.index.get_indexer([data_fim_padrao], method='nearest')[0]
        ax_price.axvspan(start_pos_view, end_pos_view, color='yellow', alpha=0.2)

        if pd.notna(padrao_info.get('data_cabeca')):
            data_cabeca_naive = pd.to_datetime(padrao_info['data_cabeca']).tz_localize(None)
            if data_cabeca_naive in df_view.index:
                head_pos_view = df_view.index.get_indexer([data_cabeca_naive], method='nearest')[0]
                ax_price.axvline(x=head_pos_view, color='dodgerblue', linestyle='--', linewidth=1.2)

        canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ... todo o resto do código (_calcular_zigzag, on_key_press, etc.) permanece igual ...
    # (Copiar e colar o restante das funções sem alteração aqui)
    def setup_dataframe(self, arquivo_entrada: str, arquivo_saida: str) -> bool:
        try:
            if os.path.exists(arquivo_saida):
                self.df_trabalho = pd.read_csv(arquivo_saida)
            else:
                self.df_trabalho = pd.read_csv(arquivo_entrada)
            rename_map = {
                'timeframe': 'intervalo', 'strategy': 'estrategia_zigzag',
                'padrao_tipo': 'tipo_padrao', 'ombro1_idx': 'data_inicio',
                'ombro2_idx': 'data_fim', 'cabeca_idx': 'data_cabeca'
            }
            self.df_trabalho.rename(
                columns={k: v for k, v in rename_map.items() if k in self.df_trabalho.columns},
                inplace=True
            )
            for col in ['data_inicio', 'data_fim', 'data_cabeca']:
                if col in self.df_trabalho.columns:
                    self.df_trabalho[col] = pd.to_datetime(self.df_trabalho[col], errors='coerce')
            if 'label_humano' not in self.df_trabalho.columns:
                self.df_trabalho['label_humano'] = np.nan
            return True
        except FileNotFoundError:
            messagebox.showerror("Erro de Arquivo", f"Arquivo de entrada não encontrado!\n{arquivo_entrada}\n\nVerifique o caminho ou execute o gerador primeiro.")
            return False
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

    def _setup_ui(self):
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)
        self.frame_grafico = tk.Frame(m)
        m.add(self.frame_grafico, minsize=400)
        frame_controles = tk.Frame(m, height=150)
        frame_controles.pack_propagate(False)
        m.add(frame_controles, minsize=150)
        frame_controles.grid_rowconfigure(0, weight=1)
        frame_controles.grid_columnconfigure(0, weight=1)
        frame_controles.grid_columnconfigure(1, weight=2)
        frame_info = tk.Frame(frame_controles)
        frame_info.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        self.info_label = tk.Label(frame_info, text="Carregando...", font=("Segoe UI", 10), justify=tk.LEFT, anchor="nw")
        self.info_label.pack(side=tk.TOP, anchor="w", pady=(0, 10))
        self.action_label = tk.Label(frame_info, text="Teclado: [A]provar | [R]ejeitar | [Q]uit", font=("Segoe UI", 12, "bold"))
        self.action_label.pack(side=tk.BOTTOM, anchor="w")
        frame_boletim = tk.Frame(frame_controles, relief=tk.RIDGE, borderwidth=1)
        frame_boletim.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        tk.Label(frame_boletim, text="Boletim de Validação do Padrão", font=("Segoe UI", 11, "bold")).pack(pady=(5, 10))
        grid_frame = tk.Frame(frame_boletim)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=15)
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)
        self.boletim_labels = {}
        for i, (key, name) in enumerate(self.regras_map.items()):
            col_base = 0 if i < 5 else 2
            tk.Label(grid_frame, text=f"{name}:", font=("Segoe UI", 9), anchor="w").grid(row=i % 5, column=col_base, sticky="w", padx=(0, 10))
            result_label = tk.Label(grid_frame, text="...", font=("Segoe UI", 9, "bold"), anchor="w")
            result_label.grid(row=i % 5, column=col_base + 1, sticky="w")
            self.boletim_labels[key] = result_label
        self.score_label = tk.Label(frame_boletim, text="SCORE FINAL: N/A", font=("Segoe UI", 11, "bold"), fg="#1E90FF")
        self.score_label.pack(side=tk.BOTTOM, pady=5)

    def _calcular_zigzag(self, df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
        peak_series, valley_series = df['high'], df['low']
        window_size = 2 * depth + 1
        rolling_max = peak_series.rolling(window=window_size, center=True, min_periods=1).max()
        rolling_min = valley_series.rolling(window=window_size, center=True, min_periods=1).min()
        candidate_peaks_df = df[peak_series == rolling_max]
        candidate_valleys_df = df[valley_series == rolling_min]
        candidates = []
        for idx, row in candidate_peaks_df.iterrows():
            candidates.append({'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
        for idx, row in candidate_valleys_df.iterrows():
            candidates.append({'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})
        candidates = sorted(list({p['idx']: p for p in candidates}.values()), key=lambda x: x['idx'])
        if len(candidates) < 2: return []
        confirmed_pivots = [candidates[0]]
        last_pivot = candidates[0]
        for i in range(1, len(candidates)):
            candidate = candidates[i]
            if candidate['tipo'] == last_pivot['tipo']:
                if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or \
                   (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                    confirmed_pivots[-1] = last_pivot = candidate
                continue
            if last_pivot['preco'] == 0: continue
            price_dev = abs(candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
            if price_dev >= deviation_percent:
                confirmed_pivots.append(candidate)
                last_pivot = candidate
        return confirmed_pivots
    
    def _preparar_zigzag_plot(self, todos_os_pivots: List[Dict[str, Any]], df_view: pd.DataFrame) -> pd.Series:
        if df_view.empty or not todos_os_pivots:
            return pd.Series(dtype='float64', index=df_view.index)
        plot_points_dict = {}
        ponto_de_ancoragem = None
        for pivot in todos_os_pivots:
            if pivot['idx'] < df_view.index[0]:
                ponto_de_ancoragem = pivot
            else:
                break 
        if ponto_de_ancoragem:
            plot_points_dict[ponto_de_ancoragem['idx']] = ponto_de_ancoragem['preco']
        for p in todos_os_pivots:
            if p['idx'] in df_view.index:
                plot_points_dict[p['idx']] = p['preco']
        if Config.ZIGZAG_EXTEND_TO_LAST_BAR:
            if not df_view.empty:
                plot_points_dict[df_view.index[-1]] = df_view['close'].iloc[-1]
        if not plot_points_dict:
            return pd.Series(dtype='float64', index=df_view.index)
        points_series = pd.Series(plot_points_dict)
        combined_index = df_view.index.union(points_series.index)
        aligned_series = points_series.reindex(combined_index)
        interpolated_series = aligned_series.interpolate(method='linear')
        final_series = interpolated_series.reindex(df_view.index).bfill().ffill()
        return final_series
    
    def on_key_press(self, event: tk.Event):
        key = event.keysym.lower()
        if key in ['a', 'r']:
            self.marcar_e_avancar(1 if key == 'a' else 0)
        elif key == 'q':
            self.on_closing()

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
                     f"Ativo: {padrao['ticker']} ({padrao['intervalo']})\n"
                     f"Tipo: {padrao['tipo_padrao']}\n"
                     f"Estratégia: {padrao.get('estrategia_zigzag', 'N/A')}")
        self.info_label.config(text=info_text)
        score = padrao.get('score_total', 0)
        self.score_label.config(text=f"SCORE FINAL: {score:.0f} / 100")
        for key, name in self.regras_map.items():
            if key in self.boletim_labels:
                status_bool = padrao.get(key, False)
                status_text = "SIM" if status_bool else "NÃO"
                status_color = "green" if status_bool else "red"
                self.boletim_labels[key].config(text=status_text, fg=status_color)

if __name__ == '__main__':
    # ... (código do main sem alterações) ...
    output_dir = os.path.dirname(Config.ARQUIVO_SAIDA)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(Config.ARQUIVO_ENTRADA):
        messagebox.showerror(
            "Erro de Arquivo", f"Arquivo de entrada não encontrado!\n{Config.ARQUIVO_ENTRADA}\n\nPor favor, execute o gerador v20 primeiro."
        )
    else:
        app = LabelingTool(Config.ARQUIVO_ENTRADA, Config.ARQUIVO_SAIDA)
        app.mainloop()