"""GUI de rotulagem de padrões com ZigZag e indicadores (RSI/MACD).

Carrega um CSV de padrões detectados, aplica zoom por densidade de candles,
sobrepõe ZigZag e indicadores e permite rotular rapidamente (aprovar/rejeitar).
"""
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
    """Centralized configuration parameters for the GUI."""
    ZIGZAG_EXTEND_TO_LAST_BAR = True

    MAX_CANDLES_IN_VIEW = 450  # Maximum candles to keep the chart legible

    ZIGZAG_STRATEGIES = {
        # ------- SCALPING (micro-structures) ----------
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

        # ------- SWING (hours to days) ----------
        'swing_short': {
            '15m': {'depth': 8,  'deviation': 2.0},
            '1h':  {'depth': 10, 'deviation': 2.8},
            '4h':  {'depth': 12, 'deviation': 4.0}
        },
        'swing_medium': {
            '1h':  {'depth': 10, 'deviation': 3.2},
            '4h':  {'depth': 12, 'deviation': 4.8},
            '1d':  {'depth': 10, 'deviation': 6.0}
        },
        'swing_long': {
            '4h':  {'depth': 13, 'deviation': 5.0},
            '1d':  {'depth': 12, 'deviation': 7.0},
            '1wk': {'depth': 10, 'deviation': 8.5}
        },

        # ------- POSITION / MACRO ----------
        'position_trend': {
            '1d':  {'depth': 15, 'deviation': 9.0},
            '1wk': {'depth': 12, 'deviation': 12.0},
            '1mo': {'depth': 8,  'deviation': 15.0}
        },
        'macro_trend_primary': {
            '1wk': {'depth': 16, 'deviation': 13.0},
            '1mo': {'depth': 10, 'deviation': 18.0}
        }
    }

    ARQUIVO_ENTRADA = 'data/datasets/patterns_by_strategy/dataset_patterns_final.csv'
    ARQUIVO_SAIDA = 'data/datasets/patterns_by_strategy/dataset_patterns_labeled.csv'

    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5
    ZIGZAG_LOOKBACK_DAYS_DEFAULT = 400  # Generous lookback for zoom calculations
    ZIGZAG_LOOKBACK_DAYS_MINUTE = 5     # Extra context for minute intervals


class LabelingTool(tk.Tk):
    """Janela principal para rotulagem de padrões e visualização."""

    def __init__(self, arquivo_entrada: str, arquivo_saida: str):
        super().__init__()
        self.title("Ferramenta de Anotação (v13.2 - Zoom por Densidade)")
        # initialize window and state
        self.geometry("1300x950")
        self.arquivo_saida = arquivo_saida
        self.df_trabalho: Optional[pd.DataFrame] = None
        self.indice_atual: int = 0
        self.fig: Optional[plt.Figure] = None
        # rule maps by pattern type
        self.regras_map_hns: Dict[str, str] = {
            'valid_divergencia_rsi': 'Divergência RSI',
            'valid_divergencia_macd': 'Divergência MACD',
            'valid_proeminencia_cabeca': 'Cabeça Proeminente',
            'valid_ombro_direito_fraco': 'Ombro Direito Fraco',
            'valid_perfil_volume': 'Perfil de Volume',
        }
        self.regras_map_dtb: Dict[str, str] = {
            'valid_perfil_volume_decrescente': 'Perfil de Volume Decrescente',
            'valid_divergencia_obv': 'Divergência OBV',
            'valid_divergencia_rsi': 'Divergência RSI',
            'valid_divergencia_macd': 'Divergência MACD',
        }
        self.max_rule_slots: int = len(self.regras_map_hns)
        if not self.setup_dataframe(arquivo_entrada, arquivo_saida):
            self.destroy()
            return
        self._setup_ui()
        self.bind('<Key>', self.on_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.carregar_proximo_padrao()

    # setup and UI helpers

    def plotar_grafico_com_zigzag(self):
        """Baixa dados, aplica zoom por densidade e plota ZigZag/indicadores."""
        if self.fig is not None:
            plt.close(self.fig)
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()
        if self.df_trabalho is None:
            return

        padrao_info = self.df_trabalho.loc[self.indice_atual]
        # extract ticker/interval/strategy and parameters
        ticker, intervalo = padrao_info['ticker'], padrao_info['intervalo']
        estrategia = padrao_info.get('estrategia_zigzag')
        if not estrategia or not isinstance(estrategia, str):
            messagebox.showerror(
                "Erro de Dados", f"Padrão {self.indice_atual} não possui uma 'estrategia_zigzag' válida.")
            self.marcar_e_avancar(-1)
            return

        params = Config.ZIGZAG_STRATEGIES.get(estrategia, {}).get(intervalo)
        if not params:
            messagebox.showerror(
                "Erro de Configuração", f"Não foi encontrada configuração de ZigZag para:\nEstratégia: {estrategia}\nIntervalo: {intervalo}")
            self.marcar_e_avancar(-1)
            return

        data_inicio_padrao = pd.to_datetime(
            padrao_info['data_inicio']).tz_localize(None)
        # prefer retest date if available; else use right-shoulder end
        data_retest_padrao = padrao_info.get('data_retest')
        if pd.notna(data_retest_padrao):
            data_fim_padrao = pd.to_datetime(
                data_retest_padrao).tz_localize(None)
        else:
            data_fim_padrao = pd.to_datetime(
                padrao_info['data_fim']).tz_localize(None)

        if pd.isna(data_inicio_padrao) or pd.isna(data_fim_padrao):
            self.marcar_e_avancar(-1)
            return

        # compute lookback to ensure enough data for zoom window
        lookback_days = (
            Config.ZIGZAG_LOOKBACK_DAYS_MINUTE if intervalo.endswith('m') and not intervalo.endswith('mo')
            else Config.ZIGZAG_LOOKBACK_DAYS_DEFAULT
        )
        # wide download first; zoom via pandas afterwards

        download_start_date = data_inicio_padrao - \
            pd.Timedelta(days=lookback_days)

        # minimal delta to include retest bar without overshooting too much
        if intervalo.endswith('mo'):
            interval_delta = pd.Timedelta(days=31)
        elif intervalo.endswith('wk'):
            interval_delta = pd.Timedelta(weeks=1)
        elif intervalo.endswith('d'):
            interval_delta = pd.Timedelta(days=1)
        elif intervalo.endswith('h'):
            interval_delta = pd.Timedelta(
                hours=int(''.join(filter(str.isdigit, intervalo))) or 1)
        elif intervalo.endswith('m'):
            interval_delta = pd.Timedelta(
                minutes=int(''.join(filter(str.isdigit, intervalo))) or 1)
        else:
            interval_delta = pd.Timedelta(days=1)
        download_end_date = data_fim_padrao + interval_delta

        df_full = None
        for _ in range(Config.MAX_DOWNLOAD_TENTATIVAS):
            # yfinance download with retries
            try:
                df_full = yf.download(
                    tickers=ticker, start=download_start_date, end=download_end_date,
                    interval=intervalo, auto_adjust=True, progress=False
                )
                if not df_full.empty:
                    df_full.columns = [col.lower() for col in (df_full.columns.get_level_values(
                        0) if isinstance(df_full.columns, pd.MultiIndex) else df_full.columns)]
                    break
                else:
                    raise ValueError("Download retornou um DataFrame vazio.")
            except Exception:
                time.sleep(Config.RETRY_DELAY_SEGUNDOS)
        else:
            self.marcar_e_avancar(-1)
            return

        df_full.index = df_full.index.tz_localize(None)

        # build zoom window by candle density
        try:
            # locate start/end positions nearest to pattern dates
            start_pos = df_full.index.get_indexer(
                [data_inicio_padrao], method='nearest')[0]
            end_pos = df_full.index.get_indexer(
                [data_fim_padrao], method='nearest')[0]

            # number of candles in the pattern
            pattern_candle_count = end_pos - start_pos + 1

            # compute buffer in candles (cap by MAX_CANDLES_IN_VIEW)
            buffer_candles = 0
            if pattern_candle_count < Config.MAX_CANDLES_IN_VIEW:
                buffer_candles = (Config.MAX_CANDLES_IN_VIEW -
                                  pattern_candle_count) // 2

            # derive final view window positions
            view_start_pos = max(0, start_pos - buffer_candles)
            view_end_pos = min(len(df_full), end_pos + buffer_candles)

            # slice view window
            df_view = df_full.iloc[view_start_pos:view_end_pos].copy()

        except IndexError:
            # if pattern dates are not found, skip to next
            self.marcar_e_avancar(-1)
            return

        if df_view.empty:
            self.marcar_e_avancar(-1)
            return
        # end zoom logic

        pivots_visuais = self._calcular_zigzag(
            df_full, params['depth'], params['deviation'])
        zigzag_line = self._preparar_zigzag_plot(pivots_visuais, df_view)

        df_view.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_view['rsi_high'] = ta.rsi(df_view['high'], length=14)
        df_view['rsi_low'] = ta.rsi(df_view['low'], length=14)

        ad_plots = [mpf.make_addplot(
            zigzag_line, color='dodgerblue', width=1.2)]
        rsi_plot = mpf.make_addplot(
            df_view[['rsi_high', 'rsi_low']], panel=1, ylabel='RSI')
        macd_hist = mpf.make_addplot(
            df_view['MACDh_12_26_9'], type='bar', panel=2, ylabel='MACD Hist')
        macd_lines = mpf.make_addplot(
            df_view[['MACD_12_26_9', 'MACDs_12_26_9']], panel=2)
        ad_plots.extend([rsi_plot, macd_lines, macd_hist])

        # render candlestick with overlays
        self.fig, axlist = mpf.plot(df_view, type='candle', style='yahoo', returnfig=True,
                                    figsize=(13, 10), addplot=ad_plots,
                                    panel_ratios=(10, 2, 2, 2),
                                    title=f"{ticker} ({intervalo}) - Padrão {self.indice_atual}",
                                    volume=True, volume_panel=3,
                                    warn_too_much_data=Config.MAX_CANDLES_IN_VIEW + 50)  # Avisa só se exceder muito nosso limite

        # highlight uses df_view positions
        ax_price = axlist[0]
        # recompute positions relative to df_view
        start_pos_view = df_view.index.get_indexer(
            [data_inicio_padrao], method='nearest')[0]
        end_pos_view = df_view.index.get_indexer(
            [data_fim_padrao], method='nearest')[0]
        ax_price.axvspan(start_pos_view, end_pos_view,
                         color='yellow', alpha=0.2)

        if pd.notna(padrao_info.get('data_cabeca')):
            data_cabeca_naive = pd.to_datetime(
                padrao_info['data_cabeca']).tz_localize(None)
            if data_cabeca_naive in df_view.index:
                head_pos_view = df_view.index.get_indexer(
                    [data_cabeca_naive], method='nearest')[0]
                ax_price.axvline(
                    x=head_pos_view, color='dodgerblue', linestyle='--', linewidth=1.2)

        canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # remaining helpers
    def setup_dataframe(self, arquivo_entrada: str, arquivo_saida: str) -> bool:
        """Carrega o CSV e prepara colunas: datas, rótulos e renomeações."""
        try:
            # load dataset (prioritize output if exists)
            if os.path.exists(arquivo_saida):
                df = pd.read_csv(arquivo_saida)
            else:
                df = pd.read_csv(arquivo_entrada)

            # 1) rename common columns
            df.rename(columns={
                'timeframe': 'intervalo',
                'strategy': 'estrategia_zigzag',
            }, inplace=True)

            # ensure 'tipo_padrao' exists
            if 'tipo_padrao' not in df.columns and 'padrao_tipo' in df.columns:
                df['tipo_padrao'] = df['padrao_tipo']

            num_rows = len(df)

            # helpers to get existing series or NaT
            def series_or_nat(col_name: str) -> pd.Series:
                if col_name in df.columns:
                    return df[col_name]
                return pd.Series([pd.NaT] * num_rows, index=df.index)

            # 2) create data_inicio/data_fim conditionally by pattern type
            tipo_series = df['tipo_padrao'] if 'tipo_padrao' in df.columns else pd.Series(
                [None] * num_rows, index=df.index)
            tipo_upper = tipo_series.astype(str).str.upper()
            is_hns = tipo_upper.isin(['OCO', 'OCOI'])

            ombro1 = series_or_nat('ombro1_idx')
            ombro2 = series_or_nat('ombro2_idx')
            p0 = series_or_nat('p0_idx')
            p3 = series_or_nat('p3_idx')

            df['data_inicio'] = np.where(is_hns, ombro1, p0)
            df['data_fim'] = np.where(is_hns, ombro2, p3)

            # 3) create data_cabeca/data_retest safely
            df['data_cabeca'] = series_or_nat('cabeca_idx')
            # Map retest date by pattern type: H&S uses p6, DT/DB uses p4
            retest_hns = series_or_nat('retest_p6_idx')
            retest_dtb = series_or_nat('p4_idx')
            df['data_retest'] = np.where(is_hns, retest_hns, retest_dtb)

            # convert to datetime
            for col in ['data_inicio', 'data_fim', 'data_cabeca', 'data_retest']:
                df[col] = pd.to_datetime(df[col], errors='coerce')

            # ensure manual label column exists
            if 'label_humano' not in df.columns:
                df['label_humano'] = np.nan

            self.df_trabalho = df
            return True
        except FileNotFoundError:
            messagebox.showerror(
                "Erro de Arquivo", f"Arquivo de entrada não encontrado!\n{arquivo_entrada}\n\nVerifique o caminho ou execute o gerador primeiro.")
            return False
        except Exception as e:
            messagebox.showerror(
                "Erro Crítico no Setup", f"Não foi possível carregar ou adaptar o dataset.\nErro: {e}")
            return False

    def carregar_proximo_padrao(self):
        """Seleciona o próximo índice sem rótulo e atualiza a visualização."""
        if self.df_trabalho is None:
            return
        indices_pendentes = self.df_trabalho[self.df_trabalho['label_humano'].isnull(
        )].index
        if indices_pendentes.empty:
            messagebox.showinfo(
                "Fim!", "Parabéns! Todos os padrões foram rotulados.")
            self.on_closing()
            return
        self.indice_atual = indices_pendentes[0]
        self.plotar_grafico_com_zigzag()
        self.atualizar_info_label()

    def _setup_ui(self):
        """Constroi frames do gráfico, painel de infos e boletim de regras."""
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
        self.info_label = tk.Label(frame_info, text="Carregando...", font=(
            "Segoe UI", 10), justify=tk.LEFT, anchor="nw")
        self.info_label.pack(side=tk.TOP, anchor="w", pady=(0, 10))
        self.action_label = tk.Label(
            frame_info, text="Teclado: [A]provar | [R]ejeitar | [Q]uit", font=("Segoe UI", 12, "bold"))
        self.action_label.pack(side=tk.BOTTOM, anchor="w")
        frame_boletim = tk.Frame(
            frame_controles, relief=tk.RIDGE, borderwidth=1)
        frame_boletim.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        tk.Label(frame_boletim, text="Boletim de Validação do Padrão",
                 font=("Segoe UI", 11, "bold")).pack(pady=(5, 10))
        grid_frame = tk.Frame(frame_boletim)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=15)
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)
        # fixed slots (based on H&S set), dynamic labels
        self.boletim_name_labels: List[tk.Label] = []
        self.boletim_value_labels: List[tk.Label] = []
        for i, (_, name) in enumerate(self.regras_map_hns.items()):
            col_base = 0 if i < 5 else 2
            name_label = tk.Label(
                grid_frame, text=f"{name}:", font=("Segoe UI", 9), anchor="w")
            name_label.grid(row=i % 5, column=col_base,
                            sticky="w", padx=(0, 10))
            result_label = tk.Label(grid_frame, text="...", font=(
                "Segoe UI", 9, "bold"), anchor="w")
            result_label.grid(row=i % 5, column=col_base + 1, sticky="w")
            self.boletim_name_labels.append(name_label)
            self.boletim_value_labels.append(result_label)
        self.score_label = tk.Label(
            frame_boletim, text="SCORE FINAL: N/A", font=("Segoe UI", 11, "bold"), fg="#1E90FF")
        self.score_label.pack(side=tk.BOTTOM, pady=5)

    def _calcular_zigzag(self, df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
        """Calcula pivôs visuais para sobreposição no gráfico."""
        peak_series, valley_series = df['high'], df['low']
        window_size = 2 * depth + 1
        rolling_max, rolling_min = peak_series.rolling(window=window_size, center=True, min_periods=1).max(), \
            valley_series.rolling(window=window_size,
                                  center=True, min_periods=1).min()
        candidate_peaks_df, candidate_valleys_df = df[peak_series ==
                                                      rolling_max], df[valley_series == rolling_min]
        candidates: List[Dict[str, Any]] = []
        for idx, row in candidate_peaks_df.iterrows():
            candidates.append(
                {'idx': idx, 'preco': row[peak_series.name], 'tipo': 'PICO'})
        for idx, row in candidate_valleys_df.iterrows():
            candidates.append(
                {'idx': idx, 'preco': row[valley_series.name], 'tipo': 'VALE'})
        # remove duplicates and sort
        candidates = sorted(
            {p['idx']: p for p in candidates}.values(), key=lambda x: x['idx'])
        if len(candidates) < 2:
            return []

        confirmed_pivots = [candidates[0]]
        last_pivot = candidates[0]
        for i in range(1, len(candidates)):
            candidate = candidates[i]
            if candidate['tipo'] == last_pivot['tipo']:
                if (candidate['tipo'] == 'PICO' and candidate['preco'] > last_pivot['preco']) or \
                   (candidate['tipo'] == 'VALE' and candidate['preco'] < last_pivot['preco']):
                    confirmed_pivots[-1] = last_pivot = candidate
                continue
            if last_pivot['preco'] == 0:
                continue
            price_dev = abs(
                candidate['preco'] - last_pivot['preco']) / last_pivot['preco'] * 100
            if price_dev >= deviation_percent:
                confirmed_pivots.append(candidate)
                last_pivot = candidate

        # optional extension to last bar with pivot merge
        if Config.ZIGZAG_EXTEND_TO_LAST_BAR and confirmed_pivots:
            last_confirmed_pivot = confirmed_pivots[-1]
            last_bar = df.iloc[-1]

            # if last bar extends same direction: update pivot; else create opposite pivot
            if last_confirmed_pivot['tipo'] == 'PICO':
                # up move: update last high pivot
                if last_bar['high'] > last_confirmed_pivot['preco']:
                    last_confirmed_pivot['preco'] = last_bar['high']
                    last_confirmed_pivot['idx'] = df.index[-1]
                else:
                    # reversed: create a low pivot
                    potential_pivot = {
                        'idx': df.index[-1],
                        'tipo': 'VALE',
                        'preco': last_bar['low']
                    }
                    if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                        confirmed_pivots.append(potential_pivot)
            else:  # last pivot is a low
                # down move: update last low pivot
                if last_bar['low'] < last_confirmed_pivot['preco']:
                    last_confirmed_pivot['preco'] = last_bar['low']
                    last_confirmed_pivot['idx'] = df.index[-1]
                else:
                    # reversed: create a high pivot
                    potential_pivot = {
                        'idx': df.index[-1],
                        'tipo': 'PICO',
                        'preco': last_bar['high']
                    }
                    if potential_pivot['idx'] != last_confirmed_pivot['idx']:
                        confirmed_pivots.append(potential_pivot)

        return confirmed_pivots

    def _preparar_zigzag_plot(self, todos_os_pivots: List[Dict[str, Any]], df_view: pd.DataFrame) -> pd.Series:
        """Intercala pivôs visíveis e interpola para uma linha contínua."""
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
            plot_points_dict[ponto_de_ancoragem['idx']
                             ] = ponto_de_ancoragem['preco']
        for p in todos_os_pivots:
            if p['idx'] in df_view.index:
                plot_points_dict[p['idx']] = p['preco']

        # ensure line reaches the last visible candle
        if plot_points_dict:
            last_visible_idx = max(plot_points_dict.keys())
            if last_visible_idx != df_view.index[-1]:
                plot_points_dict[df_view.index[-1]] = df_view['close'].iloc[-1]

        if not plot_points_dict:
            return pd.Series(dtype='float64', index=df_view.index)
        points_series = pd.Series(plot_points_dict)
        combined_index = df_view.index.union(points_series.index)
        aligned_series = points_series.reindex(combined_index)
        interpolated_series = aligned_series.interpolate(method='linear')
        final_series = interpolated_series.reindex(
            df_view.index).bfill().ffill()
        return final_series

    def on_key_press(self, event: tk.Event):
        """Mapeia atalhos: A/R para rótulo e Q para sair."""
        key = event.keysym.lower()
        if key in ['a', 'r']:
            self.marcar_e_avancar(1 if key == 'a' else 0)
        elif key == 'q':
            self.on_closing()

    def marcar_e_avancar(self, label: int):
        """Salva rótulo atual e avança para o próximo padrão."""
        if self.df_trabalho is None:
            return
        self.df_trabalho.loc[self.indice_atual, 'label_humano'] = label
        self.df_trabalho.to_csv(
            self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
        self.carregar_proximo_padrao()

    def on_closing(self):
        """Fecha a aplicação e libera recursos gráficos."""
        print("Saindo...")
        plt.close('all')
        self.destroy()

    def atualizar_info_label(self):
        """Atualiza painel textual com progresso, metadados e score."""
        if self.df_trabalho is None:
            return
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
        # choose rule set by pattern type
        tipo = str(padrao.get('tipo_padrao', '')).upper()
        regras_ativas = self.regras_map_hns if tipo in [
            'OCO', 'OCOI'] else self.regras_map_dtb

        # clear bulletin slots
        for i in range(self.max_rule_slots):
            self.boletim_name_labels[i].config(text="")
            self.boletim_value_labels[i].config(text="...", fg="black")

        # fill dynamically from selected rule set
        for i, (key, nome_regra) in enumerate(regras_ativas.items()):
            if i >= self.max_rule_slots:
                break
            self.boletim_name_labels[i].config(text=f"{nome_regra}:")
            status_bool = bool(padrao.get(key, False))
            status_text = "SIM" if status_bool else "NÃO"
            status_color = "green" if status_bool else "red"
            self.boletim_value_labels[i].config(
                text=status_text, fg=status_color)


if __name__ == '__main__':
    # main entry
    output_dir = os.path.dirname(Config.ARQUIVO_SAIDA)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(Config.ARQUIVO_ENTRADA):
        messagebox.showerror(
            "Erro de Arquivo", f"Arquivo de entrada não encontrado!\n{Config.ARQUIVO_ENTRADA}\n\nPor favor, execute o gerador v20 primeiro."
        )
    else:
        app = LabelingTool(Config.ARQUIVO_ENTRADA, Config.ARQUIVO_SAIDA)
        app.mainloop()
