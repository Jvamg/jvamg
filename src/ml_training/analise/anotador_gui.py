# anotador_gui.py (v12.3 - Correção do Erro de Atributo)
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

    # Parâmetros para o ZIGZAG VISUAL (devem espelhar a estratégia principal do gerador)
    TIMEFRAME_PARAMS = {
        'default': {'depth': 2, 'deviation': 2.0},
        '1h':      {'depth': 2, 'deviation': 3.0},
        '4h':      {'depth': 3, 'deviation': 5.0},
        '1d':      {'depth': 25, 'deviation': 12.0}
    }

    # Arquivos de entrada e saída
    ARQUIVO_ENTRADA = 'data/datasets_hns_scored_mandatory/dataset_hns_scored_mandatory_final.csv'
    ARQUIVO_SAIDA = 'data/datasets_hns_scored_mandatory/dataset_hns_labeled_confluencia.csv'

    MAX_DOWNLOAD_TENTATIVAS = 3
    RETRY_DELAY_SEGUNDOS = 5
    ZIGZAG_LOOKBACK_DAYS = 200  # Lookback para encontrar a âncora do ZigZag

# --- FUNÇÕES DE VALIDAÇÃO (para o assistente visual) ---
# NOTA: Estas funções não estão sendo usadas ativamente na GUI v12,
# mas são mantidas para referência ou uso futuro. A validação é lida do CSV.


def check_rsi_divergence(df: pd.DataFrame, p1_idx, p3_idx, p1_price, p3_price, tipo_padrao: str) -> bool:
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
    """Encontra os pivôs p0, p2, p4 na lista e verifica o perfil de volume."""
    try:
        # Encontra os índices dos pivôs na lista
        indices = {p['idx']: i for i, p in enumerate(pivots)}
        idx_p1, idx_p3, idx_p5 = indices.get(
            p1_idx), indices.get(p3_idx), indices.get(p5_idx)

        # Garante que os pivôs do padrão foram encontrados na lista recalculada
        if any(i is None for i in [idx_p1, idx_p3, idx_p5]) or idx_p1 < 2:
            return False

        # Pega os pivôs adjacentes para definir os períodos de volume
        p0_idx, p2_idx, p4_idx = pivots[idx_p1 -
                                        1]['idx'], pivots[idx_p3-1]['idx'], pivots[idx_p5-1]['idx']

        vol_cabeca = df.loc[p2_idx:p3_idx]['volume'].mean()
        vol_od = df.loc[p4_idx:p5_idx]['volume'].mean()

        return vol_cabeca > vol_od
    except Exception:
        return False
    return False


class LabelingTool(tk.Tk):
    def __init__(self, arquivo_entrada: str, arquivo_saida: str):
        super().__init__()
        self.title(f"Ferramenta de Anotação (v12.3 - Corrigido)")
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

    def _setup_ui(self):
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)
        self.frame_grafico = tk.Frame(m)
        m.add(self.frame_grafico, minsize=400)

        frame_controles = tk.Frame(m, height=150)
        frame_controles.pack_propagate(False)
        m.add(frame_controles, minsize=150)

        # --- Usando o gerenciador GRID para layout ---
        frame_controles.grid_rowconfigure(0, weight=1)
        frame_controles.grid_columnconfigure(0, weight=1)
        frame_controles.grid_columnconfigure(
            1, weight=2)  # Boletim terá o dobro do espaço

        # -- Frame da Esquerda --
        frame_info = tk.Frame(frame_controles)
        frame_info.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        self.info_label = tk.Label(frame_info, text="Carregando...", font=(
            "Segoe UI", 10), justify=tk.LEFT, anchor="nw")
        self.info_label.pack(side=tk.TOP, anchor="w", pady=(0, 10))

        self.action_label = tk.Label(
            frame_info, text="Teclado: [A]provar | [R]ejeitar | [Q]uit", font=("Segoe UI", 12, "bold"))
        self.action_label.pack(side=tk.BOTTOM, anchor="w")

        # -- Frame da Direita (Boletim) --
        frame_boletim = tk.Frame(
            frame_controles, relief=tk.RIDGE, borderwidth=1)
        frame_boletim.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

        # Título
        tk.Label(frame_boletim, text="Boletim de Validação do Padrão",
                 font=("Segoe UI", 11, "bold")).pack(pady=(5, 10))

        # Frame para a grade de validações
        grid_frame = tk.Frame(frame_boletim)
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=15)
        grid_frame.grid_columnconfigure(0, weight=1)  # Coluna nomes
        grid_frame.grid_columnconfigure(1, weight=1)  # Coluna status

        self.boletim_labels = {}

        # Cria as labels para cada regra em uma grade
        for i, (key, name) in enumerate(self.regras_map.items()):
            # Separa em duas colunas para não ficar muito longo
            col_base = 0 if i < 5 else 2

            tk.Label(grid_frame, text=f"{name}:", font=("Segoe UI", 9), anchor="w").grid(
                row=i % 5, column=col_base, sticky="w", padx=(0, 10))

            result_label = tk.Label(grid_frame, text="...", font=(
                "Segoe UI", 9, "bold"), anchor="w")
            result_label.grid(row=i % 5, column=col_base + 1, sticky="w")
            self.boletim_labels[key] = result_label

        # Label separada para o Score Final
        self.score_label = tk.Label(
            frame_boletim, text="SCORE FINAL: N/A", font=("Segoe UI", 11, "bold"), fg="#1E90FF")
        self.score_label.pack(side=tk.BOTTOM, pady=5)

    def setup_dataframe(self, arquivo_entrada: str, arquivo_saida: str) -> bool:
        try:
            if os.path.exists(arquivo_saida):
                self.df_trabalho = pd.read_csv(arquivo_saida)
            else:
                self.df_trabalho = pd.read_csv(arquivo_entrada)
            rename_map = {
                'timeframe': 'intervalo', 'padrao_tipo': 'tipo_padrao',
                'ombro1_idx': 'data_inicio', 'ombro2_idx': 'data_fim',
                'cabeca_idx': 'data_cabeca'
            }
            self.df_trabalho.rename(columns=rename_map, inplace=True)
            for col in ['data_inicio', 'data_fim', 'data_cabeca']:
                if col in self.df_trabalho.columns:
                    self.df_trabalho[col] = pd.to_datetime(
                        self.df_trabalho[col], errors='coerce')
            if 'label_humano' not in self.df_trabalho.columns:
                self.df_trabalho['label_humano'] = np.nan
            return True
        except Exception as e:
            messagebox.showerror(
                "Erro Crítico no Setup", f"Não foi possível carregar ou adaptar o dataset.\nErro: {e}")
            return False

    def carregar_proximo_padrao(self):
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

    def plotar_grafico_com_zigzag(self):
        if self.fig is not None:
            plt.close(self.fig)
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()
        if self.df_trabalho is None:
            return

        padrao_info = self.df_trabalho.loc[self.indice_atual]
        ticker, intervalo = padrao_info['ticker'], padrao_info['intervalo']

        # --- ALTERAÇÃO PRINCIPAL: Cálculo de datas ANTES do download ---
        data_inicio_padrao = pd.to_datetime(
            padrao_info['data_inicio']).tz_localize(None)
        data_fim_padrao = pd.to_datetime(
            padrao_info['data_fim']).tz_localize(None)

        if pd.isna(data_inicio_padrao) or pd.isna(data_fim_padrao):
            self.marcar_e_avancar(-1)
            return

        # Define um buffer para visualização ao redor do padrão
        duracao = data_fim_padrao - data_inicio_padrao
        if 'm' in intervalo:
            base_buffer = pd.Timedelta(hours=12)
        elif '1h' in intervalo:
            base_buffer = pd.Timedelta(days=3)
        elif '4h' in intervalo:
            base_buffer = pd.Timedelta(days=7)
        else:
            base_buffer = pd.Timedelta(days=15)

        buffer_dinamico = base_buffer + (duracao * 0.75)

        # Define as datas exatas para a janela de visualização
        start_date_view = data_inicio_padrao - buffer_dinamico
        end_date_view = data_fim_padrao + buffer_dinamico

        # Define a data de início do download para incluir o lookback do ZigZag
        # Esta é a correção chave para garantir dados suficientes para o cálculo.
        download_start_date = start_date_view - \
            pd.Timedelta(days=Config.ZIGZAG_LOOKBACK_DAYS)
        # Adiciona um dia ao final para garantir que a data final seja inclusiva
        download_end_date = end_date_view + pd.Timedelta(days=1)

        df_full = None
        for tentativa in range(Config.MAX_DOWNLOAD_TENTATIVAS):
            try:
                # Usa start/end para um download preciso em vez de 'period'
                df_full = yf.download(tickers=ticker,
                                      start=download_start_date,
                                      end=download_end_date,
                                      interval=intervalo,
                                      auto_adjust=True,
                                      progress=False)

                if not df_full.empty:
                    if isinstance(df_full.columns, pd.MultiIndex):
                        df_full.columns = df_full.columns.get_level_values(0)
                    df_full.columns = [col.lower() for col in df_full.columns]
                    break
                else:
                    raise ValueError("Download retornou um DataFrame vazio.")
            except Exception as e:
                if tentativa < Config.MAX_DOWNLOAD_TENTATIVAS - 1:
                    time.sleep(Config.RETRY_DELAY_SEGUNDOS)
                else:
                    # Marca como erro se o download falhar
                    self.marcar_e_avancar(-1)
                    return

        df_full.index = df_full.index.tz_localize(None)

        # Cria a janela de visualização a partir do dataframe completo já baixado
        df_view = df_full.loc[start_date_view:end_date_view].copy()
        if df_view.empty:
            # Marca como erro se a janela de visualização estiver vazia
            self.marcar_e_avancar(-1)
            return

        # --- CÁLCULO DO ZIGZAG VISUAL ---
        # Agora passamos o df_full inteiro, que contém os dados de lookback necessários.
        # Não precisamos mais de um 'df_calculo' separado.
        params = Config.TIMEFRAME_PARAMS.get(
            intervalo, Config.TIMEFRAME_PARAMS['default'])
        pivots_visuais = self._calcular_zigzag(
            df_full, params['depth'], params['deviation'])
        zigzag_line = self._preparar_zigzag_plot(pivots_visuais, df_view)

        # --- CÁLCULO E PREPARAÇÃO DOS INDICADORES ---
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

        # --- PLOTAGEM MULTI-PAINEL ---
        self.fig, axlist = mpf.plot(df_view, type='candle', style='charles', returnfig=True,
                                    figsize=(13, 10), addplot=ad_plots,
                                    panel_ratios=(10, 2, 2, 2),
                                    title=f"{ticker} ({intervalo}) - Padrão {self.indice_atual}",
                                    volume=True, volume_panel=3,
                                    warn_too_much_data=10000)

        ax_price = axlist[0]
        start_pos, end_pos = df_view.index.get_indexer([data_inicio_padrao], method='nearest')[
            0], df_view.index.get_indexer([data_fim_padrao], method='nearest')[0]
        ax_price.axvspan(start_pos, end_pos, color='yellow', alpha=0.2)

        if pd.notna(padrao_info['data_cabeca']):
            data_cabeca_naive = pd.to_datetime(
                padrao_info['data_cabeca']).tz_localize(None)
            head_pos = df_view.index.get_indexer(
                [data_cabeca_naive], method='nearest')[0]
            ax_price.axvline(x=head_pos, color='dodgerblue',
                             linestyle='--', linewidth=1.2)

        canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _calcular_zigzag(self, df: pd.DataFrame, depth: int, deviation_percent: float) -> List[Dict[str, Any]]:
        # ... (cópia da função 'calcular_zigzag_oficial' do gerador)
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
        return confirmed_pivots

    def _preparar_zigzag_plot(self, todos_os_pivots: List[Dict[str, Any]], df_view: pd.DataFrame) -> pd.Series:
        """
        Prepara os dados do ZigZag para plotagem de forma robusta, garantindo que a
        série de saída tenha o mesmo tamanho da janela de visualização.
        (v12.3 - CORREÇÃO FINAL)
        """
        if df_view.empty or not todos_os_pivots:
            return pd.Series(dtype='float64', index=df_view.index)

        # 1. Coletar todos os pontos-chave (âncora, pivôs, último) em um dicionário.
        plot_points_dict = {}

        # Encontra e adiciona o ponto de âncora (último pivô antes da janela)
        ponto_de_ancoragem = None
        for pivot in todos_os_pivots:
            if pivot['idx'] < df_view.index[0]:
                ponto_de_ancoragem = pivot
            else:
                break  # Pivôs são ordenados, podemos parar

        if ponto_de_ancoragem:
            plot_points_dict[ponto_de_ancoragem['idx']
                             ] = ponto_de_ancoragem['preco']

        # Adiciona os pivôs que estão dentro da área visível
        for p in todos_os_pivots:
            if p['idx'] in df_view.index:
                plot_points_dict[p['idx']] = p['preco']

        # Adiciona a barra final para estender a linha até o fim
        if Config.ZIGZAG_EXTEND_TO_LAST_BAR:
            plot_points_dict[df_view.index[-1]] = df_view['close'].iloc[-1]

        if not plot_points_dict:
            return pd.Series(dtype='float64', index=df_view.index)

        # 2. Criar uma série apenas com os pontos-chave.
        points_series = pd.Series(plot_points_dict)

        # 3. Alinhar a série de pontos com o índice do gráfico principal.
        #    .reindex() expande a série para o índice completo, preenchendo com NaN.
        #    O union() garante que o ponto de âncora (que está fora) seja incluído.
        combined_index = df_view.index.union(points_series.index)
        aligned_series = points_series.reindex(combined_index)

        # 4. Interpolar para preencher os espaços em branco (NaNs) entre os pontos.
        interpolated_series = aligned_series.interpolate(method='linear')

        # 5. Fatiar a série para garantir que ela tenha o tamanho exato da visualização.
        #    Isso resolve o `ValueError` de "x and y must have same first dimension".
        #    .bfill().ffill() garante que a linha se estenda até as bordas do gráfico.
        final_series = interpolated_series.reindex(
            df_view.index).bfill().ffill()

        return final_series

    def on_key_press(self, event: tk.Event):
        key = event.keysym.lower()
        if key in ['a', 'r']:
            self.marcar_e_avancar(1 if key == 'a' else 0)
        elif key == 'q':
            self.on_closing()

    def marcar_e_avancar(self, label: int):
        if self.df_trabalho is None:
            return
        self.df_trabalho.loc[self.indice_atual, 'label_humano'] = label
        self.df_trabalho.to_csv(
            self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
        self.carregar_proximo_padrao()

    def on_closing(self):
        print("Saindo...")
        plt.close('all')
        self.destroy()

    def atualizar_info_label(self):
        """ Atualiza os painéis de informação da UI com os dados do padrão atual, incluindo o score e o boletim de validação detalhado. """
        if self.df_trabalho is None:
            return

        total = len(self.df_trabalho)
        feitos = self.df_trabalho['label_humano'].notna().sum()
        padrao = self.df_trabalho.loc[self.indice_atual]

        # --- Atualiza o Painel de Informações Básicas (Esquerda) ---
        info_text = (f"Progresso: {feitos}/{total} | Padrão Índice: {self.indice_atual}\n"
                     f"Ativo: {padrao['ticker']} ({padrao['intervalo']})\n"
                     f"Tipo: {padrao['tipo_padrao']}\n"
                     f"Estratégia: {padrao.get('estrategia_zigzag', 'N/A')}")
        self.info_label.config(text=info_text)

        # --- Atualiza o Boletim de Validação (Direita) ---
        score = padrao.get('score_total', 0)
        self.score_label.config(text=f"SCORE FINAL: {score:.0f} / 100")

        # Itera sobre o mapa de regras para preencher o boletim
        for key, name in self.regras_map.items():
            if key in self.boletim_labels:
                status_bool = padrao.get(key, False)
                status_text = "SIM" if status_bool else "NÃO"
                status_color = "green" if status_bool else "red"

                # Atualiza a label correspondente com o texto e a cor
                self.boletim_labels[key].config(
                    text=status_text, fg=status_color)


if __name__ == '__main__':
    os.makedirs(os.path.dirname(Config.ARQUIVO_ENTRADA), exist_ok=True)
    os.makedirs(os.path.dirname(Config.ARQUIVO_SAIDA), exist_ok=True)
    if not os.path.exists(Config.ARQUIVO_ENTRADA):
        messagebox.showerror(
            "Erro de Arquivo", f"Arquivo de entrada não encontrado!\n{Config.ARQUIVO_ENTRADA}\n\nPor favor, execute o gerador v17 primeiro.")
    else:
        app = LabelingTool(Config.ARQUIVO_ENTRADA, Config.ARQUIVO_SAIDA)
        app.mainloop()
