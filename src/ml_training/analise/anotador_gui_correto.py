# prediction_viewer_gui.py (v2.1 - À Prova de Balas)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import traceback
from datetime import timedelta

# --- NOVA FUNÇÃO ASSISTENTE PARA DATAS ---


def safe_to_naive_datetime(date_series):
    """
    Converte uma série de strings de data, potencialmente com formatos mistos
    de fuso horário, para um formato de data "naive" (sem fuso) de forma robusta.
    """
    # Tenta a conversão rápida primeiro
    try:
        # Este é o método mais robusto que o Pandas oferece
        return pd.to_datetime(date_series, errors='coerce', utc=True).dt.tz_localize(None)
    except (ValueError, TypeError):
        # Se o método rápido falhar devido a formatos muito mistos,
        # recorremos ao método lento, processando cada data individualmente.
        print("AVISO: Formato de data inconsistente detectado. Usando método de conversão mais lento.")

        def convert_single(d):
            if pd.isna(d):
                return pd.NaT
            dt = pd.to_datetime(d, errors='coerce')
            if pd.notna(dt) and dt.tzinfo is not None:
                return dt.tz_localize(None)
            return dt

        return date_series.apply(convert_single)


class PredictionViewerTool(tk.Tk):
    """
    Ferramenta gráfica para visualizar e anotar as previsões de um modelo.
    v2.1 com parsing de data individual para máxima robustez.
    """

    def __init__(self, arquivo_predicoes):
        super().__init__()
        self.title("Visualizador de Previsões do Modelo v2.1 (Final)")
        # ... resto do __init__ ...
        self.geometry("1300x950")
        self.arquivo_predicoes_path = arquivo_predicoes
        self.df_predicoes = None
        self.indice_atual = 0
        self.data_cache = {}
        self.alteracoes_pendentes = False
        self.canvas_widget = None
        self.fig = None
        if not self._setup_dataframe():
            self.destroy()
            return
        if self.df_predicoes.empty:
            messagebox.showinfo(
                "Informação", "O arquivo de previsões está vazio.")
            self.destroy()
            return
        self._setup_ui()
        self.bind_all('<Key>', self._handle_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._carregar_item_atual()

    def _setup_dataframe(self):
        try:
            if not os.path.exists(self.arquivo_predicoes_path):
                raise FileNotFoundError(
                    f"Arquivo de previsões não encontrado: {self.arquivo_predicoes_path}")
            self.df_predicoes = pd.read_csv(self.arquivo_predicoes_path)

            if 'data_inicio' not in self.df_predicoes.columns and 'ombro1_idx' in self.df_predicoes.columns:
                self.df_predicoes.rename(
                    columns={'ombro1_idx': 'data_inicio'}, inplace=True)
            if 'data_fim' not in self.df_predicoes.columns and 'ombro2_idx' in self.df_predicoes.columns:
                self.df_predicoes.rename(
                    columns={'ombro2_idx': 'data_fim'}, inplace=True)

            colunas_necessarias = ['ticker', 'timeframe', 'data_inicio',
                                   'data_fim', 'previsao_modelo', 'confianca_modelo']
            if not all(col in self.df_predicoes.columns for col in colunas_necessarias):
                raise ValueError(
                    f"Colunas obrigatórias não encontradas: {[col for col in colunas_necessarias if col not in self.df_predicoes.columns]}")

            # --- AQUI ESTÁ A LÓGICA FINAL ---
            print("Padronizando colunas de data...")
            for col in ['data_inicio', 'data_fim']:
                # Usa a nossa nova função "à prova de balas"
                self.df_predicoes[col] = safe_to_naive_datetime(
                    self.df_predicoes[col])

            if 'notas_revisao' not in self.df_predicoes.columns:
                self.df_predicoes['notas_revisao'] = ""
            else:
                self.df_predicoes['notas_revisao'] = self.df_predicoes['notas_revisao'].fillna(
                    '')
            return True
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro Crítico na Inicialização",
                                 f"Não foi possível carregar ou processar o arquivo de previsões:\n\n{e}")
            return False

    # (O resto do código, incluindo a v1.8 de _plotar_grafico, permanece o mesmo)
    def _setup_ui(self):
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)

        self.frame_grafico = tk.Frame(m, bg='black')
        m.add(self.frame_grafico, minsize=400)

        # --- NOVA LINHA: PREPARANDO O FRAME PARA O GRID ---
        # Dizemos ao frame do gráfico que sua primeira linha (0) e primeira coluna (0)
        # devem se expandir para preencher qualquer espaço extra. O 'weight=1' faz a mágica.
        self.frame_grafico.grid_rowconfigure(0, weight=1)
        self.frame_grafico.grid_columnconfigure(0, weight=1)
        # --- FIM DA NOVA LINHA ---

        container_inferior = tk.Frame(m)
        m.add(container_inferior, minsize=150)

        frame_info = tk.Frame(container_inferior)
        frame_info.pack(side=tk.LEFT, fill=tk.BOTH,
                        expand=True, padx=10, pady=5)

        self.info_label = tk.Label(frame_info, text="Carregando...", font=(
            "Courier New", 12), justify=tk.LEFT, anchor='nw')
        self.info_label.pack(side=tk.TOP, fill=tk.X)

        tk.Label(frame_info, text="Suas Notas:", font=(
            "Arial", 10, "bold")).pack(anchor='w', pady=(10, 0))

        self.notas_entry = tk.Entry(frame_info, font=("Arial", 11))
        self.notas_entry.pack(fill=tk.X, expand=True)

        frame_acoes = tk.Frame(container_inferior, width=350)
        frame_acoes.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        frame_acoes.pack_propagate(False)

        botoes_texto = (
            "NAVEGAÇÃO E CONTROLE:\n\n[<] Anterior | [>] Próximo\n\n[S] - SALVAR Notas\n[Q] - Sair")

        self.action_label = tk.Label(frame_acoes, text=botoes_texto, font=(
            "Arial", 12, "bold"), justify=tk.LEFT, fg='#333')
        self.action_label.pack(side=tk.LEFT, padx=10)

    def _carregar_item_atual(self):
        self._limpar_grafico()
        if not (0 <= self.indice_atual < len(self.df_predicoes)):
            return
        self.item_atual_info = self.df_predicoes.iloc[self.indice_atual]
        self.notas_entry.delete(0, tk.END)
        nota_existente = self.item_atual_info.get('notas_revisao', '')
        self.notas_entry.insert(0, str(nota_existente))
        self._plotar_grafico()
        self._atualizar_info_label()

    # prediction_viewer_gui.py

    def _plotar_grafico(self):
        self._limpar_grafico()
        padrao = self.item_atual_info
        cache_key = f"{padrao['ticker']}-{padrao['timeframe']}"
        df_historico = self.data_cache.get(cache_key)

        if df_historico is None:
            try:
                data_inicio_req = padrao['data_inicio'] - \
                    pd.Timedelta(weeks=20)
                data_fim_req = padrao['data_fim'] + pd.Timedelta(weeks=20)
                df_download = yf.download(tickers=padrao['ticker'], start=data_inicio_req,
                                          end=data_fim_req, interval=padrao['timeframe'], progress=False, auto_adjust=True)
                if not df_download.empty:
                    self.data_cache[cache_key] = df_download.copy()
                    df_historico = df_download
            except Exception as e:
                print(f"ERRO de Download: {e}")

        if df_historico is None or df_historico.empty:
            messagebox.showwarning(
                "Aviso", f"Não foi possível obter dados para {padrao['ticker']}.")
            self._navegar(1)
            return

        df_para_plotar = df_historico.copy()
        if isinstance(df_para_plotar.columns, pd.MultiIndex):
            df_para_plotar.columns = df_para_plotar.columns.get_level_values(0)
        if df_para_plotar.index.tz is not None:
            df_para_plotar.index = df_para_plotar.index.tz_localize(None)

        df_para_plotar.columns = [str(col).lower()
                                  for col in df_para_plotar.columns]

        ohlc_cols = ['open', 'high', 'low', 'close']
        if not all(col in df_para_plotar.columns for col in ohlc_cols):
            messagebox.showwarning(
                "Dados Inválidos", f"Os dados para {padrao['ticker']} não contêm as colunas OHLC. Pulando.")
            self._navegar(1)
            return

        for col in ohlc_cols:
            df_para_plotar[col] = pd.to_numeric(
                df_para_plotar[col], errors='coerce')
        df_para_plotar.dropna(subset=ohlc_cols, inplace=True)

        # A renomeação crucial acontece aqui, no DataFrame principal
        rename_map = {'open': 'Open', 'high': 'High',
                      'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df_para_plotar.rename(columns={k: v for k, v in rename_map.items(
        ) if k in df_para_plotar.columns}, inplace=True)

        try:
            data_inicio_plot = padrao['data_inicio']
            data_fim_plot = padrao['data_fim']

            # --- INÍCIO DA CORREÇÃO PRECISA ---
            # O bloco a seguir substitui o cálculo de buffer antigo para desachatar o gráfico.
            # Ele opera no df_para_plotar que JÁ TEM as colunas renomeadas para 'Open', 'High', etc.
            duracao_padrao = data_fim_plot - data_inicio_plot
            buffer = max(pd.Timedelta(days=30), duracao_padrao * 4)
            df_para_plotar_final = df_para_plotar.loc[data_inicio_plot -
                                                      buffer: data_fim_plot + buffer]
            # --- FIM DA CORREÇÃO PRECISA ---

            if df_para_plotar_final.empty:
                messagebox.showwarning(
                    "Aviso", "Não há dados no intervalo de datas do padrão.")
                self._navegar(1)
                return

            previsao = int(padrao['previsao_modelo'])
            cor_highlight = 'green' if previsao == 1 else 'red'

            self.fig, axlist = mpf.plot(
                df_para_plotar_final, type='candle', style='charles', returnfig=True, warn_too_much_data=10000, figsize=(13, 7))

            axlist[0].axvspan(data_inicio_plot, data_fim_plot,
                              color=cor_highlight, alpha=0.3)

            self.canvas_widget = FigureCanvasTkAgg(
                self.fig, master=self.frame_grafico)
            self.canvas_widget.draw()
            self.canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro de Plotagem",
                                 f"Erro ao gerar gráfico:\n{e}")
            self._navegar(1)

    def _atualizar_info_label(self):
        padrao = self.item_atual_info
        previsao = int(padrao['previsao_modelo'])
        confianca = float(padrao['confianca_modelo'])
        texto_previsao = "VÁLIDO (1)" if previsao == 1 else "INVÁLIDO (0)"
        cor_previsao = "green" if previsao == 1 else "red"
        status_salvo = "" if self.alteracoes_pendentes else " (Notas Salvas)"
        self.title(f"Visualizador de Previsões v2.1{status_salvo}")
        info_text = (f"Progresso:           Visualizando Padrão {self.indice_atual + 1} de {len(self.df_predicoes)}\n"
                     f"Ativo / Intervalo:   {padrao['ticker']} ({padrao['timeframe']})\n"
                     f"Início do Padrão:    {padrao['data_inicio'].strftime('%Y-%m-%d %H:%M')}\n\n"
                     f"Confiança do Modelo: {confianca:.2%}\n"
                     f"PREVISÃO DO MODELO:  {texto_previsao}")
        self.info_label.config(text=info_text, fg=cor_previsao)

    def _update_current_note(self):
        if not hasattr(self, 'item_atual_info'):
            return
        nota_atual = self.notas_entry.get().strip()
        nota_antiga = self.df_predicoes.at[self.item_atual_info.name,
                                           'notas_revisao']
        if nota_atual != nota_antiga:
            self.df_predicoes.at[self.item_atual_info.name,
                                 'notas_revisao'] = nota_atual
            self.alteracoes_pendentes = True

    def _handle_key_press(self, event):
        key = event.keysym.lower()
        if key == 's':
            self._salvar_alteracoes()
        elif key == 'q':
            self.on_closing()
        elif key == 'left':
            self._navegar(-1)
        elif key == 'right':
            self._navegar(1)

    def _salvar_alteracoes(self):
        self._update_current_note()
        if not self.alteracoes_pendentes:
            messagebox.showinfo("Salvar", "Nenhuma nota nova para salvar.")
            return
        try:
            self.df_predicoes.to_csv(
                self.arquivo_predicoes_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
            self.alteracoes_pendentes = False
            self._atualizar_info_label()
            messagebox.showinfo(
                "Salvo", "Notas salvas com sucesso no arquivo original!")
        except Exception as e:
            messagebox.showerror(
                "Erro ao Salvar", f"Não foi possível salvar as notas:\n{e}")

    def _navegar(self, direcao: int):
        self._update_current_note()
        novo_indice = self.indice_atual + direcao
        if 0 <= novo_indice < len(self.df_predicoes):
            self.indice_atual = novo_indice
            self._carregar_item_atual()
        else:
            messagebox.showinfo("Fim da Navegação",
                                "Você chegou ao fim da lista de previsões.")

    def on_closing(self):
        self._update_current_note()
        if self.alteracoes_pendentes:
            if messagebox.askyesno("Salvar Notas", "Você tem notas não salvas. Deseja salvá-las antes de sair?"):
                self._salvar_alteracoes()
        self._limpar_grafico()
        self.destroy()

    def _limpar_grafico(self):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None
        if self.fig:
            plt.close(self.fig)
            self.fig = None


if __name__ == '__main__':
    ARQUIVO_DE_PREVISOES = 'data/reports/resultado_predicoes_com_features.csv'
    if not os.path.exists(ARQUIVO_DE_PREVISOES):
        messagebox.showerror(
            "Erro", f"Arquivo de previsões não encontrado em:\n{ARQUIVO_DE_PREVISOES}\n\nExecute o script de predição primeiro.")
    else:
        app = PredictionViewerTool(ARQUIVO_DE_PREVISOES)
        app.mainloop()
