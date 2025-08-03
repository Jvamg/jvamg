# error_analyzer_gui.py (v3.2 - Versão Final Corrigida)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import traceback

# --- INÍCIO DA CONFIGURAÇÃO ---
COLUNA_LABEL_REAL = 'label_humano'
COLUNA_PREVISAO_MODELO = 'previsao_modelo'
# --- FIM DA CONFIGURAÇÃO ---


class ErrorCorrectionTool(tk.Tk):
    """
    Ferramenta gráfica para analisar, corrigir e anotar erros de um modelo de ML.
    """

    def __init__(self, arquivo_de_erros, arquivo_mestre):
        super().__init__()
        self.title("Ferramenta de Análise de Erros v3.2 (Final)")
        self.geometry("1300x950")

        self.arquivo_erros_path = arquivo_de_erros
        self.arquivo_mestre_path = arquivo_mestre
        self.df_erros, self.df_mestre = None, None
        self.indice_erro_atual = 0

        self.data_cache = {}
        self.alteracoes_pendentes = False
        self.canvas_widget = None
        self.fig = None

        if not self._setup_dataframes():
            self.destroy()
            return

        if self.df_erros.empty:
            messagebox.showinfo("Concluído", "O arquivo de erros está vazio.")
            self.destroy()
            return

        self._setup_ui()

        # Vínculos de eventos (Bindings)
        self.bind_all('<Key>', self._handle_key_press)
        self.bind('<Left>', lambda event: self._navegar_erro_anterior())
        self.bind('<Right>', lambda event: self._navegar_proximo_erro())
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._carregar_erro_atual()

    def _setup_dataframes(self):
        try:
            if not os.path.exists(self.arquivo_erros_path):
                raise FileNotFoundError(
                    f"Arquivo de erros não encontrado: {self.arquivo_erros_path}")
            self.df_erros = pd.read_csv(self.arquivo_erros_path)

            colunas_necessarias = [COLUNA_LABEL_REAL, COLUNA_PREVISAO_MODELO,
                                   'ticker', 'data_inicio', 'data_fim', 'intervalo']
            if not all(col in self.df_erros.columns for col in colunas_necessarias):
                raise ValueError(
                    "Uma ou mais colunas obrigatórias não foram encontradas no arquivo de erros.")

            self.df_erros['data_inicio'] = pd.to_datetime(
                self.df_erros['data_inicio'], errors='coerce')
            self.df_erros['data_fim'] = pd.to_datetime(
                self.df_erros['data_fim'], errors='coerce')
            if self.df_erros['data_inicio'].isnull().any() or self.df_erros['data_fim'].isnull().any():
                raise ValueError(
                    "Datas inválidas (null) encontradas no arquivo de erros após a conversão.")

            if not os.path.exists(self.arquivo_mestre_path):
                raise FileNotFoundError(
                    f"Arquivo mestre não encontrado: {self.arquivo_mestre_path}")
            self.df_mestre = pd.read_csv(self.arquivo_mestre_path)
            self.df_mestre['data_inicio'] = pd.to_datetime(
                self.df_mestre['data_inicio'], errors='coerce')

            if 'notas_revisao' not in self.df_mestre.columns:
                self.df_mestre['notas_revisao'] = ""
            return True
        except Exception as e:
            messagebox.showerror("Erro Crítico na Inicialização",
                                 f"Não foi possível carregar os arquivos:\n\n{e}")
            return False

    def _setup_ui(self):
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)
        self.frame_grafico = tk.Frame(m, bg='black')
        m.add(self.frame_grafico, minsize=400)
        container_inferior = tk.Frame(m)
        m.add(container_inferior, minsize=150)
        frame_info = tk.Frame(container_inferior)
        frame_info.pack(side=tk.LEFT, fill=tk.BOTH,
                        expand=True, padx=10, pady=5)
        self.info_label = tk.Label(frame_info, text="Carregando...", font=(
            "Courier New", 12), justify=tk.LEFT, anchor='nw')
        self.info_label.pack(side=tk.TOP, fill=tk.X)
        tk.Label(frame_info, text="Notas de Revisão:", font=(
            "Arial", 10, "bold")).pack(anchor='w', pady=(10, 0))
        self.notas_entry = tk.Entry(frame_info, font=("Arial", 11))
        self.notas_entry.pack(fill=tk.X, expand=True)
        frame_acoes = tk.Frame(container_inferior, width=350)
        frame_acoes.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=5)
        frame_acoes.pack_propagate(False)
        botoes_texto = (
            "AÇÕES (TECLADO):\n\n[C] - Manter Rótulo Correto\n[I] - INVERTER Rótulo\n[M] - Marcar como AMBÍGUO (-2)\n\nNAVEGAÇÃO E CONTROLE:\n\n[<] Anterior | [>] Próximo\n[S] - SALVAR Alterações\n[Q] - Sair")
        self.action_label = tk.Label(frame_acoes, text=botoes_texto, font=(
            "Arial", 12, "bold"), justify=tk.LEFT, fg='#333')
        self.action_label.pack(side=tk.LEFT, padx=10)

    def _carregar_erro_atual(self):
        self._limpar_grafico()
        if not (0 <= self.indice_erro_atual < len(self.df_erros)):
            return
        self.erro_atual_info = self.df_erros.iloc[self.indice_erro_atual]
        self.notas_entry.delete(0, tk.END)
        self._plotar_grafico()
        self._atualizar_info_label()

    def _limpar_grafico(self):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None
        if self.fig:
            plt.close(self.fig)
            self.fig = None

    def _plotar_grafico(self):
        self._limpar_grafico()

        padrao = self.erro_atual_info
        cache_key = f"{padrao['ticker']}-{padrao['intervalo']}"
        df_grafico_bruto = None

        if cache_key in self.data_cache:
            print(f"INFO: Dados para {cache_key} carregados do cache.")
            df_grafico_bruto = self.data_cache[cache_key]
        else:
            try:
                print(f"INFO: Novos dados para {cache_key} sendo baixados...")
                data_inicio_req = padrao['data_inicio'] - \
                    pd.Timedelta(weeks=12)
                data_fim_req = padrao['data_fim'] + pd.Timedelta(weeks=12)

                df_download = yf.download(tickers=padrao['ticker'], start=data_inicio_req,
                                          end=data_fim_req, interval=padrao['intervalo'],
                                          progress=False, auto_adjust=False, group_by='ticker')

                if not df_download.empty:
                    if isinstance(df_download.columns, pd.MultiIndex):
                        df_download.columns = df_download.columns.get_level_values(
                            -1)

                    df_download.index = df_download.index.tz_localize(None)
                    self.data_cache[cache_key] = df_download.copy()
                    df_grafico_bruto = df_download
            except Exception as e:
                print(f"ERRO de Download para {cache_key}: {e}")

        if df_grafico_bruto is None or df_grafico_bruto.empty:
            messagebox.showwarning(
                "Aviso de Dados", f"Não foi possível obter dados para {padrao['ticker']}. Pulando.")
            self._navegar_proximo_erro()
            return

        df_grafico = df_grafico_bruto.copy()
        df_grafico.columns = [str(col).lower() for col in df_grafico.columns]

        expected_cols_lower = {'open', 'high', 'low', 'close'}
        if not expected_cols_lower.issubset(df_grafico.columns):
            messagebox.showwarning(
                "Dados Incompletos", f"Os dados para {padrao['ticker']} não contêm as colunas OHLC. Pulando.")
            self._navegar_proximo_erro()
            return

        for col in list(expected_cols_lower):
            df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce')
        df_grafico.dropna(subset=list(expected_cols_lower), inplace=True)

        if df_grafico.empty:
            self._navegar_proximo_erro()
            return

        try:
            data_inicio_plot, data_fim_plot = padrao['data_inicio'], padrao['data_fim']
            buffer = max(pd.Timedelta(weeks=2),
                         (data_fim_plot - data_inicio_plot) * 1.5)

            df_para_plotar = df_grafico.loc[data_inicio_plot -
                                            buffer: data_fim_plot + buffer].copy()

            # ================================================================================= #
            # === CORREÇÃO: Verifica se a fatia de dados para o gráfico não está vazia      === #
            # ================================================================================= #
            if df_para_plotar.empty:
                messagebox.showwarning(
                    "Dados Não Encontrados", f"Não foi possível encontrar dados no intervalo de datas específico para {padrao['ticker']}.\nO padrão pode ser muito antigo.\nPulando para o próximo.")
                self._navegar_proximo_erro()
                return

            rename_map = {'open': 'Open', 'high': 'High',
                          'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df_para_plotar.rename(columns=rename_map, inplace=True)

            self.fig, axlist = mpf.plot(df_para_plotar, type='candle', style='charles', returnfig=True, figsize=(
                12, 8), warn_too_much_data=10000)
            self.ax = axlist[0]

            start_pos_idx = df_para_plotar.index.get_indexer(
                [data_inicio_plot], method='nearest')[0]
            end_pos_idx = df_para_plotar.index.get_indexer(
                [data_fim_plot], method='nearest')[0]
            self.ax.axvspan(start_pos_idx, end_pos_idx,
                            color='yellow', alpha=0.3)

            self.canvas_widget = FigureCanvasTkAgg(
                self.fig, master=self.frame_grafico)
            self.canvas_widget.draw()
            self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro de Plotagem",
                                 f"Ocorreu um erro ao gerar o gráfico:\n{e}")
            self._navegar_proximo_erro()

    def _atualizar_info_label(self):
        padrao = self.erro_atual_info
        label_real = int(padrao[COLUNA_LABEL_REAL])
        previsao = int(padrao[COLUNA_PREVISAO_MODELO])
        resultado = "FALSO POSITIVO" if label_real == 0 else "FALSO NEGATIVO"
        status_salvo = "" if self.alteracoes_pendentes else " (Alterações Salvas)"
        self.title(f"Ferramenta de Análise de Erros v3.2{status_salvo}")
        info_text = (f"Progresso:           Analisando Erro {self.indice_erro_atual + 1} de {len(self.df_erros)}\n"
                     f"Ativo / Intervalo:   {padrao['ticker']} ({padrao['intervalo']})\n"
                     f"Data Início:         {padrao['data_inicio'].strftime('%Y-%m-%d %H:%M')}\n\n"
                     f"RÓTULO REAL:         {label_real}\n"
                     f"PREVISÃO DO MODELO:  {previsao}\n\n"
                     f"RESULTADO:           {resultado}")
        self.info_label.config(text=info_text)

    def _handle_key_press(self, event):
        key = event.keysym.lower()
        actions = {'c': 'manter', 'i': 'inverter', 'm': 'marcar_ambiguo'}
        if key in actions:
            self._processar_acao(tipo=actions[key])
        elif key == 's':
            self._salvar_mestre()
        elif key == 'q':
            self.on_closing()

    def _processar_acao(self, tipo: str):
        erro_info = self.erro_atual_info
        ticker, data_inicio = erro_info['ticker'], erro_info['data_inicio']
        mask = (self.df_mestre['ticker'] == ticker) & (
            self.df_mestre['data_inicio'] == data_inicio)
        indices_mestre = self.df_mestre.index[mask]
        if indices_mestre.empty:
            messagebox.showerror("Erro de Correspondência",
                                 "Padrão não encontrado no arquivo mestre.")
            return

        idx_mestre = indices_mestre[0]
        notas = self.notas_entry.get().strip()

        acao_realizada = False
        if notas:
            self.df_mestre.loc[idx_mestre, 'notas_revisao'] = notas
            acao_realizada = True

        if tipo == 'inverter':
            label_antigo = int(self.df_mestre.loc[idx_mestre, 'label_humano'])
            novo_label = 1 - label_antigo
            self.df_mestre.loc[idx_mestre, 'label_humano'] = novo_label
            acao_realizada = True
            print(
                f"Rótulo invertido para o índice mestre {idx_mestre}: de {label_antigo} para {novo_label}.")
        elif tipo == 'marcar_ambiguo':
            self.df_mestre.loc[idx_mestre, 'label_humano'] = -2
            acao_realizada = True
            print(
                f"Padrão no índice mestre {idx_mestre} marcado como AMBÍGUO (-2).")

        if acao_realizada:
            self.alteracoes_pendentes = True
            self._atualizar_info_label()

        self._navegar_proximo_erro()

    def _salvar_mestre(self):
        if not self.alteracoes_pendentes:
            messagebox.showinfo(
                "Salvar", "Nenhuma alteração pendente para salvar.")
            return
        try:
            self.df_mestre.to_csv(self.arquivo_mestre_path,
                                  index=False, date_format='%Y-%m-%d %H:%M:%S')
            self.alteracoes_pendentes = False
            self._atualizar_info_label()
            messagebox.showinfo("Salvo", "Alterações salvas com sucesso!")
        except Exception as e:
            messagebox.showerror(
                "Erro ao Salvar", f"Não foi possível salvar as alterações:\n{e}")

    def _navegar_proximo_erro(self):
        if self.indice_erro_atual < len(self.df_erros) - 1:
            self.indice_erro_atual += 1
            self._carregar_erro_atual()
        else:
            messagebox.showinfo("Fim da Lista", "Você analisou o último erro.")

    def _navegar_erro_anterior(self):
        if self.indice_erro_atual > 0:
            self.indice_erro_atual -= 1
            self._carregar_erro_atual()
        else:
            messagebox.showinfo("Início da Lista",
                                "Você já está no primeiro erro.")

    def on_closing(self):
        if self.alteracoes_pendentes:
            if messagebox.askyesno("Salvar Alterações", "Você tem alterações não salvas. Deseja salvá-las antes de sair?"):
                self._salvar_mestre()
        print("Saindo e liberando recursos...")
        self._limpar_grafico()
        self.destroy()


if __name__ == '__main__':
    # --- CAMINHOS DOS ARQUIVOS ---
    ARQUIVO_DE_ERROS = 'data/datasets/erros/analise_de_erros.csv'
    ARQUIVO_MESTRE = 'data/datasets/enriched/dataset_final_ml.csv'

    # Criador de arquivos de exemplo
    if not os.path.exists(ARQUIVO_DE_ERROS):
        print(
            f"AVISO: '{ARQUIVO_DE_ERROS}' não encontrado. Criando um de exemplo.")
        data = {'ticker': ['PETR4.SA'], 'data_inicio': ['2023-01-01 10:00:00'],
                'data_fim': ['2023-01-10 15:00:00'], 'intervalo': ['1h'],
                COLUNA_LABEL_REAL: [0], COLUNA_PREVISAO_MODELO: [1]}
        pd.DataFrame(data).to_csv(ARQUIVO_DE_ERROS, index=False)

    if not os.path.exists(ARQUIVO_MESTRE):
        print(
            f"AVISO: '{ARQUIVO_MESTRE}' não encontrado. Criando um de exemplo.")
        mestre_dir = os.path.dirname(ARQUIVO_MESTRE)
        if mestre_dir and not os.path.exists(mestre_dir):
            os.makedirs(mestre_dir)
        data = {'ticker': ['PETR4.SA'], 'data_inicio': ['2023-01-01 10:00:00'],
                'data_fim': ['2023-01-10 15:00:00'], 'intervalo': ['1h'], 'label_humano': [0],
                'notas_revisao': ['']}
        pd.DataFrame(data).to_csv(ARQUIVO_MESTRE, index=False)

    app = ErrorCorrectionTool(ARQUIVO_DE_ERROS, ARQUIVO_MESTRE)
    app.mainloop()
