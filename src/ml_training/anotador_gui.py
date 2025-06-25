# anotador_gui.py (versão final consolidada)
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import re
import os


class LabelingToolHybrid(tk.Tk):
    """
    Ferramenta de labeling híbrida que combina ajuste fino com botões e
    rotulagem de alta velocidade com o teclado. Inclui gerenciamento de
    estado, de memória e lida com diversos erros de dados e plotagem.
    """

    def __init__(self, arquivo_entrada, arquivo_saida):
        super().__init__()

        self.title("Ferramenta de Labeling Otimizada (v6.3 - Final)")
        self.geometry("1300x900")

        self.arquivo_saida = arquivo_saida
        self.df_trabalho = None

        # Variáveis de estado
        self.indice_atual = 0
        self.padrao_atual_info = None
        self.canvas_widget = None
        self.fig = None
        self.data_inicio_ajustada = None
        self.data_fim_ajustada = None
        self.data_cabeca_ajustada = None
        self.df_grafico_atual = None
        self.ax = None
        self.span_amarelo = None
        self.linha_cabeca = None

        if not self.setup_dataframe(arquivo_entrada, arquivo_saida):
            return

        indices_para_anotar = self.df_trabalho[self.df_trabalho['label_humano'].isnull(
        )].index
        if indices_para_anotar.empty:
            messagebox.showinfo(
                "Concluído", "Parabéns! Todos os padrões já foram rotulados.")
            self.destroy()
            return
        self.indice_atual = indices_para_anotar[0]

        self._setup_ui()
        self.bind('<Key>', self.on_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.carregar_proximo_padrao()

    def _setup_ui(self):
        # O setup da UI permanece o mesmo
        m = tk.PanedWindow(self, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        m.pack(fill=tk.BOTH, expand=True)
        self.frame_grafico = tk.Frame(m)
        m.add(self.frame_grafico, minsize=400)
        container_inferior = tk.Frame(m, height=150)
        container_inferior.pack_propagate(False)
        m.add(container_inferior, minsize=120)
        self.frame_inferior = tk.Frame(container_inferior)
        self.frame_inferior.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.info_label = tk.Label(
            self.frame_inferior, text="Carregando...", font=("Arial", 12), justify=tk.LEFT)
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)
        self.action_label = tk.Label(
            self.frame_inferior, text="Use o Teclado: [A]provar | [R]ejeitar | [Q]uit", font=("Arial", 12, "bold"))
        self.action_label.pack(side=tk.RIGHT, padx=10)
        self.frame_ajustes = tk.Frame(self.frame_inferior)
        self.frame_ajustes.pack(side=tk.RIGHT, padx=20)
        tk.Label(self.frame_ajustes, text="Início:",
                 font=("Arial", 10)).grid(row=0, column=0)
        tk.Button(self.frame_ajustes, text="<", command=lambda: self.ajustar_data(
            'inicio', 'tras')).grid(row=1, column=0)
        tk.Button(self.frame_ajustes, text=">", command=lambda: self.ajustar_data(
            'inicio', 'frente')).grid(row=1, column=1)
        tk.Label(self.frame_ajustes, text="Fim:", font=(
            "Arial", 10)).grid(row=0, column=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text="<", command=lambda: self.ajustar_data(
            'fim', 'tras')).grid(row=1, column=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text=">", command=lambda: self.ajustar_data(
            'fim', 'frente')).grid(row=1, column=3)
        tk.Label(self.frame_ajustes, text="Cabeça:", font=(
            "Arial", 10)).grid(row=0, column=4, padx=(10, 0))
        tk.Button(self.frame_ajustes, text="<", command=lambda: self.ajustar_data(
            'cabeca', 'tras')).grid(row=1, column=4, padx=(10, 0))
        tk.Button(self.frame_ajustes, text=">", command=lambda: self.ajustar_data(
            'cabeca', 'frente')).grid(row=1, column=5)

    def setup_dataframe(self, arquivo_entrada, arquivo_saida):
        try:
            if os.path.exists(arquivo_saida):
                self.df_trabalho = pd.read_csv(arquivo_saida)
            else:
                self.df_trabalho = pd.read_csv(arquivo_entrada)
            if 'label_humano' not in self.df_trabalho.columns:
                self.df_trabalho['label_humano'] = np.nan
            return True
        except FileNotFoundError:
            self.withdraw()
            messagebox.showerror(
                "Erro de Arquivo", f"Arquivo de entrada não encontrado!\nVerifique o caminho: '{arquivo_entrada}'")
            self.destroy()
            return False
        except Exception as e:
            self.withdraw()
            messagebox.showerror("Erro ao Ler CSV", str(e))
            self.destroy()
            return False

    def ajustar_data(self, qual_data, direcao):
        intervalo = self.padrao_atual_info['intervalo']
        match = re.match(r"(\d+)([hHdD])", intervalo)
        ajuste = pd.Timedelta(hours=4)
        if match:
            valor, unidade = int(match.groups()[0]), match.groups()[1].lower()
            ajuste = pd.Timedelta(
                hours=valor) if unidade == 'h' else pd.Timedelta(days=valor)
        fator = -1 if direcao == 'tras' else 1
        if qual_data == 'inicio':
            self.data_inicio_ajustada += (ajuste * fator)
        elif qual_data == 'fim':
            self.data_fim_ajustada += (ajuste * fator)
        elif qual_data == 'cabeca' and pd.notna(self.data_cabeca_ajustada):
            self.data_cabeca_ajustada += (ajuste * fator)
        self.atualizar_marcacoes()
        self.atualizar_info_label()

    def atualizar_marcacoes(self):
        if self.ax is None or self.df_grafico_atual is None:
            return
        if self.span_amarelo:
            self.span_amarelo.remove()
        if self.linha_cabeca:
            self.linha_cabeca.remove()
        try:
            start_pos_idx = (self.df_grafico_atual.index -
                             self.data_inicio_ajustada).to_series().abs().argmin()
            end_pos_idx = (self.df_grafico_atual.index -
                           self.data_fim_ajustada).to_series().abs().argmin()
            self.span_amarelo = self.ax.axvspan(
                start_pos_idx, end_pos_idx, color='yellow', alpha=0.3)
            if pd.notna(self.data_cabeca_ajustada):
                head_pos_idx = (self.df_grafico_atual.index -
                                self.data_cabeca_ajustada).to_series().abs().argmin()
                self.linha_cabeca = self.ax.axvline(
                    x=head_pos_idx, color='dodgerblue', linestyle='--', linewidth=1.2)
            self.canvas_widget.draw()
        except Exception as e:
            print(
                f"AVISO: Não foi possível atualizar as marcações no gráfico. Erro: {e}")

    def plotar_grafico(self):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)

        padrao = self.padrao_atual_info
        data_inicio, data_fim, intervalo = self.data_inicio_ajustada, self.data_fim_ajustada, padrao[
            'intervalo']
        duracao = data_fim - data_inicio
        buffer = max(pd.Timedelta(days=5), duracao * 1.5) if intervalo in [
            '1h', '4h'] else max(pd.Timedelta(weeks=4), duracao * 1.0)
        data_inicio_contexto, data_fim_contexto = data_inicio - buffer, data_fim + buffer

        print(
            f"Buscando dados para {padrao['ticker']} de {data_inicio_contexto.date()} a {data_fim_contexto.date()}...")
        df_grafico = yf.download(tickers=padrao['ticker'], start=data_inicio_contexto,
                                 end=data_fim_contexto, interval=intervalo, progress=False, auto_adjust=True)

        if df_grafico.empty:
            print(f"AVISO: Download falhou. Marcando como erro (-1).")
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = -1
            self.avancar_e_salvar()
            return

        # --- ORDEM DE LIMPEZA CORRIGIDA ---
        # 1. Primeiro, trata o MultiIndex, se existir.
        if isinstance(df_grafico.columns, pd.MultiIndex):
            df_grafico.columns = df_grafico.columns.get_level_values(0)
        # 2. AGORA, com nomes de colunas como strings, normaliza para minúsculas.
        df_grafico.columns = [col.lower() for col in df_grafico.columns]
        # 3. E finalmente, sanitiza o índice de data.
        df_grafico.index = pd.to_datetime(df_grafico.index).tz_localize(None)

        cols_to_validate = ['open', 'high', 'low', 'close']
        for col in cols_to_validate:
            if col in df_grafico.columns:
                df_grafico[col] = pd.to_numeric(
                    df_grafico[col], errors='coerce')
        df_grafico.dropna(subset=cols_to_validate, inplace=True)

        if df_grafico.empty:
            print(f"AVISO: DataFrame vazio após limpeza. Marcando como erro (-1).")
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = -1
            self.avancar_e_salvar()
            return

        self.df_grafico_atual = df_grafico.copy()

        try:
            self.fig, axlist = mpf.plot(
                self.df_grafico_atual, type='candle', style='charles', returnfig=True, figsize=(12, 8))
            self.ax = axlist[0]
            self.atualizar_marcacoes()  # Delega o desenho inicial das marcações
            canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
            self.canvas_widget = canvas
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"ERRO CRÍTICO ao plotar o gráfico: {e}")
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = -1
            self.avancar_e_salvar()

    # --- O RESTANTE DA CLASSE (JÁ ESTÁVEL) ---
    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in ['a', 'r']:
            label = 1 if key == 'a' else 0
            self.df_trabalho.loc[self.indice_atual,
                                 'data_inicio'] = self.data_inicio_ajustada
            self.df_trabalho.loc[self.indice_atual,
                                 'data_fim'] = self.data_fim_ajustada
            if pd.notna(self.data_cabeca_ajustada):
                self.df_trabalho.loc[self.indice_atual,
                                     'data_cabeca'] = self.data_cabeca_ajustada
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = label
            self.avancar_e_salvar()
        elif key == 'q':
            self.on_closing()

    def on_closing(self):
        print("Saindo e liberando recursos...")
        if self.fig:
            plt.close(self.fig)
        self.destroy()

    def avancar_e_salvar(self):
        self.df_trabalho.to_csv(
            self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Progresso salvo em '{self.arquivo_saida}'")
        proximos_indices = self.df_trabalho[self.df_trabalho['label_humano'].isnull(
        )].index
        if not proximos_indices.empty:
            self.indice_atual = proximos_indices[0]
            self.carregar_proximo_padrao()
        else:
            messagebox.showinfo(
                "Fim!", "Parabéns! Todos os padrões foram rotulados.")
            self.on_closing()

    def carregar_proximo_padrao(self):
        if self.indice_atual not in self.df_trabalho.index:
            messagebox.showinfo("Fim!", "Todos os padrões foram analisados.")
            self.on_closing()
            return
        self.padrao_atual_info = self.df_trabalho.loc[self.indice_atual]
        self.data_inicio_ajustada = pd.to_datetime(
            self.padrao_atual_info['data_inicio'])
        self.data_fim_ajustada = pd.to_datetime(
            self.padrao_atual_info['data_fim'])
        self.data_cabeca_ajustada = pd.to_datetime(
            self.padrao_atual_info['data_cabeca'], errors='coerce')
        self.plotar_grafico()
        self.atualizar_info_label()

    def atualizar_info_label(self):
        total, feitos = len(self.df_trabalho), len(
            self.df_trabalho.dropna(subset=['label_humano']))
        padrao = self.padrao_atual_info
        info_text = (f"Progresso: {feitos}/{total} | Padrão Índice: {self.indice_atual}\n"
                     f"Ativo: {padrao['ticker']} ({padrao['intervalo']}) | Tipo: {padrao['tipo_padrao']}\n"
                     f"Início: {self.data_inicio_ajustada.strftime('%Y-%m-%d %H:%M')}\n"
                     f"Fim:    {self.data_fim_ajustada.strftime('%Y-%m-%d %H:%M')}")
        if pd.notna(self.data_cabeca_ajustada):
            info_text += f"\nCabeça: {self.data_cabeca_ajustada.strftime('%Y-%m-%d %H:%M')}"
        self.info_label.config(text=info_text)


if __name__ == '__main__':
    ARQUIVO_ENTRADA = 'data/datasets/filtered/dataset_featured.csv'
    ARQUIVO_SAIDA = 'data/datasets/filtered/dataset_labeled.csv'
    app = LabelingToolHybrid(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)
    if app.winfo_exists():
        app.mainloop()
