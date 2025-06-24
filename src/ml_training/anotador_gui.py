# labeling_tool_final.py
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

        self.title(
            "Ferramenta de Labeling Híbrida (Ajuste + Teclado) (v6.0 - Estável)")
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

        # 1. Tenta configurar o DataFrame. Se falhar, encerra a aplicação.
        if not self.setup_dataframe(arquivo_entrada, arquivo_saida):
            return

        # 2. Encontra o próximo padrão a ser anotado
        indices_para_anotar = self.df_trabalho[self.df_trabalho['label_humano'].isnull(
        )].index
        if indices_para_anotar.empty:
            messagebox.showinfo(
                "Concluído", "Parabéns! Todos os padrões já foram rotulados.")
            self.destroy()
            return
        self.indice_atual = indices_para_anotar[0]

        # 3. Configura a UI e os eventos
        self._setup_ui()
        self.bind('<Key>', self.on_key_press)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 4. Carrega o primeiro padrão
        self.carregar_proximo_padrao()

    def _setup_ui(self):
        """Configura a interface gráfica do usuário."""
        self.frame_grafico = tk.Frame(self)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.frame_inferior = tk.Frame(self)
        self.frame_inferior.pack(fill=tk.X, padx=10, pady=5)
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
        """Carrega o dataset e retorna True em sucesso, False em falha."""
        try:
            if os.path.exists(arquivo_saida):
                print(
                    f"Arquivo de progresso '{arquivo_saida}' encontrado. Carregando...")
                self.df_trabalho = pd.read_csv(arquivo_saida)
            else:
                print(
                    f"Nenhum progresso salvo. Carregando de '{arquivo_entrada}'.")
                self.df_trabalho = pd.read_csv(arquivo_entrada)

            if 'label_humano' not in self.df_trabalho.columns:
                self.df_trabalho['label_humano'] = np.nan
            return True
        except FileNotFoundError:
            # Esconde a janela principal antes de mostrar o erro para evitar bugs
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

    def on_key_press(self, event):
        """Processa os eventos de teclado para labeling."""
        key = event.keysym.lower()
        if key in ['a', 'r']:
            label = 1 if key == 'a' else 0
            print(
                f" -> RÓTULO DEFINIDO: {'APROVADO' if label == 1 else 'REJEITADO'}")
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
        """Garante que a memória seja liberada ao fechar a janela."""
        print("Saindo e liberando recursos...")
        if self.fig:
            plt.close(self.fig)
        self.destroy()

    def ajustar_data(self, qual_data, direcao):
        """Move as datas de início, fim ou cabeça para ajuste fino."""
        intervalo = self.padrao_atual_info['intervalo']
        match = re.match(r"(\d+)([hHdD])", intervalo)
        ajuste = pd.Timedelta(hours=4)  # Fallback
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
        self.plotar_grafico()
        self.atualizar_info_label()

    def avancar_e_salvar(self):
        """Salva o progresso e avança para o próximo padrão não rotulado."""
        try:
            self.df_trabalho.to_csv(
                self.arquivo_saida, index=False, date_format='%Y-%m-%d %H:%M:%S')
            print(f"Progresso salvo em '{self.arquivo_saida}'")
        except Exception as e:
            messagebox.showerror(
                "Erro ao Salvar", f"Não foi possível salvar o progresso: {e}")
            self.on_closing()
            return

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
        """Prepara os dados do próximo padrão a ser exibido."""
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

    def plotar_grafico(self):
        """Função central que busca dados, limpa e plota o gráfico."""
        # Limpeza de recursos da iteração anterior para evitar memory leaks
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)

        padrao = self.padrao_atual_info
        data_inicio, data_fim, intervalo = self.data_inicio_ajustada, self.data_fim_ajustada, padrao[
            'intervalo']

        # Lógica de buffer para contexto visual
        duracao = data_fim - data_inicio
        if intervalo in ['1h', '4h']:
            buffer = max(pd.Timedelta(days=5), duracao * 1.5)
        elif intervalo == '1d':
            buffer = max(pd.Timedelta(weeks=4), duracao * 1.0)
        else:
            buffer = duracao * 0.5
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

        # Sanitização do índice e dos dados
        df_grafico.index = pd.to_datetime(df_grafico.index).tz_localize(None)
        if isinstance(df_grafico.columns, pd.MultiIndex):
            df_grafico.columns = df_grafico.columns.get_level_values(0)

        for col in ['Open', 'High', 'Low', 'Close']:
            df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce')
        df_grafico.dropna(
            subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        if df_grafico.empty:
            print(f"AVISO: DataFrame vazio após limpeza. Marcando como erro (-1).")
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = -1
            self.avancar_e_salvar()
            return

        # Plotagem
        try:
            self.fig, axlist = mpf.plot(
                df_grafico, type='candle', style='charles', returnfig=True, figsize=(12, 8))
            ax = axlist[0]
            start_pos_idx = (df_grafico.index -
                             data_inicio).to_series().abs().argmin()
            end_pos_idx = (df_grafico.index -
                           data_fim).to_series().abs().argmin()
            ax.axvspan(start_pos_idx, end_pos_idx, color='yellow', alpha=0.3)
            if pd.notna(self.data_cabeca_ajustada):
                head_pos_idx = (
                    df_grafico.index - self.data_cabeca_ajustada).to_series().abs().argmin()
                ax.axvline(x=head_pos_idx, color='dodgerblue',
                           linestyle='--', linewidth=1.2)
        except Exception as e:
            print(f"ERRO CRÍTICO ao plotar o gráfico: {e}")
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = -1
            self.avancar_e_salvar()
            return

        canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        self.canvas_widget = canvas
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def atualizar_info_label(self):
        """Atualiza o texto da UI com as informações do padrão atual."""
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
    # Certifique-se de que o caminho para o arquivo de entrada está correto
    ARQUIVO_ENTRADA = 'data/datasets/filtered/dataset_featured.csv'
    ARQUIVO_SAIDA = 'data/datasets/filtered/dataset_labeled.csv'

    app = LabelingToolHybrid(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)
    # A verificação de falha no __init__ impede que o mainloop seja chamado para uma janela já destruída
    if app.winfo_exists():
        app.mainloop()
