# anotador_gui.py (v6.4 - Final com correção de renderização)
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
    def __init__(self, arquivo_entrada, arquivo_saida):
        super().__init__()

        self.title("Ferramenta de Labeling Otimizada (v6.4 - Final)")
        self.geometry("1300x900")

        self.arquivo_saida = arquivo_saida
        self.df_trabalho = None

        # --- Variáveis de estado ---
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

    def plotar_grafico(self):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        if self.fig:
            plt.close(self.fig)
        self.fig, self.ax, self.df_grafico_atual, self.canvas_widget, self.span_amarelo, self.linha_cabeca = [
            None] * 6

        padrao = self.padrao_atual_info
        data_inicio, data_fim, intervalo = self.data_inicio_ajustada, self.data_fim_ajustada, padrao[
            'intervalo']
        duracao = data_fim - data_inicio
        if intervalo == '1h':
            # Buffer mínimo de 1 dia ou 75% da duração
            buffer = max(pd.Timedelta(hours=16), duracao * 0.5)
        elif intervalo == '4h':
            # Buffer mínimo de 2 dias ou 75% da duração
            buffer = max(pd.Timedelta(days=2), duracao * 0.75)
        else:
            # Corrigido: atribuição do buffer e buffer mínimo de 2 semanas
            buffer = max(pd.Timedelta(weeks=2), duracao * 0.5)
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

        if isinstance(df_grafico.columns, pd.MultiIndex):
            df_grafico.columns = df_grafico.columns.get_level_values(0)
        df_grafico.columns = [col.lower() for col in df_grafico.columns]
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
            # --- MUDANÇA: CORREÇÃO NA ORDEM DE RENDERIZAÇÃO ---
            # 1. Cria a figura e os eixos base
            self.fig, axlist = mpf.plot(
                self.df_grafico_atual, type='candle', style='charles', returnfig=True, figsize=(12, 8))
            self.ax = axlist[0]

            # 2. Cria o 'Canvas' do Tkinter para exibir a figura
            canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
            self.canvas_widget = canvas

            # 3. Desenha o canvas na tela
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # 4. SÓ AGORA, com a "tela" pronta, desenhamos as marcações por cima
            self.atualizar_marcacoes()

        except Exception as e:
            print(f"ERRO CRÍTICO ao plotar o gráfico: {e}")
            self.df_trabalho.loc[self.indice_atual, 'label_humano'] = -1
            self.avancar_e_salvar()

    def atualizar_marcacoes(self):
        if self.ax is None:
            print("AVISO: Eixos do gráfico não encontrados. Ignorando atualização.")
            return

        if self.span_amarelo:
            self.span_amarelo.remove()
        if self.linha_cabeca:
            self.linha_cabeca.remove()
        self.span_amarelo, self.linha_cabeca = None, None

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

            # A chamada .draw() é crucial para o canvas exibir a mudança
            if self.canvas_widget:
                self.canvas_widget.draw()
        except Exception as e:
            print(
                f"AVISO: Não foi possível atualizar as marcações no gráfico. Erro: {e}")

    def ajustar_data(self, qual_data, direcao):
        intervalo = self.padrao_atual_info['intervalo']
        match = re.match(r"(\d+)(\w+)", intervalo)
        ajuste = pd.Timedelta(hours=4)
        if match:
            valor, unidade = int(match.groups()[0]), match.groups()[1].lower()
            if 'h' in unidade:
                ajuste = pd.Timedelta(hours=valor)
            elif 'd' in unidade:
                ajuste = pd.Timedelta(days=valor)
            elif 'wk' in unidade or 'w' in unidade:
                ajuste = pd.Timedelta(weeks=valor)
            elif 'mo' in unidade:
                ajuste = pd.Timedelta(weeks=valor * 4)
        fator = -1 if direcao == 'tras' else 1
        if qual_data == 'inicio':
            self.data_inicio_ajustada += (ajuste * fator)
        elif qual_data == 'fim':
            self.data_fim_ajustada += (ajuste * fator)
        elif qual_data == 'cabeca' and pd.notna(self.data_cabeca_ajustada):
            self.data_cabeca_ajustada += (ajuste * fator)
        self.atualizar_marcacoes()
        self.atualizar_info_label()

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
            try:
                intervalo_str = self.padrao_atual_info['intervalo']
                # Converte '4h', '1d' etc.
                timedelta_intervalo = pd.to_timedelta(intervalo_str)

                nova_duracao = (self.data_fim_ajustada -
                                self.data_inicio_ajustada) / timedelta_intervalo

                # Atualiza a coluna duracao_em_velas com o novo valor
                self.df_trabalho.loc[self.indice_atual,
                                     'duracao_em_velas'] = int(nova_duracao)
            except Exception as e:
                print(
                    f"AVISO: Não foi possível recalcular a duração. Erro: {e}")
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
    ARQUIVO_ENTRADA = 'data/datasets/filtered/dataset_featured_sequential.csv'
    ARQUIVO_SAIDA = 'data/datasets/filtered/dataset_labeled.csv'
    app = LabelingToolHybrid(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)
    if app.winfo_exists():
        app.mainloop()
