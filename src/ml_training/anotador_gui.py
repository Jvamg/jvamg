import tkinter as tk
from tkinter import messagebox
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class AnotadorApp:
    def __init__(self, root, arquivo_candidatos, arquivo_saida):
        self.root = root
        self.root.title(
            "Ferramenta de Anotação do Garimpeiro (v2.5 com Ajuste de Cabeça)")
        self.root.geometry("1200x850")

        self.arquivo_saida = arquivo_saida

        print("Carregando padrões candidatos...")
        try:
            self.df_candidatos = pd.read_csv(arquivo_candidatos, parse_dates=[
                                             'data_inicio', 'data_fim', 'data_cabeca'])
        except FileNotFoundError:
            messagebox.showerror(
                "Erro", f"Arquivo de candidatos '{arquivo_candidatos}' não encontrado!")
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("Erro ao ler CSV", str(e))
            self.root.destroy()
            return

        self.indice_atual = 0
        self.lista_aprovados = []

        # --- Layout da Interface ---
        self.frame_grafico = tk.Frame(root)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_inferior = tk.Frame(root)
        self.frame_inferior.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = tk.Label(
            self.frame_inferior, text="Carregando...", font=("Arial", 11), justify=tk.LEFT)  # Diminuí a fonte para caber mais info
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.frame_botoes_acao = tk.Frame(self.frame_inferior)
        self.frame_botoes_acao.pack(side=tk.RIGHT, padx=5)

        self.rejeitar_btn = tk.Button(
            self.frame_botoes_acao, text="❌ Rejeitar", command=self.rejeitar, font=("Arial", 12), bg="salmon")
        self.rejeitar_btn.pack(side=tk.RIGHT, padx=5)

        self.aprovar_btn = tk.Button(self.frame_botoes_acao, text="✅ Aprovar",
                                     command=self.aprovar, font=("Arial", 12), bg="lightgreen")
        self.aprovar_btn.pack(side=tk.RIGHT)

        self.frame_ajustes = tk.Frame(self.frame_inferior)
        self.frame_ajustes.pack(side=tk.RIGHT, padx=10)

        # Controles para a data de início
        tk.Label(self.frame_ajustes, text="Início:", font=(
            "Arial", 10)).grid(row=0, column=0, columnspan=2)
        tk.Button(self.frame_ajustes, text="<<", command=lambda: self.ajustar_data(
            'inicio', 'tras')).grid(row=1, column=0)
        tk.Button(self.frame_ajustes, text=">>", command=lambda: self.ajustar_data(
            'inicio', 'frente')).grid(row=1, column=1)

        # Controles para a data de fim
        tk.Label(self.frame_ajustes, text="Fim:", font=("Arial", 10)).grid(
            row=0, column=2, columnspan=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text="<<", command=lambda: self.ajustar_data(
            'fim', 'tras')).grid(row=1, column=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text=">>", command=lambda: self.ajustar_data(
            'fim', 'frente')).grid(row=1, column=3)

        # NOVO: Controles para a data da cabeça
        tk.Label(self.frame_ajustes, text="Cabeça:", font=("Arial", 10)).grid(
            row=0, column=4, columnspan=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text="<<", command=lambda: self.ajustar_data(
            'cabeca', 'tras')).grid(row=1, column=4, padx=(10, 0))
        tk.Button(self.frame_ajustes, text=">>", command=lambda: self.ajustar_data(
            'cabeca', 'frente')).grid(row=1, column=5)

        self.canvas = None
        self.carregar_proximo_padrao()

    # MUDANÇA: Atualizado para lidar com a 'cabeca'
    def ajustar_data(self, qual_data, direcao):
        ajuste = pd.Timedelta(hours=4)
        if qual_data == 'inicio':
            self.data_inicio_ajustada = self.data_inicio_ajustada - \
                ajuste if direcao == 'tras' else self.data_inicio_ajustada + ajuste
        elif qual_data == 'fim':
            self.data_fim_ajustada = self.data_fim_ajustada - \
                ajuste if direcao == 'tras' else self.data_fim_ajustada + ajuste
        elif qual_data == 'cabeca':
            # Só ajusta se a cabeça existir
            if self.data_cabeca_ajustada:
                self.data_cabeca_ajustada = self.data_cabeca_ajustada - \
                    ajuste if direcao == 'tras' else self.data_cabeca_ajustada + ajuste

        self.atualizar_info_label()
        self.plotar_grafico()

    def plotar_grafico(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        data_inicio_contexto = self.data_inicio_ajustada - pd.Timedelta(days=3)
        data_fim_contexto = self.data_fim_ajustada + pd.Timedelta(days=5)

        print(
            f"Buscando dados para o gráfico de {data_inicio_contexto.date()} a {data_fim_contexto.date()}...")
        df_grafico = yf.download(
            tickers='BTC-USD', start=data_inicio_contexto, end=data_fim_contexto,
            interval='4h', progress=False, auto_adjust=True)

        if df_grafico.empty:
            print("Não foi possível buscar dados. Pulando...")
            self.root.after(100, self.rejeitar)
            return

        if isinstance(df_grafico.columns, pd.MultiIndex):
            df_grafico.columns = df_grafico.columns.get_level_values(0)
        df_grafico.index = pd.to_datetime(df_grafico.index).tz_localize(None)
        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_convert:
            if col in df_grafico.columns:
                df_grafico[col] = pd.to_numeric(
                    df_grafico[col], errors='coerce')
        df_grafico.dropna(subset=cols_to_convert, inplace=True)
        if df_grafico.empty:
            print("DataFrame vazio após limpeza. Pulando...")
            self.root.after(100, self.rejeitar)
            return
        df_grafico[cols_to_convert] = df_grafico[cols_to_convert].astype(float)

        fig, axlist = mpf.plot(df_grafico, type='candle',
                               style='charles', returnfig=True, figsize=(10, 6))

        ax = axlist[0]
        ax.set_title(f"Analisar Padrão: {self.tipo_padrao}")

        try:
            start_pos_idx = (
                df_grafico.index - self.data_inicio_ajustada).to_series().abs().argmin()
            end_pos_idx = (df_grafico.index -
                           self.data_fim_ajustada).to_series().abs().argmin()
            ax.axvspan(start_pos_idx, end_pos_idx, color='yellow', alpha=0.3)

            # MUDANÇA: Usa a data da cabeça AJUSTADA para desenhar a linha
            if self.tipo_padrao in ['OCO', 'OCOI'] and self.data_cabeca_ajustada:
                head_pos_idx = (
                    df_grafico.index - self.data_cabeca_ajustada).to_series().abs().argmin()
                ax.axvline(x=head_pos_idx, color='dodgerblue',
                           linestyle='--', linewidth=1.2, label='Cabeça (Head)')
                ax.legend()
        except Exception as e:
            print(f"Aviso: Não foi possível desenhar marcações. Erro: {e}")

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def carregar_proximo_padrao(self):
        if self.indice_atual >= len(self.df_candidatos):
            messagebox.showinfo("Fim!", "Todos os padrões foram analisados!")
            self.salvar_e_sair()
            return

        padrao_atual = self.df_candidatos.iloc[self.indice_atual]
        self.tipo_padrao = padrao_atual['tipo_padrao']

        self.data_inicio_original = padrao_atual['data_inicio']
        self.data_fim_original = padrao_atual['data_fim']
        self.data_inicio_ajustada = self.data_inicio_original
        self.data_fim_ajustada = self.data_fim_original

        # NOVO: Lógica para carregar a data da cabeça original e a ajustável
        self.data_cabeca_original = None
        self.data_cabeca_ajustada = None
        if 'data_cabeca' in padrao_atual and pd.notna(padrao_atual['data_cabeca']):
            self.data_cabeca_original = padrao_atual['data_cabeca']
            self.data_cabeca_ajustada = self.data_cabeca_original

        self.atualizar_info_label()
        self.plotar_grafico()

    def atualizar_info_label(self):
        # MUDANÇA: Adiciona a data da cabeça na label de informação
        info_text = (f"Padrão {self.indice_atual + 1}/{len(self.df_candidatos)} | Tipo: {self.tipo_padrao}\n"
                     f"Início: {self.data_inicio_ajustada.strftime('%Y-%m-%d %H:%M')} | "
                     f"Fim: {self.data_fim_ajustada.strftime('%Y-%m-%d %H:%M')}")
        if self.data_cabeca_ajustada:
            info_text += f"\nCabeça: {self.data_cabeca_ajustada.strftime('%Y-%m-%d %H:%M')}"

        self.info_label.config(text=info_text)

    def aprovar(self):
        print(
            f"Padrão APROVADO: {self.tipo_padrao} em {self.data_inicio_original.date()}")

        # MUDANÇA: Salva a data da cabeça ajustada
        padrao_aprovado = self.df_candidatos.iloc[self.indice_atual].to_dict()
        padrao_aprovado['data_inicio'] = self.data_inicio_ajustada
        padrao_aprovado['data_fim'] = self.data_fim_ajustada
        # Salva a data ajustada
        padrao_aprovado['data_cabeca'] = self.data_cabeca_ajustada
        padrao_aprovado['ajustado_manualmente'] = 'Sim'

        self.lista_aprovados.append(padrao_aprovado)
        self.indice_atual += 1
        self.carregar_proximo_padrao()

    def rejeitar(self):
        print(
            f"Padrão REJEITADO: {self.tipo_padrao} em {self.data_inicio_original.date()}")
        self.indice_atual += 1
        self.carregar_proximo_padrao()

    def salvar_e_sair(self):
        if self.lista_aprovados:
            df_aprovados = pd.DataFrame(self.lista_aprovados)
            # Garante que a ordem das colunas seja consistente
            colunas_principais = ['tipo_padrao', 'data_inicio',
                                  'data_fim', 'data_cabeca', 'ajustado_manualmente']
            # Pega outras colunas que possam existir e as adiciona no final
            outras_colunas = [
                col for col in df_aprovados.columns if col not in colunas_principais]
            df_aprovados = df_aprovados[colunas_principais + outras_colunas]

            df_aprovados.to_csv(self.arquivo_saida, index=False,
                                float_format='%.2f',
                                date_format='%Y-%m-%d %H:%M:%S')
            print(
                f"Arquivo de padrões verificados salvo como '{self.arquivo_saida}'!")
        self.root.destroy()


if __name__ == '__main__':
    ARQUIVO_CANDIDATOS = 'todos_os_padroes_btc_automatico.csv'
    ARQUIVO_SAIDA_VERIFICADO = 'dataset_verificado.csv'

    root = tk.Tk()
    app = AnotadorApp(root, ARQUIVO_CANDIDATOS, ARQUIVO_SAIDA_VERIFICADO)
    root.mainloop()
